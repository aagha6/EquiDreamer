import functools
import re

import jax
import jax.numpy as jnp
import equinox as eqx
import escnn_jax.nn as nn
import numpy as np
from tensorflow_probability.substrates import jax as tfp

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj

cast = jaxutils.cast_to_compute
eqx_conv = functools.partial(nj.ESCNNModule, eqx.nn.Conv2d)
econv_module = functools.partial(nj.ESCNNModule, nn.R2Conv)
pooling_module = functools.partial(nj.ESCNNModule, nn.GroupPooling)


class RSSM(nj.Module):

    def __init__(
        self,
        key,
        act_dim,
        grp,
        num_prototypes=1024,
        proto=32,
        deter=1024,
        stoch=32,
        classes=32,
        unroll=False,
        sim_norm=False,
        sim_norm_dim=8,
        initial="learned",
        unimix=0.01,
        action_clip=1.0,
        conv_gru=False,
        equiv=False,
        embed_size=None,
        cup_catch=False,
        **kw,
    ):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self.conv_gru = conv_gru
        self._kw = kw
        self._act_dim = act_dim
        self._num_prototypes = num_prototypes
        self._proto = proto
        self._warm_up = 1
        self._temperature = 0.1
        self._sinkhorn_eps = 0.05
        self._sinkhorn_iters = 3
        self._inputs = Input(["stoch", "deter"], dims="deter")
        self._cup_catch = cup_catch
        self._sim_norm = sim_norm
        self._sim_norm_dim = sim_norm_dim
        self._equiv = equiv
        if self.conv_gru and self._equiv:
            raise ValueError("both can't be True")
        if self._classes and self._sim_norm:
            raise ValueError("both can't be True")
        if self._equiv:
            assert embed_size is not None
            self.embed_size = embed_size
            self._grp = grp
            self._factor = self._grp.grp_act.regular_repr.size // self._grp.scaler
            self.init_equiv_nets(key)

    def init_equiv_nets(self, key):
        units = self._kw["units"] // self._grp.scaler
        stoch = self._stoch // self._grp.scaler
        deter = self._deter // self._grp.scaler
        gspace = self._grp.grp_act
        if self._classes:
            self._field_type_stoch = nn.FieldType(
                gspace, stoch * self._classes * [gspace.regular_repr]
            )
        else:
            self._field_type_stoch = nn.FieldType(gspace, stoch * [gspace.regular_repr])
        self._field_type_deter = nn.FieldType(gspace, deter * [gspace.regular_repr])
        self._field_type_embed = nn.FieldType(gspace, units * [gspace.regular_repr])
        self._field_type_gru_in = nn.FieldType(
            gspace, (deter + units) * [gspace.regular_repr]
        )
        self._field_type_gru_out = nn.FieldType(
            gspace, 3 * deter * [gspace.regular_repr]
        )
        self.sign_mat = None
        if gspace.fibergroup.name == "C2":
            if self._cup_catch:
                self._field_type_act = nn.FieldType(
                    gspace, [gspace.regular_repr] + [gspace.trivial_repr]
                )
            else:
                self._field_type_act = nn.FieldType(
                    gspace,
                    self._act_dim * [gspace.regular_repr],
                )

        elif gspace.fibergroup.name == "D2":
            # Reacher
            self._field_type_act = nn.FieldType(
                gspace,
                self._act_dim * [gspace.quotient_repr((None, gspace.rotations_order))],
            )
        else:
            raise NotImplementedError("only implemented for groups C2,D2")
        self._field_type_img_in = self._field_type_stoch + self._field_type_act
        self._field_type_inf_in = nn.FieldType(
            gspace, (deter + self.embed_size) * [gspace.regular_repr]
        )
        (
            img_in_key,
            img_out_key,
            obs_out_key,
            stoch_mean_key_img,
            stoch_mean_key_obs,
            gru_key,
            feat_proj_key,
        ) = jax.random.split(key, 7)
        if self._num_prototypes:
            if self._classes:
                self._field_type_feat_proj = nn.FieldType(
                    gspace, (stoch * self._classes + deter) * [gspace.regular_repr]
                )
            else:
                self._field_type_feat_proj = nn.FieldType(
                    gspace, (stoch + deter) * [gspace.regular_repr]
                )
            self._field_type_proto = nn.FieldType(
                gspace, self._proto * [gspace.regular_repr]
            )
            self.init_feat_proj = nn.R2Conv(
                in_type=self._field_type_feat_proj,
                out_type=self._field_type_proto,
                kernel_size=1,
                key=feat_proj_key,
            )
        self.init_img_in = nn.R2Conv(
            in_type=self._field_type_img_in,
            out_type=self._field_type_embed,
            kernel_size=1,
            key=img_in_key,
        )
        self.init_img_out = nn.R2Conv(
            in_type=self._field_type_deter,
            out_type=self._field_type_embed,
            kernel_size=1,
            key=img_out_key,
        )
        self.init_obs_out = nn.R2Conv(
            in_type=self._field_type_inf_in,
            out_type=self._field_type_embed,
            kernel_size=1,
            key=obs_out_key,
        )
        self.init_stoch_mean = {
            "img_stats": nn.R2Conv(
                in_type=self._field_type_embed,
                out_type=(
                    self._field_type_stoch
                    if self._classes
                    else self._field_type_stoch + self._field_type_stoch
                ),
                kernel_size=1,
                key=stoch_mean_key_img,
            ),
            "obs_stats": nn.R2Conv(
                in_type=self._field_type_embed,
                out_type=(
                    self._field_type_stoch
                    if self._classes
                    else self._field_type_stoch + self._field_type_stoch
                ),
                kernel_size=1,
                key=stoch_mean_key_obs,
            ),
        }
        self._embed_group_pooling = pooling_module(
            self._field_type_embed, name="embed_group_pooling"
        )
        gru_kw = {
            "in_type": self._field_type_gru_in,
            "out_type": self._field_type_gru_out,
            "kernel_size": 1,
            "stride": 1,
            "key": gru_key,
        }
        self.init_gru_cell = nn.R2Conv(**gru_kw)

    def initial(self, bs):
        if self._equiv:
            stoch = self._stoch * self._factor
            deter = self._deter * self._factor
        else:
            stoch = self._stoch
            deter = self._deter
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, deter], f32),
                logit=jnp.zeros([bs, stoch, self._classes], f32),
                stoch=jnp.zeros([bs, stoch, self._classes], f32),
            )
        else:
            state = dict(
                mean=jnp.zeros([bs, stoch], f32),
                std=jnp.ones([bs, stoch], f32),
                stoch=jnp.zeros([bs, stoch], f32),
                deter=jnp.zeros([bs, deter], f32),
            )
        if self._initial == "zeros":
            return cast(state)
        elif self._initial == "learned":
            deter = self.get("initial", jnp.zeros, state["deter"][0].shape, f32)
            state["deter"] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state["stoch"] = self.get_stoch(cast(state["deter"]))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll, modify=True)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        if self._classes:
            logit = state["logit"].astype(f32)
            if self._equiv:
                logit = jnp.moveaxis(logit, -1, -2)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            mean = state["mean"].astype(f32)
            std = state["std"].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)

    def obs_step(self, prev_state, prev_action, embed, is_first):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(
                self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action))
            )
        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action)
        )
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state,
            self.initial(len(is_first)),
        )
        prior = self.img_step(prev_state, prev_action)
        x = jnp.concatenate([prior["deter"], embed], -1)
        if self._equiv:
            x = self.get(
                "obs_out",
                EquivLinear,
                **{
                    "net": self.init_obs_out,
                    "in_type": self._field_type_inf_in,
                    "out_type": self._field_type_embed,
                    "norm": self._kw["norm"],
                    "act": "equiv_relu",
                },
            )(x)
        else:
            x = self.get("obs_out", Linear, **self._kw)(x)
        stats = self._stats("obs_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        if self._classes and self._equiv:
            stoch = jnp.moveaxis(stoch, -1, -2)
        if self._sim_norm:
            shape = stoch.shape
            stoch = stoch.reshape(shape[:-1] + (-1, self._sim_norm_dim))
            stoch = jax.nn.softmax(stoch, -1)
            stoch = jnp.reshape(stoch, shape)
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_stoch = prev_state["stoch"]
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(
                self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action))
            )
        if self._classes:
            if self._equiv:
                n_stoch = self._stoch * self._factor
            else:
                n_stoch = self._stoch
            shape = prev_stoch.shape[:-2] + (n_stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)
        if self._equiv:
            if self._cup_catch:
                act = prev_action @ jnp.array(
                    [[1, -1, 0], [0, 0, 1]], dtype=jnp.float32
                )
            else:
                act = jnp.stack([prev_action, -prev_action], -1).reshape(
                    prev_action.shape[:-1] + (-1,)
                )

            prev_stoch = nn.GeometricTensor(
                prev_stoch[:, :, jnp.newaxis, jnp.newaxis], self._field_type_stoch
            )
            act = nn.GeometricTensor(
                act[:, :, jnp.newaxis, jnp.newaxis], self._field_type_act
            )
            x = nn.tensor_directsum([prev_stoch, act]).tensor.mean(-1).mean(-1)
            x = self.get(
                "img_in",
                EquivLinear,
                **{
                    "net": self.init_img_in,
                    "in_type": self._field_type_img_in,
                    "out_type": self._field_type_embed,
                    "norm": self._kw["norm"],
                    "act": "equiv_relu",
                },
            )(x)
        else:
            x = jnp.concatenate([prev_stoch, prev_action], -1)
            x = self.get("img_in", Linear, **self._kw)(x)
        if self.conv_gru:
            x, deter = self._conv_gru(x, prev_state["deter"])
        elif self._equiv:
            x, deter = self._equiv_gru(x, prev_state["deter"])
        else:
            x, deter = self._gru(x, prev_state["deter"])
        if self._equiv:
            x = self.get(
                "img_out",
                EquivLinear,
                **{
                    "net": self.init_img_out,
                    "in_type": self._field_type_deter,
                    "out_type": self._field_type_embed,
                    "norm": self._kw["norm"],
                    "act": "equiv_relu",
                },
            )(x)
        else:
            x = self.get("img_out", Linear, **self._kw)(x)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        if self._classes and self._equiv:
            stoch = jnp.moveaxis(stoch, -1, -2)
        if self._sim_norm:
            shape = stoch.shape
            stoch = stoch.reshape(shape[:-1] + (-1, self._sim_norm_dim))
            stoch = jax.nn.softmax(stoch, -1)
            stoch = jnp.reshape(stoch, shape)
        prior = {"stoch": stoch, "deter": deter, **stats}
        return cast(prior)

    def get_stoch(self, deter):
        if self._equiv:
            x = self.get(
                "img_out",
                EquivLinear,
                **{
                    "net": self.init_img_out,
                    "in_type": self._field_type_deter,
                    "out_type": self._field_type_embed,
                    "norm": self._kw["norm"],
                    "act": "equiv_relu",
                },
            )(deter)
        else:
            x = self.get("img_out", Linear, **self._kw)(deter)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def _conv_gru(self, x, deter):
        x = jnp.concatenate([deter, x], -1)
        kw = {
            "in_channels": self._deter + self._kw["units"],
            "out_channels": self._deter * 3,
            "kernel_size": 1,
            "stride": 1,
            "key": nj.rng(),
        }
        x = jax.vmap(self.get("gru", eqx_conv, **kw))(x[:, :, jnp.newaxis, jnp.newaxis])
        x = self.get("norm", Norm, "layer")(x.mean(-1).mean(-1))
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _equiv_gru(self, x, deter):
        x = jnp.concatenate([x, deter], 1)
        gru_out = (
            self.get(
                "gru",
                EquivGRUCell,
                **{
                    "net": self.init_gru_cell,
                    "in_type": self._field_type_gru_in,
                    "out_type": self._field_type_gru_out,
                    "norm": self._kw["norm"],
                    "act": "none",
                },
            )(x)
            .tensor.mean(-1)
            .mean(-1)
        )
        reset, cand, update = jnp.split(gru_out, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _gru(self, x, deter):
        x = jnp.concatenate([deter, x], -1)
        kw = {**self._kw, "act": "none", "units": 3 * self._deter}
        x = self.get("gru", Linear, **kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _stats(self, name, x):
        if self._classes:
            if self._equiv:
                flat_logits = self.get(
                    f"{name}",
                    EquivLinear,
                    **{
                        "net": self.init_stoch_mean[name],
                        "in_type": self._field_type_embed,
                        "out_type": self._field_type_stoch,
                        "norm": "none",
                        "act": "none",
                    },
                )(x)
                logit = jnp.stack(
                    jnp.split(flat_logits, self._stoch * self._factor, -1), 1
                )
                logit = logit.reshape(x.shape[:-1] + logit.shape[-2:])
            else:
                x = self.get(name, Linear, self._stoch * self._classes)(x)
                logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {"logit": logit}
            return stats
        else:
            if self._equiv:
                x = (
                    self.get(
                        f"{name}",
                        EquivGRUCell,
                        **{
                            "net": self.init_stoch_mean[name],
                            "in_type": self._field_type_embed,
                            "out_type": self._field_type_stoch + self._field_type_stoch,
                            "act": "none",
                        },
                    )(x)
                    .tensor.mean(-1)
                    .mean(-1)
                )
                mean, std = jnp.split(x, 2, -1)
            else:
                x = self.get(name, Linear, 2 * self._stoch)(x)
                mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {"mean": mean, "std": std}

    def _mask(self, value, mask):
        return jnp.einsum("b...,b->b...", value, mask.astype(value.dtype))

    def sinkhorn(self, scores):
        shape = scores.shape
        K = shape[0]
        scores = jnp.reshape(scores, [-1])
        log_Q = jax.nn.log_softmax(scores / self._sinkhorn_eps, axis=0)
        log_Q = jnp.reshape(log_Q, [K, -1])
        N = log_Q.shape[1]
        for _ in range(self._sinkhorn_iters):
            log_row_sums = jax.scipy.special.logsumexp(log_Q, axis=1, keepdims=True)
            log_Q = log_Q - log_row_sums - jnp.log(K)
            log_col_sums = jax.scipy.special.logsumexp(log_Q, axis=0, keepdims=True)
            log_Q = log_Q - log_col_sums - jnp.log(N)
        log_Q = log_Q + jnp.log(N)
        Q = jnp.exp(log_Q)
        return jnp.reshape(Q, shape)

    def get_prob_and_target(self, prototypes, obs_proj, ema_proj, feat_proj, B, T):
        obs_norm = jnp.linalg.norm(obs_proj, axis=-1, ord=2)
        feat_norm = jnp.linalg.norm(feat_proj, axis=-1, ord=2)

        obs_proj = jaxutils.l2_normalize(obs_proj, axis=-1)
        ema_proj = jaxutils.l2_normalize(ema_proj, axis=-1)
        feat_proj = jaxutils.l2_normalize(feat_proj, axis=-1)

        obs_scores = jnp.linalg.matmul(prototypes, obs_proj.T)
        ema_scores = jnp.linalg.matmul(prototypes, ema_proj.T)
        feat_scores = jnp.linalg.matmul(prototypes, feat_proj.T)

        obs_scores = jnp.reshape(obs_scores, [self._num_prototypes, B, T])
        ema_scores = jnp.reshape(ema_scores, [self._num_prototypes, B, T])
        feat_scores = jnp.reshape(feat_scores, [self._num_prototypes, B, T])

        obs_scores = obs_scores[..., self._warm_up :]
        obs_logits = jax.nn.log_softmax(obs_scores / self._temperature, axis=0)
        obs_logits_1, obs_logits_2 = jnp.split(obs_logits, 2, axis=1)

        ema_scores = ema_scores[:, :, self._warm_up :]
        ema_scores_1, ema_scores_2 = jnp.split(ema_scores, 2, axis=1)

        ema_targets_1 = jax.lax.stop_gradient(self.sinkhorn(ema_scores_1))
        ema_targets_2 = jax.lax.stop_gradient(self.sinkhorn(ema_scores_2))
        ema_targets = jnp.concat([ema_targets_1, ema_targets_2], axis=1)

        feat_scores = jnp.reshape(feat_scores, [self._num_prototypes, B, T])
        feat_scores = feat_scores[:, :, self._warm_up :]
        feat_logits = jax.nn.log_softmax(feat_scores / self._temperature, axis=0)

        swav_loss = -0.5 * jnp.mean(
            jnp.sum(ema_targets_2 * obs_logits_1, axis=0)
        ) - 0.5 * jnp.mean(jnp.sum(ema_targets_1 * obs_logits_2, axis=0))
        temp_loss = -jnp.mean(jnp.sum(ema_targets * feat_logits, axis=0))
        norm_loss = +1.0 * jnp.mean(jnp.square(obs_norm - 1)) + 1.0 * jnp.mean(
            jnp.square(feat_norm - 1)
        )

        return swav_loss, temp_loss, norm_loss

    def proto_loss(self, post, obs_proj, ema_proj):
        prototypes = self.get(
            "prototypes",
            Initializer("unit_normal"),
            (self._num_prototypes, self._proto),
        )
        prototypes = jaxutils.l2_normalize(prototypes, axis=-1)
        prototypes = self.put("prototypes", prototypes)

        B, T = obs_proj.shape[:2]
        if self._equiv:
            obs_proj = jnp.reshape(obs_proj, [B * T, -1])
            obs_proj = obs_proj.reshape(
                [obs_proj.shape[0], self._proto, self._grp.grp_act.regular_repr.size]
            ).transpose(0, 2, 1)
        else:
            obs_proj = jnp.reshape(obs_proj, [B * T, self._proto])

        if self._equiv:
            ema_proj = jnp.reshape(ema_proj, [B * T, -1])
            ema_proj = ema_proj.reshape(
                [ema_proj.shape[0], self._proto, self._grp.grp_act.regular_repr.size]
            ).transpose(0, 2, 1)
        else:
            ema_proj = jnp.reshape(ema_proj, [B * T, self._proto])

        feat = self._inputs(post)
        if self._equiv:
            feat_proj = self.get(
                "feat_proj",
                EquivLinear,
                **{
                    "net": self.init_feat_proj,
                    "in_type": self._field_type_feat_proj,
                    "out_type": self._field_type_proto,
                    "norm": "none",
                    "act": "none",
                },
            )(feat.reshape([-1] + list(feat.shape[2:])))
            feat_proj = feat_proj.reshape(post["deter"].shape[:2] + (-1,))
        else:
            feat_proj = self.get("feat_proj", Linear, **{"units": self._proto})(feat)

        if self._equiv:
            feat_proj = jnp.reshape(feat_proj, [B * T, -1])
            feat_proj = feat_proj.reshape(
                [feat_proj.shape[0], self._proto, self._grp.grp_act.regular_repr.size]
            ).transpose(0, 2, 1)
        else:
            feat_proj = jnp.reshape(feat_proj, [B * T, self._proto])

        if self._equiv:
            swav_loss, temp_loss, norm_loss = jax.vmap(
                self.get_prob_and_target, [None, 1, 1, 1, None, None]
            )(prototypes, obs_proj, ema_proj, feat_proj, B, T)
        else:
            swav_loss, temp_loss, norm_loss = self.get_prob_and_target(
                prototypes, obs_proj, ema_proj, feat_proj, B, T
            )

        losses = {
            "swav": swav_loss.mean(),
            "temp": temp_loss.mean(),
            "norm": norm_loss.mean(),
        }
        return losses

    def dyn_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == "logprob":
            loss = -self.get_dist(prior).log_prob(sg(post["stoch"]))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        elif impl == "uniform":
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
        elif impl == "entropy":
            loss = -self.get_dist(post).entropy()
        elif impl == "none":
            loss = jnp.zeros(post["deter"].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss


class MultiEncoder(nj.Module):

    def __init__(
        self,
        shapes,
        key,
        grp,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        resize="stride",
        symlog_inputs=False,
        minres=4,
        **kw,
    ):
        excluded = ("is_first", "is_last")
        shapes = {
            k: v
            for k, v in shapes.items()
            if (k not in excluded and not k.startswith("log_"))
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if (len(v) in (1, 2) and re.match(mlp_keys, k))
        }
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "name": "cnn"}
        mlp_kw = {**kw, "symlog_inputs": symlog_inputs, "name": "mlp"}
        if cnn == "resnet":
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
        elif cnn == "equiv":
            self._cnn = EquivImageEncoder(cnn_depth, grp=grp, key=key, **cnn_kw)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist="none", **mlp_kw)

    def __call__(self, data):
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[: -len(some_shape)]
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()
        }
        outputs = []
        if self.cnn_shapes:
            inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
            output = self._cnn(inputs)
            output = output.reshape((output.shape[0], -1))
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [
                data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
                for k in self.mlp_shapes
            ]
            inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
            inputs = jaxutils.cast_to_compute(inputs)
            outputs.append(self._mlp(inputs))
        outputs = jnp.concatenate(outputs, -1)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        return outputs


class MultiDecoder(nj.Module):

    def __init__(
        self,
        shapes,
        key,
        grp,
        inputs=["tensor"],
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        image_dist="mse",
        vector_dist="mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        deter=None,
        stoch=None,
        **kw,
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        }
        self.mlp_shapes = {
            k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        }
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "sigmoid": cnn_sigmoid}
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}
        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(
                    shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name="cnn"
                )
            elif cnn == "equiv":
                assert deter is not None and stoch is not None
                self._cnn = EquivImageDecoder(
                    key=key,
                    grp=grp,
                    deter=deter,
                    cnn_depth=cnn_depth,
                    stoch=stoch,
                    **cnn_kw,
                    name="cnn",
                )
            else:
                raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(
                self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name="mlp"
            )
        self._inputs = Input(inputs, dims="deter")
        self._image_dist = image_dist

    def __call__(self, inputs, drop_loss_indices=None):
        features = self._inputs(inputs)
        dists = {}
        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat = feat.reshape([-1, feat.shape[-1]])
            output = self._cnn(flat)
            output = output.reshape(feat.shape[:-1] + output.shape[1:])
            split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
            means = jnp.split(output, split_indices, -1)
            dists.update(
                {
                    key: self._make_image_dist(key, mean)
                    for (key, shape), mean in zip(self.cnn_shapes.items(), means)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.astype(f32)
        if self._image_dist == "normal":
            return tfd.Independent(tfd.Normal(mean, 1), 3)
        if self._image_dist == "mse":
            return jaxutils.MSEDist(mean, 3, "sum")
        raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):

    def __init__(self, depth, blocks, resize, minres, **kw):
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth = self._depth
        x = jaxutils.cast_to_compute(x) - 0.5
        # print(x.shape)
        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, **kw)(x)
            elif self._resize == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, **kw)(x)
            elif self._resize == "mean":
                N, H, W, D = x.shape
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
            elif self._resize == "max":
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = jax.lax.reduce_window(
                    x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "same"
                )
            else:
                raise NotImplementedError(self._resize)
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth *= 2
        if self._blocks:
            x = get_act(self._kw["act"])(x)
        x = x.reshape((x.shape[0], -1))
        # print(x.shape)
        return x


class EquivImageEncoder(nj.Module):

    def __init__(self, depth, grp, key, **kw):
        gspace = grp.grp_act
        depth = depth // grp.scaler
        self.feat_type_in = nn.FieldType(gspace, 3 * [gspace.trivial_repr])
        self.feat_type_out1 = nn.FieldType(gspace, depth * [gspace.regular_repr])
        depth *= 2
        self.feat_type_out2 = nn.FieldType(gspace, depth * [gspace.regular_repr])
        depth *= 2
        self.feat_type_out3 = nn.FieldType(gspace, depth * [gspace.regular_repr])
        depth *= 2
        self.feat_type_out4 = nn.FieldType(gspace, depth * [gspace.regular_repr])
        depth *= 2
        self.feat_type_out5 = nn.FieldType(gspace, depth * [gspace.regular_repr])
        depth *= 6
        self.feat_type_linear = nn.FieldType(gspace, depth * [gspace.regular_repr])

        keys = jax.random.split(key, 7)
        self.escnn1 = econv_module(
            in_type=self.feat_type_in,
            out_type=self.feat_type_out1,
            kernel_size=4,
            stride=2,
            key=keys[0],
            name="s1conv",
        )
        self.equiv_relu1 = nn.ReLU(self.feat_type_out1)
        self.escnn2 = econv_module(
            in_type=self.feat_type_out1,
            out_type=self.feat_type_out2,
            kernel_size=3,
            stride=2,
            key=keys[1],
            name="s2conv",
        )
        self.equiv_relu2 = nn.ReLU(self.feat_type_out2)
        self.escnn3 = econv_module(
            in_type=self.feat_type_out2,
            out_type=self.feat_type_out3,
            kernel_size=3,
            stride=2,
            key=keys[2],
            name="s3conv",
        )
        self.equiv_relu3 = nn.ReLU(self.feat_type_out3)
        self.escnn4 = econv_module(
            in_type=self.feat_type_out3,
            out_type=self.feat_type_out4,
            kernel_size=3,
            stride=2,
            key=keys[3],
            name="s4conv",
        )
        self.equiv_relu4 = nn.ReLU(self.feat_type_out4)
        self.escnn5 = econv_module(
            in_type=self.feat_type_out4,
            out_type=self.feat_type_out5,
            kernel_size=3,
            stride=1,
            key=keys[4],
            name="s5conv",
        )
        self.equiv_relu5 = nn.ReLU(self.feat_type_out5)
        self.linear = econv_module(
            in_type=self.feat_type_out5,
            out_type=self.feat_type_linear,
            kernel_size=1,
            stride=1,
            key=keys[5],
            name="linear",
        )
        self.equiv_relu_linear = nn.ReLU(self.feat_type_linear)

    def __call__(self, x):
        x = jaxutils.cast_to_compute(x) - 0.5
        x = jnp.moveaxis(x, -1, 1)
        x = nn.GeometricTensor(x, self.feat_type_in)
        x = self.escnn1(x)
        x = self.equiv_relu1(x)
        x = self.escnn2(x)
        x = self.equiv_relu2(x)
        x = self.escnn3(x)
        x = self.equiv_relu3(x)
        x = self.escnn4(x)
        x = self.equiv_relu4(x)
        x = self.escnn5(x)
        x = self.equiv_relu5(x)
        x = self.linear(x)
        x = self.equiv_relu_linear(x)
        x = x.tensor.reshape((x.shape[0], -1))
        return x


class EquivImageDecoder(nj.Module):

    def __init__(self, grp, deter, cnn_depth, stoch, key, **kw):
        r2_act = grp.grp_act
        minres = kw["minres"]
        depth = cnn_depth
        self.feat_type_in = nn.FieldType(
            r2_act, (deter // grp.scaler + stoch // grp.scaler) * [r2_act.regular_repr]
        )
        self.feat_type_linear = nn.FieldType(
            r2_act, depth * minres * minres * [r2_act.regular_repr]
        )
        # TODO: clean this up
        depth = depth * minres * minres // grp.scaler
        self.feat_type_hidden1 = nn.FieldType(r2_act, depth * [r2_act.trivial_repr])
        depth = depth // 2
        self.feat_type_hidden2 = nn.FieldType(r2_act, depth * [r2_act.regular_repr])
        depth = depth // 2
        self.feat_type_hidden3 = nn.FieldType(r2_act, depth * [r2_act.regular_repr])
        depth = depth // 2
        self.feat_type_hidden4 = nn.FieldType(r2_act, depth * [r2_act.regular_repr])
        depth = depth // 2
        self.feat_type_hidden5 = nn.FieldType(r2_act, depth * [r2_act.regular_repr])
        depth = depth // 2
        self.feat_type_hidden6 = nn.FieldType(r2_act, depth * [r2_act.regular_repr])
        self.feat_type_out = nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])

        keys = jax.random.split(key, 7)
        self.linear = econv_module(
            in_type=self.feat_type_in,
            out_type=self.feat_type_linear,
            kernel_size=1,
            stride=1,
            key=keys[0],
            name="linear",
        )
        self.equiv_relu0 = nn.ReLU(self.feat_type_linear)
        self.escnn1 = econv_module(
            in_type=self.feat_type_hidden1,
            out_type=self.feat_type_hidden2,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[1],
            name="s1conv",
        )
        self.equiv_relu1 = nn.ReLU(self.feat_type_hidden2)
        self.escnn2 = econv_module(
            in_type=self.feat_type_hidden2,
            out_type=self.feat_type_hidden3,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[2],
            name="s2conv",
        )
        self.equiv_relu2 = nn.ReLU(self.feat_type_hidden3)
        self.escnn3 = econv_module(
            in_type=self.feat_type_hidden3,
            out_type=self.feat_type_hidden4,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[3],
            name="s3conv",
        )
        self.equiv_relu3 = nn.ReLU(self.feat_type_hidden4)
        self.escnn4 = econv_module(
            in_type=self.feat_type_hidden4,
            out_type=self.feat_type_hidden5,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[4],
            name="s4conv",
        )
        self.equiv_relu4 = nn.ReLU(self.feat_type_hidden5)
        self.escnn5 = econv_module(
            in_type=self.feat_type_hidden5,
            out_type=self.feat_type_hidden6,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[5],
            name="s5conv",
        )
        self.equiv_relu5 = nn.ReLU(self.feat_type_hidden6)
        self.escnn6 = econv_module(
            in_type=self.feat_type_hidden6,
            out_type=self.feat_type_out,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[5],
            name="s6conv",
        )
        self._gspace = r2_act

    def _create_spatial_dims(self, x):
        tensors = x.split(list(range(len(x.type)))[1:])
        tensors = jax.tree.map(lambda x: x.tensor, tensors)
        midpoint = len(tensors) // 2
        upper, lower = tensors[:midpoint], tensors[midpoint:]
        lower = jnp.concatenate(
            jax.tree.map(lambda x: jnp.moveaxis(x, 1, -1), lower), 1
        )
        upper = jnp.concatenate(
            jax.tree.map(lambda x: jnp.moveaxis(x, 1, -1), upper), 1
        )
        return jnp.concatenate([upper, lower], -2)

    def __call__(self, x):
        x = x[:, :, jnp.newaxis, jnp.newaxis]

        x = nn.GeometricTensor(x, self.feat_type_in)
        x = self.linear(x)
        x = self.equiv_relu0(x)
        y = self._create_spatial_dims(x)
        x = nn.GeometricTensor(y, self.feat_type_hidden1)
        x = self.escnn1(x)
        x = jnp.repeat(jnp.repeat(x.tensor, 2, -1), 2, -2)
        x = self.get("norm1", Norm, "escnn_layer")(x)
        x = nn.GeometricTensor(x, self.feat_type_hidden2)
        x = self.equiv_relu1(x)
        x = self.escnn2(x)
        x = jnp.repeat(jnp.repeat(x.tensor, 2, -1), 2, -2)
        x = self.get("norm2", Norm, "escnn_layer")(x)
        x = nn.GeometricTensor(x, self.feat_type_hidden3)
        x = self.equiv_relu2(x)
        x = self.escnn3(x)
        x = jnp.repeat(jnp.repeat(x.tensor, 2, -1), 2, -2)
        x = self.get("norm3", Norm, "escnn_layer")(x)
        x = nn.GeometricTensor(x, self.feat_type_hidden4)
        x = self.equiv_relu3(x)
        x = self.escnn4(x)
        x = jnp.repeat(jnp.repeat(x.tensor, 2, -1), 2, -2)
        x = self.get("norm4", Norm, "escnn_layer")(x)
        x = nn.GeometricTensor(x, self.feat_type_hidden5)
        x = self.equiv_relu4(x)
        x = self.escnn5(x)
        x = jnp.repeat(jnp.repeat(x.tensor, 2, -1), 2, -2)
        x = self.get("norm5", Norm, "escnn_layer")(x)
        x = nn.GeometricTensor(x, self.feat_type_hidden6)
        x = self.equiv_relu5(x)
        x = self.escnn6(x)
        x = jaxutils.cast_to_compute(x.tensor) + 0.5
        x = jnp.moveaxis(x, 1, -1)
        return x


class ImageDecoderResnet(nj.Module):

    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get("in", Linear, (self._minres, self._minres, depth))(x)
        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x


class MLP(nj.Module):

    def __init__(
        self,
        shape,
        layers,
        units,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        **kw,
    ):
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
        distkeys = ("dist", "outscale", "minstd", "maxstd", "outnorm", "unimix", "bins")
        self._dense = {
            k: v for k, v in kw.items() if k not in distkeys and k != "equiv"
        }
        self._dist = {k: v for k, v in kw.items() if k in distkeys and k != "equiv"}

    def __call__(self, inputs, invariant=False):
        if invariant:
            feat = inputs
        else:
            feat = self._inputs(inputs)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = jaxutils.cast_to_compute(feat)
        x = x.reshape([-1, x.shape[-1]])
        for i in range(self._layers):
            x = self.get(f"h{i}", Linear, self._units, **self._dense)(x)
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(x)


class InvMLP(MLP):

    def __init__(
        self,
        shape,
        layers,
        units,
        deter,
        stoch,
        grp,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        **kw,
    ):

        super().__init__(
            shape=shape,
            layers=layers,
            units=units,
            inputs=inputs,
            dims=dims,
            symlog_inputs=symlog_inputs,
            **kw,
        )

        r2_act = grp.grp_act
        self.feat_type_in = nn.FieldType(
            r2_act, (deter // grp.scaler + stoch // grp.scaler) * [r2_act.regular_repr]
        )
        self.group_pooling = pooling_module(self.feat_type_in, name="group_pooling")

    def __call__(self, inputs):
        feat = self._inputs(inputs)
        x = feat.reshape([-1, feat.shape[-1]])
        x = x[:, :, jnp.newaxis, jnp.newaxis]
        assert len(x.shape) == 4
        x = nn.GeometricTensor(x, self.feat_type_in)
        x = self.group_pooling(x).tensor.mean(-1).mean(-1)
        return super().__call__(
            x.reshape(feat.shape[:-1] + (x.shape[-1],)), invariant=True
        )


class EquivMLP(MLP):

    def __init__(
        self,
        shape,
        layers,
        units,
        deter,
        stoch,
        key,
        grp,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        invariant=True,
        cup_catch=False,
        **kw,
    ):

        super().__init__(
            shape=shape,
            layers=layers,
            units=units,
            inputs=inputs,
            dims=dims,
            symlog_inputs=symlog_inputs,
            **kw,
        )
        r2_act = grp.grp_act
        factor = r2_act.regular_repr.size // grp.scaler
        units = units // factor
        self.feat_type_in = nn.FieldType(
            r2_act, (deter // grp.scaler + stoch // grp.scaler) * [r2_act.regular_repr]
        )
        self.feat_type_hidden = nn.FieldType(r2_act, units * [r2_act.regular_repr])
        keys = jax.random.split(key, 4)
        self.escnn1 = econv_module(
            in_type=self.feat_type_in,
            out_type=self.feat_type_hidden,
            kernel_size=1,
            key=keys[0],
            name="s1conv",
        )
        self.escnn2 = econv_module(
            in_type=self.feat_type_hidden,
            out_type=self.feat_type_hidden,
            kernel_size=1,
            key=keys[1],
            name="s2conv",
        )
        if invariant:
            self._field_out_type = None
            self._init_equiv_actor = None
            self.group_pooling = pooling_module(
                self.feat_type_hidden, name="group_pooling"
            )
        else:
            assert isinstance(shape, tuple)
            gspace = grp.grp_act
            if gspace.fibergroup.name == "C2":
                if cup_catch:
                    self._field_out_type = nn.FieldType(
                        gspace, [gspace.regular_repr] + [gspace.trivial_repr]
                    )
                else:
                    self._field_out_type = nn.FieldType(
                        gspace,
                        shape[0] * [gspace.regular_repr],
                    )
            elif gspace.fibergroup.name == "D2":
                # Reacher
                self._field_out_type = nn.FieldType(
                    gspace,
                    shape[0] * [gspace.quotient_repr((None, gspace.rotations_order))],
                )
            else:
                raise NotImplementedError("only implemented for groups C2,D2")
            act_dim = None
            if cup_catch:
                act_dim = 2
            else:
                act_dim = shape[0]
            self._field_std_type = nn.FieldType(gspace, act_dim * [r2_act.trivial_repr])
            self._init_equiv_actor = nn.R2Conv(
                in_type=self.feat_type_hidden,
                out_type=self._field_out_type,
                kernel_size=1,
                key=keys[2],
            )
            self._init_equiv_std = nn.R2Conv(
                in_type=self.feat_type_hidden,
                out_type=self._field_std_type,
                kernel_size=1,
                key=keys[3],
            )
            self.group_pooling = None
        self.invariant = invariant
        self.equiv_relu = nn.ReLU(self.feat_type_hidden)
        self._cup_catch = cup_catch

    def __call__(self, inputs):
        feat = self._inputs(inputs)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = jaxutils.cast_to_compute(feat)
        x = x.reshape([-1, x.shape[-1]])

        x = x[:, :, jnp.newaxis, jnp.newaxis]
        assert len(x.shape) == 4
        x = nn.GeometricTensor(x, self.feat_type_in)
        x = self.escnn1(x)
        x = self.equiv_relu(x)
        x = self.escnn2(x)
        x = self.equiv_relu(x)
        if self.invariant:
            x = self.group_pooling(x).tensor.mean(-1).mean(-1)
        else:
            x = x.tensor.mean(-1).mean(-1)

        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        if self._dist["dist"] == "equiv_normal":
            self._dist["in_type"] = self.feat_type_hidden
            self._dist["out_type"] = self._field_out_type
            self._dist["std_type"] = self._field_std_type
            self._dist["init_equiv_actor"] = self._init_equiv_actor
            self._dist["init_equiv_std"] = self._init_equiv_std
            self._dist["group_pooling"] = self.group_pooling
            self._dist["cup_catch"] = self._cup_catch
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(x)


class Dist(nj.Module):

    def __init__(
        self,
        shape,
        dist="mse",
        outscale=0.1,
        outnorm=False,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
        in_type=None,
        out_type=None,
        std_type=None,
        init_equiv_actor=None,
        init_equiv_std=None,
        group_pooling=None,
        cup_catch=False,
    ):
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._outscale = outscale
        self._outnorm = outnorm
        self._bins = bins
        if dist == "equiv_normal":
            assert (
                in_type is not None
                and out_type is not None
                and init_equiv_actor is not None
            )
            self._field_in_type = in_type
            self._field_out_type = out_type
            self._field_std_type = std_type
            self._init_equiv_actor = init_equiv_actor
            self._init_equiv_std = init_equiv_std
            self._cup_catch = cup_catch
            self._group_pooling = group_pooling

    def __call__(self, inputs):
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape,
            dist.event_shape,
            inputs.shape,
        )
        return dist

    def inner(self, inputs):
        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"] = self._outnorm
        shape = self._shape
        if self._dist.endswith("_disc"):
            shape = (*self._shape, self._bins)
        if self._dist == "equiv_normal":
            out = self.get(
                "out",
                EquivLinear,
                **{
                    "net": self._init_equiv_actor,
                    "in_type": self._field_in_type,
                    "out_type": self._field_out_type,
                    "norm": "none",
                    "act": "none",
                },
            )(inputs.reshape([-1, inputs.shape[-1]]))
            std = self.get(
                "std",
                EquivLinear,
                **{
                    "net": self._init_equiv_std,
                    "in_type": self._field_in_type,
                    "out_type": self._field_std_type,
                    "norm": "none",
                    "act": "none",
                },
            )(inputs.reshape([-1, inputs.shape[-1]]))
            out = out.reshape(inputs.shape[:-1] + (out.shape[-1],)).astype(f32)
            std = std.reshape(inputs.shape[:-1] + (std.shape[-1],)).astype(f32)
        else:
            out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
            out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
        if self._dist == "symlog_mse":
            return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
        if self._dist == "symlog_disc":
            return jaxutils.DiscDist(
                out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp
            )
        if self._dist == "mse":
            return jaxutils.MSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "equiv_normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            if self._cup_catch:
                out = out @ jnp.array([[1, 0], [-1, 0], [0, 1]])
            else:
                out = out.reshape(out.shape[:-1] + (-1, 2))
                out = out @ jnp.array([1, -1])
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "onehot":
            if self._unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

    def __init__(
        self,
        depth,
        kernel,
        stride=1,
        transp=False,
        act="none",
        norm="none",
        pad="same",
        bias=True,
        preact=False,
        winit="uniform",
        fan="avg",
    ):
        self._depth = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act = get_act(act)
        self._norm = Norm(norm, name="norm")
        self._pad = pad.upper()
        self._bias = bias and (preact or norm == "none")
        self._preact = preact
        self._winit = winit
        self._fan = fan

    def __call__(self, hidden):
        if self._preact:
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        if self._transp:
            shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
            kernel = self.get("kernel", Initializer(self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_transpose(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )
        else:
            shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
            kernel = self.get("kernel", Initializer(self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_general_dilated(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        if self._bias:
            bias = self.get("bias", jnp.zeros, self._depth, np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        return x


class Linear(nj.Module):

    def __init__(
        self,
        units,
        act="none",
        norm="none",
        bias=True,
        outscale=1.0,
        outnorm=False,
        winit="uniform",
        fan="avg",
    ):
        self._units = tuple(units) if hasattr(units, "__len__") else (units,)
        self._act = get_act(act)
        self._norm = norm
        self._bias = bias and norm == "none"
        self._outscale = outscale
        self._outnorm = outnorm
        self._winit = winit
        self._fan = fan

    def __call__(self, x):
        shape = (x.shape[-1], np.prod(self._units))
        kernel = self.get(
            "kernel", Initializer(self._winit, self._outscale, fan=self._fan), shape
        )
        kernel = jaxutils.cast_to_compute(kernel)
        x = x @ kernel
        if self._bias:
            bias = self.get("bias", jnp.zeros, np.prod(self._units), np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)
        x = self.get("norm", Norm, self._norm)(x)
        x = self._act(x)
        return x


class EquivLinear(nj.Module):

    def __init__(self, net, in_type, out_type, act="none", norm="none"):
        self._ecnn = econv_module(net=net, name="conv")
        self._in_type = in_type
        self._out_type = out_type
        self._act = get_act(act, in_type=out_type)
        self._norm = norm

    def __call__(self, x):
        x = x[:, :, jnp.newaxis, jnp.newaxis]
        assert len(x.shape) == 4
        x = nn.GeometricTensor(x, self._in_type)
        x = self._ecnn(x)
        x = self.get("norm", Norm, self._norm)(x.tensor.mean(-1).mean(-1))
        x = nn.GeometricTensor(x[:, :, jnp.newaxis, jnp.newaxis], self._out_type)
        x = self._act(x).tensor.mean(-1).mean(-1)
        return x


class EquivGRUCell(EquivLinear):

    def __call__(self, x):
        x = x[:, :, jnp.newaxis, jnp.newaxis]
        assert len(x.shape) == 4
        x = nn.GeometricTensor(x, self._in_type)
        x = self._ecnn(x)
        x = self.get("norm", Norm, self._norm)(x.tensor.mean(-1).mean(-1))
        x = nn.GeometricTensor(x[:, :, jnp.newaxis, jnp.newaxis], self._out_type)
        return x


class Norm(nj.Module):

    def __init__(self, impl):
        self._impl = impl

    def __call__(self, x):
        dtype = x.dtype
        if self._impl == "none":
            return x
        elif self._impl == "layer":
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
            x *= self.get("scale", jnp.ones, x.shape[-1], f32)
            x += self.get("bias", jnp.zeros, x.shape[-1], f32)
            return x.astype(dtype)
        elif self._impl == "escnn_layer":
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=1, epsilon=1e-3)
            x *= self.get("scale", jnp.ones, x.shape[-1], f32)
            x += self.get("bias", jnp.zeros, x.shape[-1], f32)
            return x.astype(dtype)
        else:
            raise NotImplementedError(self._impl)


class Input:

    def __init__(self, keys=["tensor"], dims=None):
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}
        inputs = inputs.copy()
        for key in self._keys:
            if key.startswith("softmax_"):
                inputs[key] = jax.nn.softmax(inputs[key[len("softmax_") :]])
        if not all(k in inputs for k in self._keys):
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f"Cannot find keys {needs} among inputs {found}.")
        values = [inputs[k] for k in self._keys]
        dims = len(inputs[self._dims].shape)
        for i, value in enumerate(values):
            if len(value.shape) > dims:
                values[i] = value.reshape(
                    value.shape[: dims - 1] + (np.prod(value.shape[dims - 1 :]),)
                )
        values = [x.astype(inputs[self._dims].dtype) for x in values]
        return jnp.concatenate(values, -1)


class Initializer:

    def __init__(self, dist="uniform", scale=1.0, fan="avg"):
        self.scale = scale
        self.dist = dist
        self.fan = fan

    def __call__(self, shape):
        if self.scale == 0.0:
            value = jnp.zeros(shape, f32)
        elif self.dist == "uniform":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            limit = np.sqrt(3 * scale)
            value = jax.random.uniform(nj.rng(), shape, f32, -limit, limit)
        elif self.dist == "unit_normal":
            value = jax.random.normal(nj.rng(), shape, f32)
        elif self.dist == "normal":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)), "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            std = np.sqrt(scale) / 0.87962566103423978
            value = std * jax.random.truncated_normal(nj.rng(), -2, 2, shape, f32)
        elif self.dist == "ortho":
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(nj.rng(), matshape, f32)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = self.scale * jnp.moveaxis(qmat, 0, -1)
        else:
            raise NotImplementedError(self.dist)
        return value

    def _fans(self, shape):
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space


def get_act(name, in_type=None):
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "equiv_relu":
        assert in_type is not None
        return lambda x: nn.ReLU(in_type=in_type)(x)
    elif name == "equiv_silu":
        assert in_type is not None
        return lambda x: nn.SiLU(in_type=in_type)(x)
    elif name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)
