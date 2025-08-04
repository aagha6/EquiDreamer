# EquiDreamer: Equivaraint Model Based Reinforcement Learning 

This is a Jax implementation of EquiDreamer

This implementation is based on [DreamerV3](https://github.com/danijar/dreamerv3) and [escnn-jax](https://github.com/emilemathieu/escnn_jax).

### Dependencies

* [Jax](http://pytorch.org/)
* [escnn-jax](https://github.com/emilemathieu/escnn_jax)

### Installing

```bash
conda create -n equidreamer-env python=3.9
```
#### jax
```bash
pip install -U "jax[cuda12]==0.4.19"
```
#### Other requirements
```bash
pip install -r requirements.txt
```
#### escnn-jax
```bash
git clone https://github.com/emilemathieu/escnn_jax
cd escnn_jax
pip install -e .
```
## Training

### Dreamer

```bash
 python3 dreamerv3/train.py --run.steps 0.5e5 --logdir ./logdir --configs dmc_vision --task dmc_cartpole_swingup --seed 0
```
### EquiDreamer
```bash
 python3 dreamerv3/train.py --run.steps 0.5e5 --logdir ./logdir --configs dmc_vision_equidreamer --task dmc_cartpole_swingup --seed 0
```