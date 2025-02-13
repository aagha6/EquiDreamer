import functools
import elements
import embodied
import gym
import numpy as np
from helping_hands_rl_envs import env_factory

def decode_actions(unscaled_action, p_range, dx_range, dy_range, dz_range, dtheta_range):
    unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta = unscaled_action[0], unscaled_action[1], unscaled_action[2], unscaled_action[3], unscaled_action[4]
    p = 0.5 * (unscaled_p + 1) * (p_range[1] - p_range[0]) + p_range[0]
    dx = 0.5 * (unscaled_dx + 1) * (dx_range[1] - dx_range[0]) + dx_range[0]
    dy = 0.5 * (unscaled_dy + 1) * (dy_range[1] - dy_range[0]) + dy_range[0]
    dz = 0.5 * (unscaled_dz + 1) * (dz_range[1] - dz_range[0]) + dz_range[0]
    dtheta = 0.5 * (unscaled_dtheta + 1) * (dtheta_range[1] - dtheta_range[0]) + dtheta_range[0]
    action = np.stack([p, dx, dy, dz, dtheta])
    
    return action

class Manipulation(embodied.Env):
  def __init__(self, task, size=(128, 128), repeat=1, camera=-1, obs_key="image", act_key="action"):

    self._size = size
    workspace_size = 0.4
    dpos = 0.05
    drot = np.pi/8
    workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                            [0-workspace_size/2, 0+workspace_size/2],
                            [0.01, 0.25]])
    if task in ['close_loop_block_stacking', 'close_loop_house_building_1', 'close_loop_block_pulling']:
      num_objects = 2
    else:
      num_objects = 1    
    env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': size[0],
              'fast_mode': True,  'action_sequence': "pxyzr", 'render': False, 'num_objects': num_objects,
              'random_orientation':True, 'robot': "kuka",
              'workspace_check': 'point', 'object_scale_range': (1, 1),
              'hard_reset_freq': 1000, 'physics_mode' : 'fast', 'view_type': "camera_center_xyz", 'obs_type': "pixel"}
    planner_config = {'random_orientation':True, 'dpos': dpos, 'drot': drot}
    
    self.p_range = np.array([0, 1])
    self.dtheta_range = np.array([-drot, drot])
    self.dx_range = np.array([-dpos, dpos])
    self.dy_range = np.array([-dpos, dpos])
    self.dz_range = np.array([-dpos, dpos])
    self._env = env_factory.createEnvs(0, 'pybullet', task, env_config, planner_config)
    self._obs_dict = False
    self._act_dict = False
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
  
  @property
  def env(self):
    return self._env
  
  @property
  def info(self):
    return self._info
  
  @functools.cached_property
  def obs_space(self):
    spaces = {}
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (2,), dtype=np.uint8)
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    action_space = self._env.env.action_space
    minimum = np.concatenate([np.array([0.]), action_space[0]],0)
    maximum = np.concatenate([np.array([1.]), action_space[1]],0)
    action = gym.spaces.Box(minimum, maximum, dtype=np.float32)
    spaces = gym.spaces.Dict({'action': action})
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def get_next_action(self):
    return self._env.getNextAction()

  def process_obs(self, state, depth_img):
      depth_img = np.clip(depth_img, 0, 0.32)
      depth_img = depth_img/0.4*255
      depth_img = depth_img.astype(np.uint8).reshape(depth_img.shape[1], depth_img.shape[2], 1)
      
      state *= 255
      state = np.array(state)[np.newaxis].astype(np.uint8)
      state_tile = np.tile(state.reshape(1, 1, 1),(depth_img.shape[0],depth_img.shape[1],1))
      stacked = np.concatenate([depth_img, state_tile], -1)
      return stacked

  def decode_actions(self, unscaled_action):
    unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta = unscaled_action[0], unscaled_action[1], unscaled_action[2], unscaled_action[3], unscaled_action[4]

    p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
    dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
    dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
    dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]
    dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]

    action = np.stack([p, dx, dy, dz, dtheta])
    
    return action
  
  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      state, _, depth_img = self._env.reset()
      obs = self.process_obs(state=state, depth_img=depth_img)      
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    
    scaled_action = self.decode_actions(action)
    (state, _, depth_img), reward, self._done = self._env.step(scaled_action, auto_reset=False)
    obs = self.process_obs(state=state, depth_img=depth_img)      
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=False)

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)