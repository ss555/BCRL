import argparse
import collections

import numpy as np
import scipy.signal

from policy import PPO
import robosuite as suite
from robosuite.wrappers import IKWrapper
from gym import spaces as spaces

class FlattenWrapper:
    def __init__(self, env, nodict=False):
        self.env = env
        self.nodict = nodict

        obs = self.env._get_observation()
        obs = self.flatten_obs(obs)
        self.observation_space = dict([(k, obs[k].shape) for k in obs])
        self.action_space = spaces.Box(-1, 1, (self.dof,))

    def flatten_obs(self, obs):
        if self.nodict:
            robot_state = np.concatenate([
                obs[k] for k in  obs if k != 'robot-state'
            ], -1)

            return robot_state

        if 'image' in obs:
            return {
                'image': obs['image'],
                'robot-state': obs['robot-state'],
            }
        else:
            robot_state = np.concatenate([
                obs[k] for k in  obs if k != 'robot-state'
            ], -1)

            return {
                'robot-state': robot_state
            }

    def _get_observation(self):
        obs = self.env._get_observation()
        return self.flatten_obs(obs)

    def reset(self):
        obs = self.env.reset()
        return self.flatten_obs(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return self.flatten_obs(obs), r, done, info

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def dof(self):
        return self.env.dof


class Normalizer(object):

    def __init__(self, obs_dim):

        if len(obs_dim) == 3:
            self.image = True
        else:
            self.image = False

        if self.image:
            self.var = np.zeros((3,))
            self.mean = np.zeros((3,))
        else:
            self.var = np.zeros(obs_dim)
            self.mean = np.zeros(obs_dim)

        if self.image:
            self.axis = (0,1)
        else:
            self.axis=0

        self.m = 0
        self.first = True

    def update(self, x):
        if self.first:

            self.mean = np.mean(x, axis=self.axis)
            self.var = np.var(x, axis=self.axis)
            self.m = x.shape[0]

            if self.image:
                self.m = x.shape[0] * x.shape[1]

            self.first = False
        else:
            if self.image:
                n = x.shape[0] * x.shape[1]
            else:
                n = x.shape[0]
            self.m += n

            var, mean = np.var(x, axis=self.axis), np.mean(x, axis=self.axis)
            mean_sq = np.square(mean)

            new_mean = ((self.mean * self.m) + (mean * n)) / (self.m + n)
            self.var = (((self.m * (self.var + np.square(self.mean))) +
                          (n * (var + mean_sq))) / (self.m + n) -
                         np.square(new_mean))
            self.var = np.maximum(0.0, self.var)
            self.mean = new_mean

    def get(self):
        return 1/(np.sqrt(self.var) + 0.1)/3, self.mean


def init_env():
    env = suite.make('SawyerLift', ignore_done=False, use_camera_obs=True, use_object_obs=False,
                     reward_shaping=True, camera_name='agentview', camera_height=84, camera_width=84, horizon=200, control_freq=10)
    env = IKWrapper(env, action_repeat=100)
    env = FlattenWrapper(env)

    obs_dim = env.observation_space
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_n_episodes(env, policy, normalizer_state, normalizer_image, episodes):
    total_steps = 0
    trajectories = []
    for _ in range(episodes):
        done = False
        images, states, actions, rewards, unscaled_imgs, unscaled_states, values = [], [], [], [], [], [], []
        var_state, mean_state = normalizer_state.get()
        var_image, mean_image = normalizer_image.get()

        obs = env.reset()

        proprio_stack = collections.deque(maxlen=5)
        for _ in range(5):
            proprio_stack.append(np.zeros(normalizer_state.mean.shape))

        while not done:

            img = obs['image']
            state = obs['robot-state']

            #obs = obs.astype(np.float32).reshape((1, -1))
            unscaled_imgs.append(img)
            unscaled_states.append(state)

            #img = (img - mean_image) * var_image
            state = (state - mean_state)*var_state
            proprio_stack.append(state)

            images.append(img)

            stack = np.array(proprio_stack)
            stack = np.reshape(stack, (-1,))
            states.append(stack)

            stack = np.reshape(stack, (1, -1))

            action, value = policy.sample(stack, img)
            action = action.reshape((1, -1)).astype(np.float32)
            values.append(value)

            actions.append(action)
            obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
            rewards.append(reward)

        images = np.stack(images)
        states = np.stack(states)
        rewards = np.stack(rewards)
        actions = np.concatenate(actions)
        values = np.concatenate(values)
        unscaled_imgs = np.concatenate(unscaled_imgs)
        unscaled_states = np.stack(unscaled_states)

        total_steps += images.shape[0]
        trajectory = {'images': images,
                      'states': states,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_imgs': unscaled_imgs,
                      'unscaled_states': unscaled_states,
                      'values': values
        }
        trajectories.append(trajectory)
    unscaled_state = np.concatenate([t['unscaled_states'] for t in trajectories])
    unscaled_imgs = np.concatenate([t['unscaled_imgs'] for t in trajectories])

    normalizer_state.update(unscaled_state)
    normalizer_image.update(unscaled_imgs)

    return trajectories

def add_returns(trajectories, gamma):
    for t in trajectories:
        if gamma < 0.999:
            rewards = t['rewards'] * (1 - gamma)
        else:
            rewards = t['rewards']
        returns = scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]
        t['returns'] = returns

def add_gae(trajectories, gamma, lam):
    for trajectory in trajectories:
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = scipy.signal.lfilter([1.0], [1.0, -(gamma*lam)], tds[::-1])[::-1]
        trajectory['advantages'] = advantages

def main(num_episodes, gamma, lam, batch_size):

    env, obs_dim, act_dim = init_env()
    env2, _, _ = init_env()


    def thunk():
        import mujoco_py
        viewer = mujoco_py.MjViewer(env2.unwrapped.sim)

        while True:
            env2.unwrapped.sim.set_state(env.unwrapped.sim.get_state())
            env2.unwrapped.sim.step()
            viewer.render()

    import threading
    thread = threading.Thread(target=thunk)
    thread.start()


    normalizer_states = Normalizer(obs_dim['robot-state'])
    normalizer_imgs = Normalizer(obs_dim['image'])

    policy = PPO(obs_dim['robot-state'], act_dim, './eef_1')
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    run_n_episodes(env, policy, normalizer_states, normalizer_imgs, episodes=1)
    episode = 0
    while episode < num_episodes:
        trajectories = run_n_episodes(env, policy, normalizer_states, normalizer_imgs, episodes=batch_size)
        episode += len(trajectories)

        add_returns(trajectories, gamma)
        add_gae(trajectories, gamma, lam)

        images = np.concatenate([t['images'] for t in trajectories])
        states = np.concatenate([t['states'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        returns = np.concatenate([t['returns'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        sum_of_rewards = np.mean([np.sum(t['rewards']) for t in trajectories])
        print('Mean sum of rewards: {}'.format(sum_of_rewards))

        if episode <= 11 * len(trajectories):
            rng = 10
        else:
            rng = 1

        for i in range(rng):
            print(i)
            policy.optimize({
                'images': images.reshape((-1, *obs_dim['image'])),
                'states': states.reshape((-1, 5 * obs_dim['robot-state'][0])),
                'actions': actions.reshape((-1, act_dim)),
                'advantages': advantages.reshape((-1, 1)),
                'returns': returns.reshape((-1, 1))
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=10000000000000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)

    args = parser.parse_args()

    import tensorflow as tf
    config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(**vars(args))
