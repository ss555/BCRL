import pickle
import sys
import time
import argparse
from os import path
import os
import numpy as np

from glob import glob

from surreal.env import *
import surreal.utils as U
from surreal.agent import PPOAgent

from benedict import BeneDict
import pyquaternion as pq

def restore_model(folder):
    """
    Loads model from an experiment folder.
    """

    # choose latest ckpt
    max_iter = -1.
    max_ckpt = None
    print(folder)
    for ckpt in glob(path.join(folder, "*.ckpt")):
        print(ckpt)
        iter_num = int(path.basename(ckpt).split('.')[1])
        if iter_num > max_iter:
            max_iter = iter_num
            max_ckpt = ckpt
    if max_ckpt is None:
        raise ValueError('No checkpoint available in folder {}'.format())
    path_to_ckpt = max_ckpt
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs

def restore_env(env_config):
    """
    Restores the environment.
    """
    #env_config.eval_mode.render = True
    env, env_config = make_env(env_config, 'eval')
    return env, env_config

def restore_agent(agent_class, learner_config, env_config, session_config, model):
    """
    Restores an agent from a model.
    """
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=0,
        agent_mode='eval_deterministic_local',
    )
    agent.model.load_state_dict(model)
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,)
    parser.add_argument("--render", action='store_true',)
    parser.add_argument("--save_path", type=str, default='./dataset')
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    folder = args.folder
    render = args.render

    # set a seed
    np.random.seed(int(time.time() * 100000 % 100000))

    # restore policy
    print("\nLoading policy located at {}\n".format(folder))
    model = restore_model(path.join(folder, 'checkpoint'))

    # restore the configs
    configs = restore_config(path.join(folder, 'config.yml'))
    session_config, learner_config, env_config = \
        configs.session_config, configs.learner_config, configs.env_config

    # session_config.agent.num_gpus = 0
    # session_config.learner.num_gpus = 0
    # env_config.env_name = 'mujocomanip:SawyerPegsRoundEnv'

    # restore the environment
    env_config.eval_mode.render = render
    env, env_config = restore_env(env_config)


    # restore the agent
    agent = restore_agent(PPOAgent, learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")

    # do some rollouts
    states = []
    delta_eef_pos = []
    delta_eef_rotation = []


    images = []
    proprio = []
    gripper = []
    eef = []

    for _ in range(100):
        if len(states):
            states = states[:-1]

        prev_eef_pos, prev_eef_quat = None, None

        ob, info = env.reset()
        ret = 0.
        if render:
            env.unwrapped.viewer.viewer._hide_overlay = True
            env.unwrapped.viewer.set_camera(0)

        buff = {'images': [], 'proprio': [], 'gripper': [], 'eef': []}
        for i in range(200):
            a = agent.act(ob)
            #a += np.random.normal(loc=0.0, scale=0.1, size=a.shape)
            #ob_copy = obcopy()
            aux = env.unwrapped._get_observation()

            buff['images'].append(np.transpose(ob['pixel']['camera0'], (1,2,0)))

            buff['proprio'].append(np.array(ob['low_dim']['flat_inputs']).ravel())
            buff['eef'].append(np.concatenate([aux['eef_pos'], aux['eef_quat'], aux['gripper_qvel'], aux['gripper_qpos']]))
            buff['gripper'].append(a[-1])     #aux['gripper_qpos'])

            ob, r, _, _ = env.step(a)

            if render:
                env.unwrapped.render()

            ret += r
        eef.append(buff['eef'])
        images.append(buff['images'])
        proprio.append(buff['proprio'])
        gripper.append(buff['gripper'])
        
        print("return: {}".format(ret))


    dpos = []
    rotation = []
    for i in range(len(images)):
        buff = {'dpos': [], 'rotation': []}
        for j in range(1, len(images[i])):
            buff['dpos'].append(proprio[i][j][:3] - proprio[i][j-1][:3])
            buff['rotation'].append(proprio[i][j][3:3+4])
            assert buff['dpos'][-1].shape[0] == 3
            assert buff['rotation'][-1].shape[0] == 4, buff['rotation'][-1].shape[0]

        dpos.append(buff['dpos'])
        rotation.append(buff['rotation'])

        images[i] = images[i][:-1]
        proprio[i] = proprio[i][:-1]

        assert len(dpos[i]) == 199
        assert len(dpos[i]) == len(images[i])
        assert len(dpos[i]) == len(proprio[i])
    """
    states = states[:-1]
    print(len(states), len(delta_eef_pos), len(delta_eef_rotation))
    for i in range(len(states)):
        states[i]['delta_eef_pos'] = delta_eef_pos[i]
        states[i]['delta_eef_rotation'] = delta_eef_rotation[i]
    """

    #np.save(args.save_path + '/states_no_noise', states)
    np.save(args.save_path + '/sim_image', images)
    np.save(args.save_path + '/sim_joint_states', proprio)
    np.save(args.save_path + '/sim_proprio', eef)
    np.save(args.save_path + '/sim_dpos', dpos)
    np.save(args.save_path + '/sim_rotation', rotation)
    np.save(args.save_path + '/sim_grasp', gripper)
