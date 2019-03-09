import tensorflow as tf
import numpy as np
import cv2
import os
from os import path
import json

import RobotTeleop.utils as RU

try:
    from official.resnet import resnet_model
    RESNET_CAPABLE = True
except:
    RESNET_CAPABLE = False

try:
    from surreal.env import *
    import surreal.utils as U
    from robosuite.wrappers import IKWrapper
    from benedict import BeneDict
    import mujoco_py

    SIM_CAPABLE = True
except:
    SIM_CAPABLE = False

import collections
import matplotlib.pyplot as plt
import argparse
import scipy

class RoboTurkDataset:
    def __init__(self, dataset_path, n_valid=1, real=False,
                 using_robot=False, image_size=(84,84,3), n_proprio_stack=5):
        self.dataset_path = dataset_path
        self.image_mean = None
        self.image_std = None
        self.real = real
        self.using_robot =  using_robot
        self.n_valid = n_valid
        self.image_size = image_size
        self.n_proprio_stack = n_proprio_stack

        #assert not self.real
        self._load()
        self._calculate_or_load_stats()

    def _calculate_or_load_stats(self):
        suffix = '' if self.real else '_sim'
        path = '{}/image_stats{}.json'.format(self.dataset_path, suffix)
        if not os.path.exists(path):
            from tqdm import tqdm

            print('Calculating image mean and std')
            count = 0
            accum = np.array([0.,0.,0.])
            accum_square = np.array([0.,0.,0.])


            for traj in tqdm(range(len(self.images))):
                for t in tqdm(range(len(self.images[traj]))):
                    img = self.images[traj][t]

                    count += img.shape[0] * img.shape[1]
                    accum += np.sum(img, axis=(0,1))
                    accum_square += np.sum(np.square(img), axis=(0,1))

            self.image_mean = accum / count
            self.image_std = np.sqrt(accum_square / count - self.image_mean)

            print('Finished Calculating states, that was easy lol')
            print('MEAN: {}'.format(self.image_mean))
            print('STD: {}'.format(self.image_std))
            print('Lets never do that again')

            with open(path, 'w') as f:
                stats = {
                    'mean': self.image_mean.tolist(),
                    'std': self.image_std.tolist(),
                }
                json.dump(stats, f)
        else:
            with open(path, 'r') as f:
                data = json.load(f)
                self.image_mean = np.array(data['mean'])
                self.image_std = np.array(data['std'])

        assert self.image_mean.shape[0] == 3


    def _load(self):
        if self.using_robot:
            self.image_mean = np.array([107.69177, 113.32283, 118.14253])
            self.image_std = np.array([57.7754, 60.47865, 80.9573])
        else:

            prefix = '/real_' if self.real else '/sim_'
            encoding = 'bytes' if self.real else 'ASCII'

            np.random.seed(1)

            self.images = np.load(self.dataset_path + prefix + 'image.npy', encoding=encoding)
            self.proprio = np.load(self.dataset_path + prefix + 'joint_states.npy', encoding=encoding)
            self.eef = np.load(self.dataset_path  + prefix + 'proprio.npy', encoding=encoding)
            self.dpos = np.load(self.dataset_path + prefix + 'dpos.npy')
            self.rotation = np.load(self.dataset_path + prefix + 'rotation.npy')
            self.gripper = np.load(self.dataset_path + prefix + 'grasp.npy')
            self.goals = np.zeros((len(self.images), len(self.images[0]), 3))
            indices = np.arange(len(self.images))
            np.random.shuffle(indices)

            self.images = self.images[indices]
            self.proprio = self.proprio[indices]
            self.eef = self.eef[indices]
            self.dpos = self.dpos[indices]
            self.rotation = self.rotation[indices]
            self.gripper = self.gripper[indices]

            for ind in range(len(self.gripper)):
                for t in range(len(self.gripper[ind])):
                    if self.gripper[ind][t] <= 0.0001:
                        self.goals[ind][t] = self.eef[ind][t][:3]
                        break

            self.eval_images = np.array(self.images[-self.n_valid:])
            self.eval_proprio = np.array(self.proprio[-self.n_valid:])
            self.eval_dpos = np.array(self.dpos[-self.n_valid:])
            self.eval_rotation = np.array(self.rotation[-self.n_valid:])
            self.eval_gripper = np.array(self.gripper[-self.n_valid:])
            self.eval_eef = np.array(self.eef[-self.n_valid:])
            self.eval_goals = np.array(self.goals[-self.n_valid:])

            self.images = self.images[:-self.n_valid]
            self.proprio = self.proprio[:-self.n_valid]
            self.dpos = self.dpos[:-self.n_valid]
            self.rotation = self.rotation[:-self.n_valid]
            self.gripper = self.gripper[:-self.n_valid]
            self.eef = self.eef[:-self.n_valid]
            self.goals = self.goals[:-self.n_valid]

            self.proprio_size = self.proprio[0][0].shape[0]
            self.eef_size = self.eef[0][0].shape[0]

    def _dataset_generator(self):

        traj_inds = np.random.randint(0, self.images.shape[0], 1000)
        for traj_ind in range(1000):
            t_ind = traj_inds[traj_ind]
            traj = self.images[t_ind]

            time_ind = np.random.randint(0, len(traj))

            if time_ind < self.n_proprio_stack - 1:
                to_stack = [ np.zeros(self.proprio_size) for _ in range(self.n_proprio_stack - time_ind - 1)]

                if time_ind == 0:
                    to_stack.append(self.proprio[t_ind][time_ind])
                else:
                    to_stack.extend([
                        self.proprio[t_ind][time_ind -i] for i in range(time_ind+1)
                    ])
                proprio_stack = np.concatenate(to_stack)

            else:
                proprio_stack = np.concatenate([
                    self.proprio[t_ind][time_ind -i] for i in range(self.n_proprio_stack - 1, -1, -1)
                ])

            if proprio_stack.shape[0] != self.n_proprio_stack * self.proprio_size:
                #print(proprio_stack.shape[0], self.proprio_size)
                continue

            delta_eef_pos = self.dpos[t_ind][time_ind]

            if time_ind == 0:
                euler_rotation = np.array([0,0.57,1.5708])

                if self.real:
                    prev_rotation = RU.quat2mat(RU.euler2quat(euler_rotation))
                else:
                    prev_rotation = RU.euler2quat(euler_rotation)
            else:
                prev_rotation = self.rotation[t_ind][time_ind-1]
            curr_rotation = np.asarray(self.rotation[t_ind][time_ind])

            if curr_rotation.shape == (3,3):
                curr_rotation = RU.mat2euler(curr_rotation)
                prev_rotation = RU.mat2euler(prev_rotation)
            elif curr_rotation.shape[0] == 4 and prev_rotation.shape[0] == 4:
                curr_rotation = RU.quat2euler(curr_rotation)
                prev_rotation = RU.quat2euler(prev_rotation)
            else:
                print('Skipping')
                continue

            delta_eef_rotation = curr_rotation - prev_rotation
            delta_eef_quat = RU.euler2quat(delta_eef_rotation)

            image = self.images[t_ind][time_ind]
            image = cv2.resize(image, self.image_size[:-1])

            eef = self.eef[t_ind][time_ind]
            goal = self.goals[t_ind][time_ind]

            gripper = np.array(self.gripper[t_ind][time_ind]).reshape(1)

            yield {
                'image': np.array((image - self.image_mean) / (self.image_std + 1e-9)),
                'proprio': np.array(proprio_stack),
                'gripper': gripper,
                'delta_eef_pos': np.array(delta_eef_pos),
                'delta_eef_rotation': np.array(delta_eef_quat),
                'eef': eef,
                'goal': goal,
            }


    def _make_dataset(self, batch_size):

        with tf.variable_scope('Dataset'):
            output_types = {
                'image': tf.float32,
                'proprio': tf.float32,
                'gripper': tf.float32,
                'delta_eef_pos': tf.float32,
                'delta_eef_rotation': tf.float32,
                'eef': tf.float32,
                'goal': tf.float32
            }
            output_shapes = {
                'image': self.image_size,
                'proprio': self.proprio_size * self.n_proprio_stack,
                'gripper': 1,
                'delta_eef_pos': (3,),
                'delta_eef_rotation': (4,),
                'eef': (self.eef_size,),
                'goal': (3,),
            }

            self.tf_dataset = tf.data.Dataset.from_generator(self._dataset_generator,
                                    output_types=output_types,
                                    output_shapes=output_shapes).shuffle(10).repeat().batch(batch_size).make_one_shot_iterator()


    def make_tf_dataset(self, batch_size):
        self._make_dataset(batch_size)
        return self.tf_dataset

    @property
    def dataset(self):
        assert self.tf_dataset is not None, 'You need to call make_tf_dataset before attempting to retrieve the iterator'
        return self.tf_dataset
