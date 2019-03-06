import tensorflow as tf
import numpy as np
import cv2
import os
from os import path
import json

import RobotTeleop.utils as RU

try:
    from official.resnet import resnet_model
except:
    print('Cannot import the tensorflow models repository')

from surreal.env import *
import surreal.utils as U
from robosuite.wrappers import IKWrapper
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

        assert not self.real
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
            self.images = np.load(self.dataset_path + prefix + 'images.npy', encoding=encoding)
            self.proprio = np.load(self.dataset_path + prefix + 'proprio.npy', encoding=encoding)
            self.dpos = np.load(self.dataset_path + prefix + 'dpos.npy')
            self.rotation = np.load(self.dataset_path + prefix + 'rotation.npy')
            self.gripper = np.load(self.dataset_path + prefix + 'grasp.npy')

            self.images = self.images[:-self.n_valid]
            self.proprio = self.proprio[:-self.n_valid]
            self.dpos = self.dpos[:-self.n_valid]
            self.rotation = self.rotation[:-self.n_valid]
            self.gripper = self.gripper[:-self.n_valid]

            self.proprio_size = self.proprio[0][0].shape[0]

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
                print(proprio_stack.shape[0], self.proprio_size)
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
            curr_rotation = self.rotation[t_ind][time_ind]

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

            gripper = np.array(self.gripper[t_ind][time_ind]).reshape(1)

            yield {
                'image': np.array((image - self.image_mean) / (self.image_std + 1e-9)),
                'proprio': np.array(proprio_stack),
                'gripper': gripper,
                'delta_eef_pos': np.array(delta_eef_pos),
                'delta_eef_rotation': np.array(delta_eef_quat),
            }


    def _make_dataset(self, batch_size):

        with tf.variable_scope('Dataset'):
            output_types = {
                'image': tf.float32, 
                'proprio': tf.float32, 
                'gripper': tf.float32,
                'delta_eef_pos': tf.float32,
                'delta_eef_rotation': tf.float32,
            }
            output_shapes = {
                'image': self.image_size, 
                'proprio': self.proprio_size * self.n_proprio_stack, 
                'gripper': 1,
                'delta_eef_pos': (3,),
                'delta_eef_rotation': (4,),
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


class BC:
    def __init__(self, dataset_location, image_size, proprio_size, 
        action_size, batch_size, proprio_history=5, real=False, using_robot=False, use_resnet=False):

        self.proprio_history = proprio_history
        self.image_size = image_size
        self.proprio_size = (proprio_size[0] * self.proprio_history,)
        self.action_size = action_size
        self.batch_size = batch_size
        self.delta_size = (7,)
        self.real = real
        self.use_resnet = use_resnet

        self.dset = RoboTurkDataset(dataset_location, n_valid=1, real=real, using_robot=using_robot, 
                                    image_size=self.image_size, n_proprio_stack=proprio_history)
        self.dataset = self.dset.make_tf_dataset(self.batch_size)
        
        with tf.variable_scope('Inputs'):
            self.inputs = self.dataset.get_next()

        #with tf.variable_scope('BCModel'):
        self.build_model()
        self.build_model(eval=True)

        with tf.variable_scope('Loss'):
            self.build_loss()

        with tf.variable_scope('Summaries'):
            self.build_summaries()


    def build_summaries(self):
        tf.summary.scalar('L2', self.l2)
        tf.summary.scalar('L1', self.l1)
        tf.summary.scalar('LC', self.lc)
        tf.summary.scalar('LG', self.lg)
        tf.summary.scalar('LRegularization', self.regularization_loss)

        tf.summary.scalar('TotalLoss', self.loss)

        self.summary_op = tf.summary.merge_all()


    def build_model(self, eval=False):


        if self.use_resnet:
            resnet = resnet_model.Model(
                resnet_size=50,
                bottleneck=True, 
                num_classes=1001,
                num_filters=64,
                kernel_size=7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                block_sizes=[3, 4, 6, 3],
                block_strides=[1, 2, 2, 2],
                resnet_version=2, 
                data_format='channels_last',
                dtype=tf.float32
            )
            if eval:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.image_obs = tf.placeholder(shape=(1, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                    dtype=tf.float32, name='image_input')

                    outputs = resnet(self.image_obs, training=True)
                    self.eval_conv_out = tf.stop_gradient(tf.layers.flatten(outputs))
            else:
                self.image_obs = tf.placeholder(shape=(1, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                dtype=tf.float32, name='image_input')

                outputs = resnet(self.inputs['image'], training=True)
                self.conv_out = tf.stop_gradient(tf.layers.flatten(outputs))
        else:
            with tf.variable_scope('CNNStem', reuse=eval):
                if eval:
                    self.image_obs = tf.placeholder(shape=(1, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                    dtype=tf.float32, name='image_input')

                    hid = tf.layers.conv2d(self.image_obs, 16, 8, 4, activation=tf.nn.relu, name='conv1')
                else:
                    hid = tf.layers.conv2d(self.inputs['image'], 16, 8, 4, activation=tf.nn.relu, name='conv1')

                hid = tf.layers.conv2d(hid                 , 32, 4, 2, activation=tf.nn.relu, name='conv2')

                hid = tf.layers.flatten(hid, name='flatten')
                hid = tf.layers.dense(hid, 256, name='fc_conv_out', activation=tf.nn.relu)

                if eval:
                    self.eval_conv_out = tf.identity(hid)
                else:
                    self.conv_out = tf.identity(hid)

        with tf.variable_scope('concat', reuse=eval):
            if eval:
                self.proprio_obs = tf.placeholder(shape=(1, self.proprio_size[0]), dtype=tf.float32, name='proprio_input')
                self.eval_concatted = tf.concat([self.eval_conv_out, self.proprio_obs], -1, name='EvalFullInputs')
            else:
                self.concatted = tf.concat([self.conv_out, self.inputs['proprio']], -1, name='FullInputs')

        with tf.variable_scope('PredictAction', reuse=eval):
            if eval:
                hid = tf.layers.dense(self.eval_concatted, 256, name='fc1', activation=tf.nn.relu)
                hid = tf.layers.dense(hid, 256, name='fc2', activation=tf.nn.relu)

                self.eval_delta_pos = tf.layers.dense(hid, 3, activation=tf.nn.tanh, name='fc_pos')
                self.eval_delta_quat = tf.layers.dense(hid, 4, activation=tf.nn.tanh, name='fc_quat')
                self.eval_delta_quat = tf.nn.l2_normalize(self.eval_delta_quat, 1)

                self.eval_gripper_output = tf.layers.dense(hid, 1, name='fc_gripper')

            else:
                hid = tf.layers.dense(self.concatted, 256, name='fc1', activation=tf.nn.relu)
                hid = tf.layers.dense(hid, 256, name='fc2', activation=tf.nn.relu)

                self.delta_pos = tf.layers.dense(hid, 3, activation=tf.nn.tanh, name='fc_pos')
                self.delta_quat = tf.layers.dense(hid, 4, activation=tf.nn.tanh, name='fc_quat')
                self.delta_quat = tf.nn.l2_normalize(self.delta_quat, axis=-1)

                self.gripper_output = tf.layers.dense(hid, 1, name='fc_gripper')

    def get_action(self, image_obs, proprio_obs):
        sess = tf.get_default_session()
        image_obs -= self.dset.image_mean
        image_obs /= self.dset.image_std
        
        return sess.run([self.eval_delta_pos, self.eval_delta_quat, self.eval_gripper_output], feed_dict={
            self.image_obs: image_obs,
            self.proprio_obs: proprio_obs,
        })

    def build_loss(self):
        # TODO: Loss on location of object

        gt_delta_pos = self.inputs['delta_eef_pos']
        gt_delta_rotation = self.inputs['delta_eef_rotation']
        gt_gripper = tf.reshape(self.inputs['gripper'], (-1, 1))


        pred_delta_pos = self.delta_pos
        pred_delta_rotation = self.delta_quat
        pred_gripper = tf.reshape(self.gripper_output, (-1,1))

        self.l2 = tf.losses.mean_squared_error(gt_delta_pos, pred_delta_pos)

        gt_delta_rotation = tf.nn.l2_normalize(gt_delta_rotation, axis=-1)
        self.l1 = tf.reduce_mean( 1 - tf.square(tf.reduce_sum(tf.multiply(gt_delta_rotation, pred_delta_rotation), axis=-1)))

        
        norm_gt = tf.nn.l2_normalize(gt_delta_pos, 1)
        norm_pred = tf.nn.l2_normalize(pred_delta_pos, 1)

        self.lc = tf.losses.cosine_distance(norm_gt, norm_pred, axis=-1)

        #self.lg = tf.losses.sigmoid_cross_entropy(gt_gripper, pred_gripper)
        self.lg = tf.losses.absolute_difference(gt_gripper, pred_gripper)  # sim is not binary

        self.loss = 10 * ( 1. * self.l2 + 1* self.l1 + 1.0 * self.lg + 1. * self.lc )

        self.regularization_loss = 0.001 * tf.reduce_sum([
            tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'kernel' in var.name
        ])

        self.loss += self.regularization_loss

    def iterate(self):

        sess = tf.get_default_session()
        print(sess.run(self.inputs)['image'].shape)

def restore_env(env_config):
    """
    Restores the environment.
    """
    env, env_config = make_env(env_config, 'eval')
    return env, env_config

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--dataset-dir', type=str, default='./dataset_cropped')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_cropped')
    parser.add_argument('--logdir', type=str, default='./logs_new')
    parser.add_argument('--use-resnet', action='store_true')
    parser.add_argument('--restart', action='store_true')


    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.logdir
    use_resnet = args.use_resnet

    if not args.real:
        from benedict import BeneDict
        import mujoco_py


        """
        folder = '/home/jonathan/surreal-tmux/lift2'
        configs = restore_config(path.join(folder, 'config.yml'))
        env_config = configs.env_config

        env, env_config = restore_env(env_config)
        env = IKWrapper(env.unwrapped)

        env2, env_config2 = restore_env(env_config)

        def thunk():
            import os
            viewer = mujoco_py.MjViewer(env2.unwrapped.sim)

            while True:
                state = env.unwrapped.sim.get_state()
                env2.unwrapped.sim.set_state(state)
                env2.unwrapped.sim.step()
                #print(state)
                viewer.render()
        import threading
        thread = threading.Thread(target=thunk)
        thread.start()
        """

        images = np.load('{}/sim_images.npy'.format(dataset_dir))
        proprio = np.load('{}/sim_proprio.npy'.format(dataset_dir))
        dpos = np.load('{}/sim_dpos.npy'.format(dataset_dir))
        rotation = np.load('{}/sim_rotation.npy'.format(dataset_dir))

        eval_images = images[-1:]
        eval_proprio = proprio[-1:]
        eval_dpos = dpos[-1:]
        eval_rotation = rotation[-1:]


        b = BC(dataset_dir, (84,84,3), (3+4+3+1,), (8,), 32, real=False, use_resnet=use_resnet)
    else:
        thread = None
        import pickle
        
        images = np.load('{}/real_image.npy'.format(dataset_dir), encoding='bytes')
        proprio = np.load('{}/real_proprio.npy'.format(dataset_dir), encoding='bytes')
        dpos = np.load('{}/real_dpos.npy'.format(dataset_dir))
        rotation = np.load('{}/real_rotation.npy'.format(dataset_dir))

        eval_images = images[-1:]
        eval_proprio = proprio[-1:]
        eval_dpos = dpos[-1:]
        eval_rotation = rotation[-1:]

        b = BC(dataset_dir, (84,84,3), (3+3+4,), (8,), 32, real=True, use_resnet=use_resnet)


    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)

    lr = 0.001
    #lr = tf.train.cosine_decay_restarts(0.001, global_step, 10000)
    optimizer = tf.contrib.opt.AdamWOptimizer(0.001, learning_rate=lr) #better weight decay probably
    #optimizer = tf.train.AdamOptimizer()

    saver = tf.train.Saver(max_to_keep=5)
    train_op = optimizer.minimize(b.loss)
    writer = tf.summary.FileWriter(log_dir)

    iteration = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        if use_resnet:
            checkpoint = tf.train.latest_checkpoint('/home/jonathan/Desktop/modelzoo/resnet_imagenet_v2_fp32_20181001')
            variables = [var for var in tf.trainable_variables() if 'resnet' in var.name]
            s = tf.train.Saver(variables)
            s.restore(sess, checkpoint)
        elif args.restart:
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, checkpoint)

        writer.add_graph(sess.graph)

        while True:
            loss, summary, gstep, _ = sess.run([b.loss, b.summary_op, global_step, train_op])
            
            if iteration % 100 == 0:
                writer.add_summary(summary, global_step=gstep)
                writer.flush()
                print('Iteration {}\tLoss:{}'.format(iteration, loss))


            if iteration % 1000 == 0:
                saver.save(sess, '{}/model'.format(checkpoint_dir), global_step=gstep)
                if False: #not args.real:
                    reward = 0
                    ob = env.reset()
                    window = collections.deque(maxlen=5)
                    for _ in range(5):
                        window.append(np.zeros(3+3+4+1))

                    for i in range(200):
                        aux = env.unwrapped._get_observation()

                        window.append(np.concatenate([aux['eef_pos'], aux['eef_quat'], aux['gripper_qvel'], aux['gripper_qpos']]))

                        dpos, drot, gripper = b.get_action([np.transpose(ob['image'],(0,1,2))], np.array(window).reshape(1, -1))
                        drot = np.array([0.,0.,0.,1.])
                        drot = np.ravel(drot)
                        drot /= np.linalg.norm(drot)
                        a = np.concatenate([np.ravel(dpos), drot, np.ravel(gripper)])

                        ob, r, _, _ = env.step(a)
                        reward += r
                    print('Reward: {}'.format(reward))
                else:
                    losses = 0
                    for traj in range(1):
                        pos_direction_err = 0
                        quat_error = 0
                        for i in range(len(eval_proprio[traj])):
                            t_ind = traj
                            time_ind = i
                            if time_ind < 4:
                                to_stack = [ np.zeros(3+3+4+1) for _ in range(5 - time_ind - 1)]

                                if time_ind == 0:
                                    
                                    to_stack.append(eval_proprio[t_ind][time_ind])
                                else:
                                    to_stack.extend([
                                        eval_proprio[t_ind][time_ind -i] for i in range(time_ind+1)
                                    ])
                                proprio_stack = np.concatenate(to_stack)

                            else:
                                proprio_stack = np.concatenate([
                                    eval_proprio[t_ind][time_ind -i] for i in range(5 - 1, -1, -1)
                                ])

                            if proprio_stack.shape[0] != (3+3+4+1)*5:
                                continue
  
                            img, proprio = eval_images[traj][i], proprio_stack
                            img = cv2.resize(img, (84,84))
                            img = img.astype(np.float32)

                            dpos, drot, gripper = b.get_action([img], [proprio])

                            g_dpos = eval_dpos[traj][i]
                            if i == 0:
                                euler_rotation = np.array([0,0.57,1.5708])

                                if args.real:
                                    prev_rotation = RU.quat2mat(RU.euler2quat(euler_rotation))
                                else:
                                    prev_rotation = RU.euler2quat( euler_rotation)
                            else:
                                prev_rotation = eval_rotation[traj][i-1]
                            curr_rotation = eval_rotation[traj][i]

                            if curr_rotation.shape == (3,3):
                                curr_rotation = RU.mat2euler(curr_rotation)
                                prev_rotation = RU.mat2euler(prev_rotation)
                            else:
                                curr_rotation = RU.quat2euler(curr_rotation)
                                prev_rotation = RU.quat2euler(prev_rotation)

                            delta_eef_rotation = curr_rotation - prev_rotation
                            g_drot = RU.euler2quat(delta_eef_rotation)

                            dpos, drot, g_dpos, g_drot = np.ravel(dpos), np.ravel(drot), np.ravel(g_dpos), np.ravel(g_drot)

                            pos_direction_err += np.arccos(1. - scipy.spatial.distance.cosine(dpos, g_dpos)) #np.arccos(cos_err)
                            quat_error += np.arccos(2 * np.dot(g_drot, drot)**2 - 1) #np.sum(np.abs(g_drot - drot))

                        print('Traj {} MEAN errors: Pos: {}\tQuat: {}'.format(traj, pos_direction_err/len(eval_proprio[traj]), quat_error/len(eval_proprio[traj])))

            iteration += 1