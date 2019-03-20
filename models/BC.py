import tensorflow as tf
import numpy as np
import cv2
import os
from os import path
import json

from .optimizer import ADAMW

import RobotTeleop.utils as RU

try:
    from official.resnet import resnet_model
    RESNET_CAPABLE = True
except:
    RESNET_CAPABLE = False
    print('Cannot import the tensorflow models repository')

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
from .dataset_loader import RoboTurkDataset

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

        self.dset = RoboTurkDataset(dataset_location, n_valid=5, real=real, using_robot=using_robot, 
                                    image_size=self.image_size, n_proprio_stack=proprio_history)
        self.dataset = self.dset.make_tf_dataset(self.batch_size)
        
        with tf.variable_scope('Inputs'):
            self.inputs = self.dataset.get_next()

        #with tf.variable_scope('BCModel'):
        self.build_model()
        self.build_model(eval=True)

        with tf.variable_scope('Loss'):
            self.build_loss()

        with tf.variable_scope('Summaries', reuse=tf.AUTO_REUSE):
            self.build_summaries()


    def build_summaries(self):
        tf.summary.scalar('L2', self.l2)
        tf.summary.scalar('L1', self.l1)
        tf.summary.scalar('LC', self.lc)
        tf.summary.scalar('LG', self.lg)
        tf.summary.scalar('LEEF', self.leef)
        tf.summary.scalar('LGOAL', self.lgoal)

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
                    inpt = self.image_obs
                else:
                    inpt = self.inputs['image']

                hid = tf.layers.conv2d(inpt, 64, kernel_size=7, strides=2, padding='same', activation=tf.nn.relu, name='conv1')
                hid = tf.layers.conv2d(hid, 32, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu, name='conv2')
                hid = tf.layers.conv2d(hid, 32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv3')
                hid = tf.layers.conv2d(hid, 32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv4')

                if eval:
                    hid = tf.contrib.layers.spatial_softmax(hid)
                    self.eval_eef_predict = tf.stop_gradient(tf.layers.dense(hid, 3, name='eef_pos_predict'))
                    self.eval_goal_predict = tf.stop_gradient(tf.layers.dense(hid, 3, name='eef_goal_pos_predict'))
                else:
                    self.spatial_softmax = tf.contrib.layers.spatial_softmax(hid)
                    self.eef_predict = tf.layers.dense(self.spatial_softmax, 3, name='eef_pos_predict')
                    self.goal_predict = tf.layers.dense(self.spatial_softmax, 3, name='eef_goal_pos_predict')
                    hid = self.spatial_softmax

                hid = tf.layers.flatten(hid, name='flatten')
                hid = tf.layers.dense(hid, 256, name='fc_conv_out', activation=tf.nn.relu)

                if eval:
                    self.eval_conv_out = tf.identity(hid)
                else:
                    self.conv_out = tf.identity(hid)

        with tf.variable_scope('concat', reuse=eval):
            if eval:
                self.proprio_obs = tf.placeholder(shape=(None, self.proprio_size[0]), dtype=tf.float32, name='proprio_input')
                #self.eval_concatted = tf.concat([self.eval_conv_out, self.proprio_obs], -1, name='EvalFullInputs')
                #self.eval_concatted = tf.concat([self.eval_conv_out, self.proprio_obs,
                #                                self.eval_eef_predict, self.eval_goal_predict], -1, name='EvalFullInput')

                self.eval_concatted = tf.concat([self.eval_conv_out, self.proprio_obs,
                                                 self.eval_eef_predict], -1, name='EvalFullInput')


            else:
                #self.concatted = tf.concat([self.conv_out, self.inputs['proprio']], -1, name='FullInputs')
                #self.concatted = tf.concat([self.conv_out, self.inputs['proprio'],
                #                           tf.stop_gradient(self.eef_predict),
                #                           tf.stop_gradient(self.goal_predict)], -1, name='FullInput')

                self.concatted = tf.concat([self.conv_out, self.inputs['proprio'],
                                            self.eef_predict], -1, name='FullInput')

        with tf.variable_scope('PredictAction', reuse=eval):
            if eval:
                #hid = tf.layers.dense(self.eval_concatted, 256, name='fc1', activation=tf.nn.relu)
                self.value_hid = tf.identity(hid)

                hid = tf.layers.dense(hid, 256, name='fc2', activation=tf.nn.relu)

                self.eval_delta_pos = tf.layers.dense(hid, 3, activation=tf.nn.tanh, name='fc_pos')
                self.eval_delta_quat = tf.layers.dense(hid, 4, activation=tf.nn.tanh, name='fc_quat')
                self.eval_delta_quat = tf.nn.l2_normalize(self.eval_delta_quat, 1)

                self.eval_gripper_output = tf.layers.dense(hid, 1, name='fc_gripper')

            else:
                #hid = tf.layers.dense(self.concatted, 512, name='fc1', activation=tf.nn.relu)

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
        gt_delta_pos = self.inputs['delta_eef_pos']
        gt_delta_rotation = self.inputs['delta_eef_rotation']
        gt_gripper = tf.reshape(self.inputs['gripper'], (-1, 1))
        gt_eef = self.inputs['eef']
        gt_goal = self.inputs['goal']

        gt_eef_pos = gt_eef[:,:3]

        pred_delta_pos = self.delta_pos
        pred_delta_rotation = self.delta_quat
        pred_gripper = tf.reshape(self.gripper_output, (-1,1))

        self.leef = tf.losses.mean_squared_error(gt_eef_pos, self.eef_predict)
        self.lgoal = 0 #tf.losses.mean_squared_error(gt_goal, self.goal_predict)

        self.l2 = tf.losses.mean_squared_error(gt_delta_pos, pred_delta_pos)

        gt_delta_rotation = tf.nn.l2_normalize(gt_delta_rotation, axis=-1)
        self.l1 = tf.reduce_mean( 1 - tf.square(tf.reduce_sum(tf.multiply(gt_delta_rotation, pred_delta_rotation), axis=-1)))

        
        norm_gt = tf.nn.l2_normalize(gt_delta_pos, 1)
        norm_pred = tf.nn.l2_normalize(pred_delta_pos, 1)

        self.lc = tf.losses.cosine_distance(norm_gt, norm_pred, axis=-1)

        #self.lg = tf.losses.sigmoid_cross_entropy(gt_gripper, pred_gripper)
        self.lg = tf.losses.absolute_difference(gt_gripper, pred_gripper)  # sim is not binary

        self.loss = ( 1. * self.l2 + 1* self.l1 + 1.0 * self.lg + 5.* self.lc + 1. * self.leef + 1. * self.lgoal)

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
    import time

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

    if use_resnet and not RESNET_CAPABLE:
        raise ValueError('You do not have the tensorflow models repo cloned and in your PYTHONPATH')
    if not args.real and not SIM_CAPABLE:
        raise ValueError('You do not surreal or robosuite and thus cannot work in sim')

    if args.use_resnet:
        img_size = (224, 224)
    else:
        img_size = (84, 84)

    global_step = tf.train.get_or_create_global_step()

    lr = 0.0001
    #lr = tf.train.exponential_decay(0.001, global_step, 100000, 0.95, staircase=True) 
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    #optimizer = ADAMW(lr, 0.001, batch_size=32, epoch_size=10000, epochs=400)

    with tf.variable_scope('Summaries'):
        tf.summary.scalar('LR', lr)


    if not args.real:
        folder = '/home/jonathan/surreal-tmux/lift2'
        configs = restore_config(path.join(folder, 'config.yml'))
        env_config = configs.env_config

        env_config.render = False


        env, env_config = restore_env(env_config)
        env = IKWrapper(env.unwrapped, action_repeat=15)
 
        env2, env_config2 = restore_env(env_config)

        def thunk():
            import os
            viewer = mujoco_py.MjViewer(env2.unwrapped.sim)

            while True:
                state = env.unwrapped.sim.get_state()
                env2.unwrapped.sim.set_state(state)
                env2.unwrapped.sim.step()
                viewer.render()
        import threading
        thread = threading.Thread(target=thunk)
        thread.start()
        
        b = BC(dataset_dir, (*img_size,3), (30,), (8,), 32, real=False, use_resnet=use_resnet)

        eval_images = b.dset.eval_images
        eval_proprio = b.dset.eval_proprio
        eval_dpos = b.dset.eval_dpos
        eval_rotation = b.dset.eval_rotation
        eval_eef = b.dset.eval_eef
    else:
        thread = None
        import pickle

        b = BC(dataset_dir, (*img_size,3), (18,), (8,), 32, real=True, use_resnet=use_resnet)

        eval_images = b.dset.eval_images
        eval_proprio = b.dset.eval_proprio
        eval_dpos = b.dset.eval_dpos
        eval_rotation = b.dset.eval_rotation
        eval_eef = b.dset.eval_eef

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    saver = tf.train.Saver(max_to_keep=5)
    train_op = optimizer.minimize(b.loss, global_step=global_step)
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
                if not args.real:
                    reward = 0
                    ob = env.reset()
                    window = collections.deque(maxlen=5)
                    for _ in range(5):
                        window.append(np.zeros(b.dset.proprio_size))

                    for i in range(200):
                        start = time.time()
                        aux = env.unwrapped._get_observation()
                        #print(aux)
                        window.append(np.array(aux['robot-state']).ravel())
                        #window.append(np.concatenate([aux['eef_pos'], aux['eef_quat'], aux['gripper_qvel'], aux['gripper_qpos']]))
                        img = cv2.resize(ob['image'], img_size)
                        imgs = np.transpose(img, (1,0,2))
                        dpos, drot, gripper = b.get_action([img], np.array(window).reshape(1, -1))
                        #drot = np.array([0.,0.,0.,1.])
                        drot = np.ravel(drot)
                        drot /= np.linalg.norm(drot)
                        a = np.concatenate([np.ravel(dpos), drot, np.ravel(gripper)])

                        ob, r, _, _ = env.step(a)

                        elapsed = time.time() - start
                        #print(elapsed, )
                        #time.sleep(max(start + 1./10 - time.time(), 0))
                        reward += r
                    print('Reward: {}'.format(reward))
                else:
                    losses = 0
                    for traj in range(b.dset.n_valid):
                        pos_direction_err = 0
                        quat_error = 0
                        for i in range(len(eval_proprio[traj])-1):
                            t_ind = traj
                            time_ind = i
                            if time_ind < 4:
                                to_stack = [ np.zeros(b.dset.proprio_size) for _ in range(5 - time_ind - 1)]

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

                            if proprio_stack.shape[0] != (b.dset.proprio_size)*5:
                                continue
  
                            img, proprio = eval_images[traj][i], proprio_stack
                            img = cv2.resize(img, img_size)
                            img = np.transpose(img, (1,0,2))
                            img = img.astype(np.float32)

                            dpos, drot, gripper = b.get_action([img], [proprio])

                            g_dpos = eval_eef[traj][i+1][:3] - eval_eef[traj][i][:3] #eval_dpos[traj][i]
                            if i == 0:
                                euler_rotation = np.array([0,0.57,1.5708])

                                if args.real:
                                    prev_rotation = RU.quat2mat(RU.euler2quat(euler_rotation))
                                else:
                                    prev_rotation = RU.euler2quat( euler_rotation)
                            else:
                                prev_rotation = eval_rotation[traj][i-1]
                            curr_rotation = np.array(eval_rotation[traj][i])

                            if curr_rotation.shape == (3,3):
                                curr_rotation = RU.mat2euler(curr_rotation)
                                prev_rotation = RU.mat2euler(prev_rotation)
                            else:
                                curr_rotation = RU.quat2euler(curr_rotation)
                                prev_rotation = RU.quat2euler(prev_rotation)

                            delta_eef_rotation = curr_rotation - prev_rotation
                            g_drot = RU.euler2quat(delta_eef_rotation)

                            dpos, drot, g_dpos, g_drot = np.ravel(dpos), np.ravel(drot), np.ravel(g_dpos), np.ravel(g_drot)

                            check = 1. - scipy.spatial.distance.cosine(dpos, g_dpos)
                            if not np.isnan(check):
                                pos_direction_err += np.arccos(check) #np.arccos(cos_err)
                            quat_error += np.arccos(2 * np.dot(g_drot, drot)**2 - 1) #np.sum(np.abs(g_drot - drot))

                        print('Traj {} MEAN errors: Pos: {}\tQuat: {}'.format(traj, pos_direction_err/len(eval_proprio[traj]), quat_error/len(eval_proprio[traj])))

            iteration += 1
