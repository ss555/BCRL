import tensorflow as tf
import numpy as np
import cv2
import os
from os import path

import RobotTeleop.utils as RU


def make_random_dataset():

    lst = []

    for _ in range(10):
        lst.append({
            'image': np.random.randint(low=0, high=255, size=(84,84,3)),
            'proprio': np.random.randn(10),
            'action': np.random.randn(7)
        })

    np.save("./tmp.npy", lst)


def log_quaternion_loss_batch(predictions, labels, batch_size):

    assertions = []

    
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1),
                    1e-4)),
            [tf.norm(predictions, axis=-1)]))
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 1e-4)),
            ['The l2 norm of each label quaternion vector should be 1.']))
    
    with tf.control_dependencies(assertions):
        product = tf.multiply(predictions, labels)
    internal_dot_products = tf.reduce_sum(product, [1])

    logcost = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
    return logcost


def log_quaternion_loss(predictions, labels, batch_size):

    logcost = log_quaternion_loss_batch(predictions, labels, batch_size)
    logcost = tf.reduce_sum(logcost, [0])
    logcost = tf.multiply(logcost, 1.0 / batch_size, name='log_quaternion_loss')
    #if use_logging:
    #  logcost = tf.Print(
    #      logcost, [logcost], '[logcost]', name='log_quaternion_loss_print')
    return logcost

def quaternion_loss(labels, predictions, batch_size):
    loss = log_quaternion_loss(labels, predictions, batch_size)

    assert_op = tf.Assert(tf.is_finite(loss), [loss])
    with tf.control_dependencies([assert_op]):
        tf.summary.histogram(
            'Log_Quaternion_Loss', loss, collections='losses')
        tf.summary.scalar(
            'Task_Quaternion_Loss', loss, collections='losses')

    return loss

class BC:
    def __init__(self, dataset_location, image_size, proprio_size, 
        action_size, batch_size, proprio_history=5, real=False, using_robot=False):

        self.proprio_history = proprio_history
        self.image_size = image_size
        self.proprio_size = (proprio_size[0] * self.proprio_history,)
        self.action_size = action_size
        self.batch_size = batch_size
        self.delta_size = (7,)
        self.real = real

        if not using_robot:
            if self.real:
                self.np_images = np.load(dataset_location + '_image.npy', encoding='bytes')
                self.np_proprio = np.load(dataset_location + '_proprio.npy', encoding='bytes')
                self.np_dpos = np.load(dataset_location + '_dpos.npy')
                self.np_rotation = np.load(dataset_location + '_rotation.npy')

                self.np_gripper = np.load(dataset_location + '_gripper.npy')
            else:
                self.np_dataset = np.load(dataset_location)
                if dataset_location[-3:] == 'npz':
                    self.np_dataset = self.mp_dataset['arr_0']

            if self.real:
                self.np_images = self.np_images[:-1]
                self.np_proprio = self.np_proprio[:-1]
                self.np_dpos = self.np_dpos[:-1]
                self.np_rotation = self.np_rotation[:-1]
                self.np_gripper = self.np_gripper[:-1]
            else:
                self.np_dataset = self.np_dataset[:-1]

            if self.real:
                images = np.stack([self.np_images[i][j] for i in range(self.np_images.shape[0]) for j in range(len(self.np_images[i]))])
            else:
                images = np.stack([self.np_dataset[i]['image'] for i in range(self.np_dataset.shape[0])])
            self.image_mean = np.mean(images, (0, 1, 2)).copy()
            self.image_std = np.std(images, (0, 1, 2)).copy()

            del images
        else:
            self.image_mean = np.array([107.69177, 113.32283, 118.14253])
            self.image_std = np.array([57.7754, 60.47865, 80.9573])

        def real_dataset_generator():

            traj_inds = np.random.randint(0, self.np_images.shape[0], 1000)
            for traj_ind in range(1000):
                t_ind = traj_inds[traj_ind]
                traj = self.np_images[t_ind]

                time_ind = np.random.randint(0, len(traj))

                if time_ind < 4:
                    to_stack = [ np.zeros(proprio_size[0]) for _ in range(5 - time_ind - 1)]

                    if time_ind == 0:
                        to_stack.append(self.np_proprio[t_ind][time_ind])
                    else:
                        to_stack.extend([
                            self.np_proprio[t_ind][time_ind -i] for i in range(time_ind+1)
                        ])
                    proprio_stack = np.concatenate(to_stack)

                else:
                    proprio_stack = np.concatenate([
                        self.np_proprio[t_ind][time_ind -i] for i in range(self.proprio_history - 1, -1, -1)
                    ])

                if proprio_stack.shape[0] != self.proprio_size[0]:
                    continue

                delta_eef_pos = self.np_dpos[t_ind][time_ind]

                if time_ind == 0:
                    euler_rotation = np.array([0,0.57,1.5708])
                    prev_rotation = RU.quat2mat(RU.euler2quat(euler_rotation))
                else:
                    prev_rotation = self.np_rotation[t_ind][time_ind-1]
                curr_rotation = self.np_rotation[t_ind][time_ind]

                curr_rotation = RU.mat2euler(curr_rotation)
                prev_rotation = RU.mat2euler(prev_rotation)

                delta_eef_rotation = curr_rotation - prev_rotation
                delta_eef_quat = RU.euler2quat(delta_eef_rotation)
                                
                image = self.np_images[t_ind][time_ind]
                image = cv2.resize(image, self.image_size[:-1])

                gripper = self.np_gripper[t_ind][time_ind]

                yield {
                    'image': np.array((image - self.image_mean) / (self.image_std + 1e-9)),
                    'proprio': np.array(proprio_stack),
                    'gripper': gripper,
                    'delta_eef_pos': np.array(delta_eef_pos),
                    'delta_eef_rotation': np.array(delta_eef_quat),
                }


        def dataset_generator():
            data_inds = np.random.randint(0, self.np_dataset.shape[0], 1000)
            for _data_ind in range(1000):
                data_ind = data_inds[_data_ind]
                
                if data_ind < 4 or data_ind % 200 < 4:
                    offset = 0 if data_ind < 4 else data_ind // 200

                    if offset:
                        data_ind = data_ind % 200

                    to_stack = [ np.zeros(proprio_size[0]) for _ in range(5 - data_ind - 1)]

                    if data_ind == 0:
                        to_stack.append(self.np_dataset[data_ind+offset]['proprio'])
                    else:
                        to_stack.extend([
                            self.np_dataset[data_ind+offset -i]['proprio'] for i in range(data_ind+1)
                        ])
                    proprio_stack = np.concatenate(to_stack)

                else:
                    proprio_stack = np.concatenate([
                        self.np_dataset[data_ind -i]['proprio'] for i in range(self.proprio_history - 1, -1, -1)
                    ])

                yield {
                    'image': (self.np_dataset[data_ind]['image'] - self.image_mean) / (self.image_std + 1e-9),
                    'proprio': proprio_stack,
                    'gripper': self.np_dataset[data_ind]['action'][-1],
                    'delta_eef_pos': self.np_dataset[data_ind]['delta_eef_pos'],
                    'delta_eef_rotation': self.np_dataset[data_ind]['delta_eef_rotation'],
                }      


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
                'proprio': self.proprio_size, 
                'gripper': 1,
                'delta_eef_pos': (3,),
                'delta_eef_rotation': (4,),
            }

            if self.real:
                self.dataset = tf.data.Dataset.from_generator(real_dataset_generator,
                    output_types=output_types, 
                    output_shapes=output_shapes).shuffle(10).repeat().batch(self.batch_size).make_one_shot_iterator()
            else:
                self.dataset = tf.data.Dataset.from_generator(dataset_generator,
                    output_types=output_types, 
                    output_shapes=output_shapes).shuffle(10).repeat().batch(self.batch_size).make_one_shot_iterator()

        with tf.variable_scope('Inputs'):
            self.inputs = self.dataset.get_next()

        with tf.variable_scope('BCModel'):
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

        with tf.variable_scope('CNNStem', reuse=eval):
            if eval:
                self.image_obs = tf.placeholder(shape=(1, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=tf.float32, name='image_input')
                hid = tf.layers.conv2d(self.image_obs, 16, 8, 4, activation=tf.nn.relu, name='conv1')
            else:
                hid = tf.layers.conv2d(self.inputs['image'], 16, 8, 4, activation=tf.nn.relu, name='conv1')

            hid = tf.layers.conv2d(hid                 , 32, 4, 2, activation=tf.nn.relu, name='conv2')
        
            shape = hid.shape.as_list()

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
                #hid = tf.layers.dense(hid, self.delta_size[0] + 1, name='fc_out')
                
                self.eval_delta_pos = tf.layers.dense(hid, 3, name='fc_pos') #tf.identity(hid[:,:3])
                self.eval_delta_quat = tf.layers.dense(hid, 4, activation=tf.nn.tanh, name='fc_quat') #tf.identity(hid[:,3:7])
                self.eval_delta_quat = tf.nn.l2_normalize(self.eval_delta_quat, 1)

                self.eval_gripper_output = tf.layers.dense(hid, 1, name='fc_gripper') #tf.identity(hid[:,-1])

            else:
                hid = tf.layers.dense(self.concatted, 256, name='fc1', activation=tf.nn.relu)
                hid = tf.layers.dense(hid, 256, name='fc2', activation=tf.nn.relu)
                
                #hid = tf.layers.dense(hid, self.delta_size[0] + 1, name='fc_out')
                self.delta_pos = tf.layers.dense(hid, 3, name='fc_pos') #tf.identity(hid[:,:3])
                self.delta_quat = tf.layers.dense(hid, 4, activation=tf.nn.tanh, name='fc_quat') #tf.identity(hid[:,3:7])
                self.delta_quat = tf.nn.l2_normalize(self.delta_quat, axis=-1)

                self.gripper_output = tf.layers.dense(hid, 1, name='fc_gripper') #tf.identity(hid[:,-1])

    def get_action(self, image_obs, proprio_obs):
        sess = tf.get_default_session()

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

        self.l2 = tf.losses.absolute_difference(gt_delta_pos, pred_delta_pos) #tf.losses.huber_loss(gt_delta_pos, pred_delta_pos, delta=1.)
        
        gt_delta_rotation = tf.nn.l2_normalize(gt_delta_rotation, axis=-1)
        #self.l1 = quaternion_loss(gt_delta_rotation, pred_delta_rotation, self.batch_size)
        self.l1 = tf.reduce_mean( 1 - tf.square(tf.reduce_sum(tf.multiply(gt_delta_rotation, pred_delta_rotation), axis=-1)))


        norm_gt = tf.nn.l2_normalize(gt_delta_pos, 1)
        norm_pred = tf.nn.l2_normalize(pred_delta_pos, 1)

        print(norm_gt.shape, norm_pred.shape)
        self.lc = tf.losses.cosine_distance(norm_gt, norm_pred, axis=-1)

        self.lg = tf.losses.absolute_difference(gt_gripper, pred_gripper)

        #self.reconstruct_error = tf.losses.mean_squared_error(self.inputs['image'], self.reconstruct)
        self.loss = 1. * self.l2 + 0.125* self.l1 + 1.0 * self.lg + 1. * self.lc # + self.reconstruct_error * 0.1
        
        self.regularization_loss = 0.001 * tf.reduce_sum([
            tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'kernel' in var.name
        ])

    def iterate(self):

        sess = tf.get_default_session()
        print(sess.run(self.inputs)['image'].shape)

def restore_env(env_config):
    """
    Restores the environment.
    """
    #env_config.eval_mode.render = True
    env, env_config = make_env(env_config, 'eval')
    return env, env_config

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs


if __name__ == '__main__':
    from surreal.env import *
    import surreal.utils as U
    from robosuite.wrappers import IKWrapper
    import collections
    import matplotlib.pyplot as plt
    import argparse
    import scipy

    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')

    args = parser.parse_args()

    if not args.real:
        from benedict import BeneDict

        folder = '/home/jonathan/surreal-tmux/lift2'
        configs = restore_config(path.join(folder, 'config.yml'))
        env_config = configs.env_config

        env, env_config = restore_env(env_config)
        env = IKWrapper(env.unwrapped)

        eval_dataset = np.load('./dataset/states_no_noise.npy')
        eval_dataset = eval_dataset[-5*200:]


        b = BC('./dataset/states_no_noise.npy', (84,84,3), (30,), (8,), 32)
    else:
        import pickle
        
        images = np.load('./dataset/real_image.npy', encoding='bytes')
        proprio = np.load('./dataset/real_proprio.npy', encoding='bytes')
        dpos = np.load('./dataset/real_dpos.npy')
        rotation = np.load('./dataset/real_rotation.npy')

        eval_images = images[-1:]
        eval_proprio = proprio[-1:]
        eval_dpos = dpos[-1:]
        eval_rotation = rotation[-1:]

        b = BC('./dataset/real', (84,84,3), (18,), (8,), 32, real=True)


    if not os.path.isdir('./logs'):
        os.makedirs('./logs')
    if not os.path.isdir('./checkpoints'):
        os.makedirs('./checkpoints')

    optimizer = tf.train.AdamOptimizer()
    saver = tf.train.Saver(max_to_keep=5)
    train_op = optimizer.minimize(b.loss)
    writer = tf.summary.FileWriter('./logs')
    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)
    iteration = 0
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        while True:
            loss, summary, gstep, _ = sess.run([b.loss, b.summary_op, global_step, train_op])
            
            if iteration % 100 == 0:
                writer.add_summary(summary, global_step=gstep)
                writer.flush()
                print('Iteration {}\tLoss:{}'.format(iteration, loss))


            if iteration % 1000 == 0:
                saver.save(sess, './checkpoints/model', global_step=gstep)
                if not args.real:
                    reward = 0
                    ob = env.reset()
                    #print(ob)
                    window = collections.deque(maxlen=5)
                    for _ in range(5):
                        window.append(np.zeros_like(ob['robot-state']))

                    for i in range(200):
                        dpos, drot, gripper = b.get_action([np.transpose(ob['image'],(0,1,2))], np.array([window]).reshape(1, -1))

                        drot = np.ravel(drot)
                        drot /= np.linalg.norm(drot)
                        a = np.concatenate([np.ravel(dpos), drot, np.ravel(gripper)])
                        ob, r, _, _ = env.step(a)
                        window.append(ob['robot-state'])
                        reward += r
                    print('Reward: {}'.format(reward))
                else:
                    losses = 0
                    for traj in range(1):
                        pos_direction_err = 0
                        quat_error = 0
                        for i in range(len(eval_proprio[traj])):
                            #i = 5
                            #traj = 0
                            t_ind = traj
                            time_ind = i
                            if time_ind < 4:
                                to_stack = [ np.zeros(18) for _ in range(5 - time_ind - 1)]

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

                            if proprio_stack.shape[0] != 90:
                                continue
                            
                            #proprio_stack = np.concatenate([
                            #    eval_proprio[traj][i-j] for j in range(4, -1, -1)
                            #])

                            img, proprio = eval_images[traj][i], proprio_stack #eval_dataset[i]['proprio']
                            img = cv2.resize(img, (84,84))
                            img = img.astype(np.float32)
                            img -= b.image_mean
                            img /= (b.image_std + 1e-9)

                            dpos, drot, gripper = b.get_action([img], [proprio])

                            #g_dpos, g_drot = eval_dpos[traj][i] , eval_rotation[traj][i]
                            g_dpos = eval_dpos[traj][i]
                            if i == 0:
                                euler_rotation = np.array([0,0.57,1.5708])
                                prev_rotation = RU.quat2mat(RU.euler2quat(euler_rotation))
                            else:
                                prev_rotation = eval_rotation[traj][i-1]
                            curr_rotation = eval_rotation[traj][i]

                            curr_rotation = RU.mat2euler(curr_rotation)
                            prev_rotation = RU.mat2euler(prev_rotation)

                            delta_eef_rotation = curr_rotation - prev_rotation
                            g_drot = RU.euler2quat(delta_eef_rotation)

                            dpos, drot, g_dpos, g_drot = np.ravel(dpos), np.ravel(drot), np.ravel(g_dpos), np.ravel(g_drot)

                            #cos_err = np.dot(dpos, g_dpos) / (np.linalg.norm(g_dpos))
                            pos_direction_err += np.arccos(1. - scipy.spatial.distance.cosine(dpos, g_dpos)) #np.arccos(cos_err)
                            quat_error += np.arccos(2 * np.dot(g_drot, drot)**2 - 1) #np.sum(np.abs(g_drot - drot))

                            #err_dpos = np.sum(np.abs(dpos - g_dpos))
                            #print(np.concatenate([dpos, drot], -1), np.concatenate([g_dpos, g_drot]))

                        print('Traj {} MEAN errors: Pos: {}\tQuat: {}'.format(traj, pos_direction_err/len(eval_proprio[traj]), quat_error/len(eval_proprio[traj])))
            """
            if iteration % 1000 == 0:
                ret = 0
                ob, info = env.reset()
                for i in range(200):
                    a = b.get_action([np.transpose(ob['pixel']['camera0'],(1,2,0))], [ob['low_dim']['flat_inputs']])
                    a = a[0]
                    ob, r, _, _ = env.step(a)
                    ret += r
                print("return: {}".format(ret))
            """
            iteration += 1
