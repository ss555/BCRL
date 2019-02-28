from BC import BC
import numpy as np
import tensorflow as tf

from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from RobotTeleop import make_robot, make_controller, make_config
import RobotTeleop.utils as U
import time

import cv2

import rospy

import collections

class RoboTurkSawyerEnv:
        def __init__(self, reward_func=None):
                self.config = make_config('RealSawyerDemoServerConfig')
                self.config.infer_settings()

                if self.config.controller.ik.control_with_orn:
                        self.action_space = (7,)
                else:
                        self.action_space = (4,)

                self.observation_space = (7,)

                self.robot = make_robot(self.config.robot.type, config=self.config)
                self.controller = make_controller(self.config.controller.type, robot=self.robot, config=self.config)

                self.controller.reset()
                self.controller.sync_state()

                self.gripper_open = True
                self.reward_func = reward_func

                self.last_t = time.time()

        def render(self):
                self.robot.render_to_screen()

        def reset(self):
                self.robot.reset()
                self.controller.sync_state()

                self.controller.reset()
                self.last_t = time.time()
                return self.get_observation()

        def sleep(self, time_delta=0.):
                time.sleep(max(0, (self.last_t + 
                        1.0 / self.config.control.rate) - time.time()))
                #time.sleep(0.02 - time_elapsed)
                #time.sleep(max(1.0 / self.config.control.rate - time_elapsed, 0))
        def get_observation(self):
                pose = self.robot.eef_pose()
                pos, orn = U.mat2pose(pose)
                return np.concatenate([pos, orn])

        def toggle_gripper(self, action):
                if action != 0. and self.gripper_open:
                        self.robot.control_gripper(1)
                        self.gripper_open = False
                elif action == 0. and not self.gripper_open:
                        self.robot.control_gripper(0)
                        self.gripper_open = True

        def get_reward(self):
                if self.reward_func is not None:
                        return self.reward_func(self.get_observation)
                return -1

        def get_success(self):
                return False

        def step(self, action, sensitivity=2, time_delta=None):
                """
                action assumed to be 
                [ delta_pos, delta_rot (euler angles), gripper_status]
                gripper will be closed when gripper_status is non 0
                """
                assert action.shape == self.action_space
                self.robot.robot_arm.blocking = False
                #starting_time = time.time()

                if self.config.controller.ik.control_with_orn:
                        #drot = U.euler2quat(action[3:6])
                        cur_rot = self.robot.eef_orientation()
                        cur_rot = U .mat2euler(cur_rot)
                        new_rot = cur_rot + action[3:6]
                        new_rot = U.euler2quat(new_rot)
                        new_rot = U.quat2mat(new_rot)
                else:
                        new_rot = self.robot.eef_orientation()

                dpos = np.clip(np.array(action[:3]), -self.config.control.max_dpos, self.config.control.max_dpos)
                action = {
                        'dpos': dpos,
                        'rotation': new_rot,
                        'timestamp': time.time(),
                        'engaged': True,
                        'zoom': 0,
                        'sensitivity': sensitivity,
                        'valid': True,
                        'grasp': action[-1]
                }
                self.toggle_gripper(action['grasp'])

                self.controller.apply_control(action)

                self.sleep(time_delta)
                self.last_t = time.time()
                #self.sleep(time.time() - starting_time)

                obs = self.get_observation()
                reward = self.get_reward()
                done = self.get_success()
                self.robot.robot_arm.blocking = True
                return obs, reward, done, {}


class RobotState:

	def __init__(self):

		self.joint_state_reciever = rospy.Subscriber(
			'/robot/joint_states',
			JointState,
			self.joint_state_callback
		)

		self.image_reciever = rospy.Subscriber(
			'/usb_cam/image_raw/compressed',
			CompressedImage,
			self.image_callback
		)

		self.polled = True
		self.image = None
		self.joint_states = collections.deque(maxlen=5)

		self.curr_joint_state = None

		for _ in range(5):
			self.joint_states.append(np.zeros((18,)))

	def poll(self):
		assert self.image is not None
		assert self.curr_joint_state is not None

                assert self.curr_joint_state.shape == (18,)
		self.joint_states.append(self.curr_joint_state)
		return {
			'image': self.image,
			'joint_state': np.concatenate(self.joint_states, axis=-1)
		}

	def joint_state_callback(self, message):

		joint_state = np.concatenate([
			message.position, message.velocity
		])

                if joint_state.shape == (18,):
		      self.curr_joint_state = joint_state

                #print(self.curr_joint_state)


	def image_callback(self, message):
		img = CvBridge().compressed_imgmsg_to_cv2(message)
		img = cv2.resize(img, (84,84))

		self.image = img		


if __name__ == '__main__':

	robot_state = RobotState()

	env = RoboTurkSawyerEnv()

        env.reset()

	b = BC('./dataset/real', (84,84,3), (18,), (8,), 32, real=True, using_robot=True)

        saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint('./checkpoints')

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, checkpoint)
        	for i in range(25):

        		state = robot_state.poll()

        		image = (state['image'] - b.image_mean) / (b.image_std + 1.e-9)

        		action = b.get_action([image], [state['joint_state']])

                        action[0] = action[0][0]
                        action[2] = action[2][0]
                        #print(action)
                        action[1] = U.quat2euler(action[1][0])
                        
                        if i == 0:
                                action[1] *= 0
                        #if action[1][0] < -1.:
                        #        action[1] = U.quat2euler(-action[1][0])

                        print(action[1])
                        #action[1] *= 0
                        action = np.concatenate(action, axis=-1)
                        action[-1] *= 0
        		env.step(action)

                        time.sleep(0.1)
                        """
                        char = None
        		while char != 'y' and char != 'n':
        			char = raw_input('Continue? (y/n)')

                        if char == 'n':
                                break
                        """