import numpy as np
import json

if __name__ == '__main__':

	folder = 'dataset_v2'
	dataset = np.load('./{}/real.npy'.format(folder))

	images, proprio, dpos, rotation, grasp = [], [], [], [], []

	for traj in dataset:
		buff = {'image': [], 'proprio': [], 'dpos': [], 'rotation': [], 'grasp': [] }
		for time_step in traj:
			user_data = json.loads(time_step['control']['user_data'])['user_info']
			
			buff['image'].append(time_step['image']['image'])
			buff['proprio'].append(np.concatenate([time_step['joint_states']['position'], 
												   time_step['joint_states']['velocity']]))
			buff['dpos'].append(user_data['dpos'])
			buff['rotation'].append(user_data['rotation'])
			buff['grasp'].append(user_data['grasp'])

		images.append(buff['image'])
		proprio.append(buff['proprio'])
		dpos.append(buff['dpos'])
		rotation.append(buff['rotation'])
		grasp.append(buff['grasp'])

	np.save('./{}/real_image'.format(folder), images)
	np.save('./{}/real_proprio'.format(folder), proprio)
	np.save('./{}/real_dpos'.format(folder), dpos)
	np.save('./{}/real_rotation'.format(folder), rotation) 
	np.save('./{}/real_grasp'.format(folder), grasp)