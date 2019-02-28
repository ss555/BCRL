import numpy as np
import json

if __name__ == '__main__':


	dataset = np.load('./dataset/real.npy')

	images = []
	proprio = []
	dpos = []
	rotation = []

	for traj in dataset:
		buff = {'image': [], 'proprio': [], 'dpos': [], 'rotation': [] }
		for time_step in traj:
			buff['image'].append(time_step['image']['image'])

			user_data = json.loads(time_step['control']['user_data'])['user_info']
			#print(user_data.keys())
			buff['proprio'].append(np.concatenate([time_step['joint_states']['position'], time_step['joint_states']['velocity']]))
			buff['dpos'].append(user_data['dpos'])
			buff['rotation'].append(user_data['rotation'])

		images.append(buff['image'])
		proprio.append(buff['proprio'])
		dpos.append(buff['dpos'])
		rotation.append(buff['rotation'])


	np.save('./dataset/real_image', images)
	np.save('./dataset/real_proprio', proprio)
	np.save('./dataset/real_dpos', dpos)
	np.save('./dataset/real_rotation', rotation) 
