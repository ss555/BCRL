from BC import BC
import numpy as np
import tensorflow as tf
import cv2
import RobotTeleop.utils as RU
import scipy

if __name__ == '__main__':

    images = np.load('./dataset/real_image.npy', encoding='bytes')
    proprio = np.load('./dataset/real_proprio.npy', encoding='bytes')
    dpos = np.load('./dataset/real_dpos.npy')
    rotation = np.load('./dataset/real_rotation.npy')

    eval_images = images #[-1:]
    eval_proprio = proprio #[-1:]
    eval_dpos = dpos #[-1:]
    eval_rotation = rotation #[-1:]

    b = BC('./dataset/real', (84,84,3), (18,), (8,), 32, real=True)
    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint('./checkpoints')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint)

        for traj in range(len(eval_images)):

            quat_dists = []
            dpos_cos_dists = []


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

                err_cos = np.arccos(1. - scipy.spatial.distance.cosine(dpos, g_dpos))
                err_quat = np.arccos(2 * np.dot(g_drot, drot)**2 - 1)
                pos_direction_err +=  err_cos
                quat_error +=  err_quat

                quat_dists.append(err_quat)
                dpos_cos_dists.append(err_cos)

            mean_cos = np.mean(dpos_cos_dists)
            mean_rot = np.mean(quat_dists)

            std_cos = np.std(dpos_cos_dists)
            std_rot = np.std(quat_dists)
            print('Traj {} MEAN errors: Pos: {}\tQuat: {}'.format(traj, mean_cos, mean_rot))
            print('Teaj {} STD errors: Pos: {}\tQuat: {}'.format(traj, std_cos, std_rot))


            import matplotlib.pyplot as plt

            try:
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.hist(dpos_cos_dists, bins=100)
                ax.set_title('cos')

                """"
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.hist(quat_dists, bins=100)
                ax.set_title('quat')
                """

                plt.show()
            except:
                pass



