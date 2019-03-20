import numpy as np
import tensorflow as tf


class PPOPolicy:

    def __init__(self, obs_size, n_actions, dataset_dir, logsig=-1, clip_range=0.2, name='PPOPolicy', global_step=None):
        if type(obs_size) == int:
            self.obs_size = (obs_size,)
        else:
            self.obs_size = obs_size

        self.global_step = global_step
        self.dataset_dir = dataset_dir
        self.n_actions = n_actions
        self.initlogsig = logsig
        self.clip_range = clip_range
        self.name = name
        with tf.variable_scope(self.name):
            self._build()
            self._create_optimization()

    def _build(self):
        from models.BC import BC
        with tf.variable_scope('Model'):
            self.b = BC(self.dataset_dir, (84,84,3), (30,), (8,), 16, real=False, use_resnet=False)

            self.obs = self.b.proprio_obs
            self.image_obs = self.b.image_obs

        self.actions = tf.placeholder(tf.float32, (None, self.n_actions), name='actions')
        self.advantages = tf.placeholder(tf.float32, (None,), name='advantages')
        self.oldlogsig = tf.placeholder(tf.float32, (self.n_actions,), name='oldlogsig')
        self.oldmeans = tf.placeholder(tf.float32, (None, self.n_actions), name='oldmeans')
        self.returns = tf.placeholder(tf.float32, (None,), name='oldvalue')


        self.means = tf.concat([self.b.eval_delta_pos, self.b.eval_delta_quat, self.b.eval_gripper_output], axis=-1)

        self.value = tf.squeeze(tf.layers.dense(self.b.value_hid, 1, activation=None, name='fc_value'))

        with tf.variable_scope('PD'):
            self.logsig = tf.Variable(dtype=tf.float32, initial_value=tf.ones(self.n_actions,) + self.initlogsig, name='logsig')

            logp = -0.5 * tf.reduce_sum(self.logsig)
            logp += -0.5 * tf.reduce_sum(tf.square(self.actions - self.means) /
                                         tf.exp(self.logsig), axis=1)
            self.logp = logp

            logp_old = -0.5 * tf.reduce_sum(self.oldlogsig)
            logp_old += -0.5 * tf.reduce_sum(tf.square(self.actions - self.oldmeans) /
                                             tf.exp(self.oldlogsig), axis=1)
            self.logp_old = logp_old

            log_det_cov_old = tf.reduce_sum(self.oldlogsig)
            log_det_cov_new = tf.reduce_sum(self.logsig)
            tr_old_new = tf.reduce_sum(tf.exp(self.oldlogsig - self.logsig))

            self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                           tf.reduce_sum(tf.square(self.means - self.oldmeans) /
                                                         tf.exp(self.logsig), axis=1) -
                                           self.n_actions)
            self.entropy = 0.5 * (self.n_actions * (np.log(2 * np.pi) + 1) +
                                  tf.reduce_sum(self.logsig))

        with tf.variable_scope('Sample'):
            self.sample_op = (self.means +
                              tf.exp(self.logsig / 2.0) *
                              tf.random_normal(shape=(self.n_actions,)))


    def _create_optimization(self):
        ratio = tf.exp(self.logp - self.logp_old)
        clipped_pg_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
        surrogate_loss = tf.minimum(self.advantages * ratio,
                                    self.advantages * clipped_pg_ratio)
        value_loss = tf.reduce_mean(tf.square(self.value - self.returns))

        self.loss = -tf.reduce_mean(surrogate_loss) + value_loss

        step = tf.maximum(self.global_step - 1000, 0)
        self.lam = tf.train.exponential_decay(1., step, 1000, 0.9, staircase=True)
        self.loss *= ( 1. - self.lam)
        self.loss += self.lam * self.b.loss


    def sample(self, obs, image=None):
        sess = tf.get_default_session()
        assert sess is not None, 'No session'

        image = np.array([image])
        if len(image.shape) == 5:
            image = image[0]


        action, value = sess.run([self.sample_op, self.value], feed_dict={
            self.obs: np.array(obs),
            self.image_obs: image
        })
        return action.ravel(), value.ravel()

class PPO:

    def __init__(self, obs_size, n_actions, init_logsig=-1., clip_range=0.2, n_optimize=20):

        self.n_optimize = n_optimize

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.assign_add(self.global_step, 1)
        self.model = PPOPolicy(obs_size, n_actions, init_logsig, clip_range, global_step=self.global_step)

        with tf.variable_scope('Optimizer'):

            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.model.loss, global_step=self.global_step)

    def sample(self, obs, image=None):
        return self.model.sample(obs, (image - self.model.b.dset.image_mean)*self.model.b.dset.image_std)

    def optimize(self, batch):

        advantages = batch['advantages']
        actions = batch['actions']
        images = batch['images']
        states = batch['states']
        returns = batch['returns']


        for _ in range(self.n_optimize):
            inds = np.arange(advantages.shape[0])
            np.random.shuffle(inds)
            inds = inds.ravel()

            advantages = advantages[inds]
            actions = actions[inds]
            images = images[inds]
            states = states[inds]
            returns = returns[inds]

            feed_dict = {self.model.obs: states,
                         self.model.image_obs: (images - self.model.b.dset.image_mean)*self.model.b.dset.image_std,
                         self.model.actions: actions,
                         self.model.advantages: advantages.ravel(),
                         self.model.returns: returns.ravel(),
            }

            sess = tf.get_default_session()
            old_means_np, old_log_vars_np = sess.run([self.model.means, self.model.logsig], feed_dict)

            feed_dict[self.model.oldlogsig] = old_log_vars_np
            feed_dict[self.model.oldmeans] = old_means_np

            sess.run(self.train_op, feed_dict)
