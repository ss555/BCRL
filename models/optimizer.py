import tensorflow as tf
import numpy as np

class AdamBase(tf.train.Optimizer):
    """
        An implementation of ADAM optimization that allows for weight_decay and warm restarts as per
        https://arxiv.org/pdf/1711.05101.pdf. According to that paper, L2 regularization
        is not equivilent when using ADAM as the optimizer.

        Potential extensions of this class are to implement the momentum correction for
        distributed ADAM as per https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
        or some of the other things adressed in section 3 and section 2.
        Verify that Horovod does not already account for them first
    """
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, warm_restarts=False, \
                 epochs=None, batch_size=None, epoch_size=None, normalized_weight_decay=False, name='AdamOptimizer'):

        self.name = name
        with tf.variable_scope(name):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.weight_decay = weight_decay # if noramlized_weight_decay this is assumed to be w_norm from the paper

            self.warm_restarts = warm_restarts
            self.normalized_weight_decay = normalized_weight_decay

            if normalized_weight_decay:
                assert epoch_size is not None, 'you must pass an epoch_size to use normalized weight decay'
                assert batch_size is not None, 'you must pass the batch_size to use normalized_weight_decay'
                if not self.warm_restarts:
                    assert epochs is not None, 'you must pass a value for the number of epochs to use normalized weight decay'
                    self.t_i = epochs

                self.batch_size = batch_size
                self.epoch_size = epoch_size

            if self.warm_restarts:
                assert epochs is not None, 'you must pass a value of epochs for using warm restarts'

                with tf.variable_scope('WawrmRestartParameters'):
                    self.t_i = tf.Variable(0.01 * epochs, trainable=False)
                    self.t_mult = 2
                    self.t_cur = tf.Variable(0.0, trainable=False)

            self.m, self.v = {}, {}

            with tf.variable_scope('Slots'):
                for var in tf.trainable_variables():
                    self.m[var] = tf.Variable(tf.zeros_like(var), trainable=False)
                    self.v[var] = tf.Variable(tf.zeros_like(var), trainable=False)

    def warm_restart_schedule(self):
        """
        Warm restarts for ADAMWR
        """
        with tf.name_scope(self.name):
            with tf.name_scope('CosineAnnealing'):
                # cos annealing of the learning rate
                val = 0.5 + 0.5 * tf.cos(np.pi * self.t_cur / self.t_i)
                self.t_cur = tf.cond(tf.equal(self.t_cur+1, self.t_i), \
                                     lambda: tf.assign(self.t_cur, 0.0), \
                                     lambda: tf.assign_add(self.t_cur, 1.0))
                self.t_i   = tf.cond(tf.equal(self.t_cur, 0.0), \
                                     lambda: tf.assign(self.t_i, self.t_mult * self.t_i), \
                                     lambda: tf.assign_add(self.t_i, 0.0))
        return val
    def _normalized_weight_decay(self):
        """
        Normalized weight decay (w_norm in the paper) for ADAMW(R)
        """
        with tf.name_scope(self.name):
            with tf.name_scope('NormalizedWeightDecay'):
                ret = self.weight_decay * tf.sqrt(self.batch_size / (self.t_i * self.epoch_size))
        return ret
    
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        with tf.name_scope(self.name):
            with tf.name_scope('GlobalStep'):
                step = global_step.assign_add(1)
                step = tf.cast(step, tf.float32)


        with tf.variable_scope(self.name):
            with tf.variable_scope('Slots'):
                for var in tf.trainable_variables():
                    if var not in self.m:
                        self.m[var] = tf.Variable(tf.zeros_like(var), trainable=False)
                        self.v[var] = tf.Variable(tf.zeros_like(var), trainable=False)

        updates = []
        if self.warm_restarts:
            lr_scale = self.warm_restart_schedule()
        else:
            lr_scale = 1.0

        # This is used for the "more robust" version of weight decay suggested in the ADAMW paper
        if self.normalized_weight_decay:
            weight_decay = self._normalized_weight_decay()
        else:
            weight_decay = self.weight_decay

        with tf.name_scope(self.name):
            with tf.name_scope('AssignOps'):
                for grad, var in grads_and_vars:
                    if grad is None:
                        continue

                    m = self.m[var].assign( self.beta1 * self.m[var] + (1.0 - self.beta1) * grad)
                    v = self.v[var].assign( self.beta2 * self.v[var] + (1.0 - self.beta2) * tf.square(grad))

                    corrected_m = m / (1.0 - tf.pow(self.beta1, step))
                    corrected_v = v / (1.0 - tf.pow(self.beta2, step))

                    # this is the key line for ADAMW
                    update = - lr_scale * (self.lr * corrected_m / (tf.sqrt(corrected_v) + self.eps) + weight_decay * var)
                    updates.append(var.assign_add(update))
                grouped = tf.group(*updates)
        return grouped

class ADAMW(AdamBase):
    def __init__(self, lr, weight_decay, **kargs):
        super().__init__(lr, weight_decay=weight_decay, normalized_weight_decay=True, **kargs)
class ADAMWR(AdamBase):
    def __init__(self, lr, weight_decay,  **kargs):
        super().__init__(lr, weight_decay=weight_decay, normalized_weight_decay=True, warm_restart=True, **kargs)
