"""
Custom optimizers to implement lr_mult as in caffe

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

References
----------
https://github.com/keras-team/keras/issues/5920#issuecomment-328890905
"""

from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops


class customSGD(optimizers.SGD):
    """
    Custom subclass of the SGD optmizer to implement lr_mult as in Caffe
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, lr_mult=0.1, excluded_vars=[], **kwargs):
        super().__init__(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                    1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                          K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        for p, g, m in zip(params, grads, moments):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr = lr
            else:
                multiplied_lr = lr * self.lr_mult
            ###################################################

            v = self.momentum * m - multiplied_lr * g  # velocity
            self.updates.append(state_ops.assign(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'excluded_vars': self.excluded_vars,
            'lr_mult': self.lr_mult
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class customAdam(optimizers.Adam):
    """
    Custom subclass of the Adam optmizer to implement lr_mult as in Caffe
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False,
                 lr_mult=0.1, excluded_vars=[],**kwargs):
        super().__init__(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
              1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                    K.dtype(self.decay))))

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
            (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr_t = lr_t
            else:
                multiplied_lr_t = lr_t * self.lr_mult
            ###################################################

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - multiplied_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - multiplied_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'excluded_vars': self.excluded_vars,
            'lr_mult': self.lr_mult
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class customAdamW(optimizers.Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay (L2 penalty) (default: 0.025).
        batch_size: integer >= 1. Batch size used during training.
        samples_per_epoch: integer >= 1. Number of samples (training points) per epoch.
        epochs: integer >= 1. Total number of epochs for training.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)

    ################
    WARNING:
    The original implementation comes from https://github.com/GLambard/AdamW_Keras/blob/master/AdamW.py
    This customized version implements lr_mult as in Caffe

    Remember that one has to disable the standard L2 loss in Keras when using this optimizer.

    --> initial tests did not yield promising results. Maybe additional hyperparameter tuning is needed?
    ################
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.025,
                 batch_size=1, samples_per_epoch=1, epochs=1,
                 lr_mult=0.1, excluded_vars=[], **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.batch_size = K.variable(batch_size, name='batch_size')
            self.samples_per_epoch = K.variable(samples_per_epoch, name='samples_per_epoch')
            self.epochs = K.variable(epochs, name='epochs')
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        '''Bias corrections according to the Adam paper
        '''
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr_t = lr_t
            else:
                multiplied_lr_t = lr_t * self.lr_mult
            ###################################################

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            '''Schedule multiplier eta_t = 1 for simple AdamW
            According to the AdamW paper, eta_t can be fixed, decay, or 
            also be used for warm restarts (AdamWR to come). 
            '''
            eta_t = 1.
            p_t = p - eta_t * (multiplied_lr_t * m_t / (K.sqrt(v_t) + self.epsilon))
            if self.weight_decay != 0:
                '''Normalized weight decay according to the AdamW paper
                '''
                w_d = self.weight_decay * K.sqrt(self.batch_size / (self.samples_per_epoch * self.epochs))
                p_t = p_t - eta_t * (w_d * p)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'batch_size': int(K.get_value(self.batch_size)),
                  'samples_per_epoch': int(K.get_value(self.samples_per_epoch)),
                  'epochs': int(K.get_value(self.epochs)),
                  'epsilon': self.epsilon,
                  'excluded_vars': self.excluded_vars,
                  'lr_mult': self.lr_mult
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
