import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from baselines.a2c.utils import fc
from tensorflow.python.ops import math_ops

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x, portion=1.0):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other, portion=1.0):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x, portion=1.0):
        # return - self.neglogp(x)
        negologpi, pi_mean, pi_std, pi_logstd, x_ret = self.neglogp(x, portion)
        return - negologpi, pi_mean, pi_std, pi_logstd, x_ret

    def get_shape(self):
        return self.flatparam().shape
    @property
    def shape(self):
        return self.get_shape()
    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)



class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        mean = _matching_fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.dc=0.0
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        # self.portion = tf.placeholder(dtype=tf.float32, shape=())
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x, portion=1.0):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1), self.mean, self.std, self.logstd, x

    def kl(self, other, portion=1.0):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self, portion=1.0):
        self.std=portion
        self.logstd=tf.log(self.std)

        return self.mean + self.std * (tf.random_normal(tf.shape(self.mean))/1.0), self.mean, self.std, portion, self.logstd

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError

def _matching_fc(tensor, name, size, init_scale, init_bias):
    if tensor.shape[-1] == size:
        return tensor
    else:
        return fc(tensor, name, size, init_scale=init_scale, init_bias=init_bias)
