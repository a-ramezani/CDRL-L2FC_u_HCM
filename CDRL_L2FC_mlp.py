from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from CDRL_L2FC_distributions import make_pdtype
from baselines.common import colorize
import numpy as np

import time
class MlpPolicy(object):
    recurrent = False
    portion=1.0
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            # wer

    def activate(self, layer, activation_function='tanh'):
        if(activation_function=='tanh'):
            layer = tf.nn.tanh(layer)
        elif(activation_function=='relu'):
            layer = tf.nn.relu(layer)
        elif(activation_function=='leaky_relu'):
            layer = tf.nn.leaky_relu(layer)

        return layer

    def norm(self, layer, layer_norm=False):

        if(layer_norm):
            return tf.contrib.layers.layer_norm(layer, center=True, scale=True)
        else:
            return layer

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, activation_function='tanh', gaussian_fixed_var=True, state_normalization=True, neural_net='normal', nunber_of_obstacles=1):
        layer_norm=False

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        pol_output=[]

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            if(state_normalization):
                obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            else:
                obz=ob

            last_out = obz
            start_time=time.time()

            wholestate=obz[:,:-int(1+(3*nunber_of_obstacles))]
            distance_with_obstacle_goal_obstacle_pose=obz[:,-int(1+(3*nunber_of_obstacles)):]

            last_out = self.norm(self.activate(tf.layers.dense(wholestate , 256, name="fc010", kernel_initializer=U.normc_initializer(1.0)),activation_function),layer_norm)

            last_out = tf.concat([last_out, distance_with_obstacle_goal_obstacle_pose], axis=1)

            last_out = self.norm(self.activate(tf.layers.dense(last_out , 256, name="fc011", kernel_initializer=U.normc_initializer(1.0)),activation_function),layer_norm)

            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz

            wholestate=obz[:,:-int(1+(3*nunber_of_obstacles))]
            distance_with_obstacle_goal_obstacle_pose=obz[:,-int(1+(3*nunber_of_obstacles)):]

            last_out = self.norm(self.activate(tf.layers.dense(wholestate , 256, name="fc010", kernel_initializer=U.normc_initializer(1.0)),activation_function),layer_norm)

            last_out = tf.concat([last_out, distance_with_obstacle_goal_obstacle_pose], axis=1)

            last_out = self.norm(self.activate(tf.layers.dense(last_out , 256, name="fc011", kernel_initializer=U.normc_initializer(1.0)),activation_function),layer_norm)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = (tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01)))

                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())

        portion = U.get_placeholder(name="portion", dtype=tf.float32, shape=[sequence_length,4])

        ac, mean_val, std_val, portion_val, logstd_val = self.pd.sample(portion)

        self._act = U.function([stochastic, ob, portion], [ac, self.vpred, mean_val, std_val, portion_val, logstd_val, pdparam, ob, self.ob_rms.mean, self.ob_rms.std, pol_output])

    def act(self, stochastic, ob, portion=1.0, save_mine=False, nnet_path=''):
        ac1, vpred1, mean_val, std_val, portion_val, logstd_val, pdparam, ob, ob_rms_mean, ob_rms_std, pol_output =  self._act(stochastic, ob[None], [portion,portion,portion,portion])

        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
