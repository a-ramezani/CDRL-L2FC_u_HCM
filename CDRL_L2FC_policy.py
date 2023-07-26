

import time
import tensorflow as tf
import numpy as np



import math
import random

from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from collections import deque
import pickle
import baselines.common.tf_util as U
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict
import gym
from baselines.common import colorize
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
import os.path as osp
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines.common.mpi_moments import mpi_moments

class CDRL_policy():

    def __init__(self, a_action_space, a_observation_space, parameters, policy_number='',policy_name='hovakh', number_of_output_actions=0):
        self.parameters=parameters

        if(self.parameters.deterministic_initiation):
            set_global_seeds(1)

        self.policy_number=policy_number
        self.previous_macro_point=np.zeros((1,2))
        self.curiosity_reward_forward_loss_all=0.0
        self.curiosity_reward_inv_loss=0.0
        self.a_action_space = a_action_space
        self.gamma=self.parameters.gamma
        self.lam=self.parameters.lam
        adam_epsilon=1e-5
        self.a_observation_space=a_observation_space
        ob_space = self.a_observation_space
        ac_space = self.a_action_space

        seed=1

        if(self.parameters.use_multi_vpred):
            from CDRL_L2FC_mlp_multiple_value_head import MlpPolicy

        else:
            from CDRL_L2FC_mlp import MlpPolicy

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=self.parameters.neural_network_hwomany_neural_each_layer, num_hid_layers=self.parameters.neural_network_hwomany_layers,
                activation_function=self.parameters.activation_function, gaussian_fixed_var=self.parameters.gaussian_fixed_var, state_normalization=self.parameters.state_normalization, neural_net=self.parameters.neural_net,nunber_of_obstacles=self.parameters.number_of_obstacles)

        self.pi = policy_fn("pi" , ob_space, ac_space) # Construct network for new policy

        self.oldpi = policy_fn("oldpi" , ob_space, ac_space) # Network for old policy
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        self.ret_c0 = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        self.ret_c1 = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        self.lrmult = tf.placeholder(name='lrmult' , dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        self.parameters.clip_param = self.parameters.clip_param * self.lrmult # Annealed cliping parameter epislon
        self.ob = U.get_placeholder_cached(name="ob" )
        self.portion = U.get_placeholder_cached(name="portion" )
        self.ac = self.pi.pdtype.sample_placeholder([None])
        self.pilogpi, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret=self.pi.pd.logp(self.ac, self.portion)
        self.oldpilogpi, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.oldpi_x_ret=self.oldpi.pd.logp(self.ac, self.portion)
        self.kloldnew = self.oldpi.pd.kl(self.pi.pd, self.portion)
        self.ent = self.pi.pd.entropy()
        self.meankl = tf.reduce_mean(self.kloldnew)
        self.meanent = tf.reduce_mean(self.ent)
        self.pol_entpen = (-self.parameters.entcoeff) * self.meanent
        self.explogpioldpi=tf.exp(self.pilogpi - self.oldpilogpi)
        self.ratio =  self.explogpioldpi
        self.ratio_clipped=self.ratio

        self.surr1 = self.ratio_clipped * self.atarg # surrogate from conservative policy iteration
        self.surr2 = tf.clip_by_value(self.ratio_clipped, 1.0 - self.parameters.clip_param, 1.0 + self.parameters.clip_param) * self.atarg #
        self.pol_surr = - tf.reduce_mean(tf.minimum(self.surr1, self.surr2)) # PPO's pessimistic surrogate (L^CLIP)


        if(self.parameters.use_multi_vpred):
            self.vf_loss = tf.reduce_mean(tf.square(self.pi.vpred - self.ret))
            self.vf_c0_loss = tf.reduce_mean(tf.square(self.pi.vpred_c0 - self.ret_c0))
            self.total_loss = self.pol_surr + self.pol_entpen + self.vf_loss * self.parameters.vf_loss_coef #+ self.vf_c0_loss * self.parameters.vf_c0_loss_coef
            self.total_loss_c = self.vf_c0_loss * self.parameters.vf_c0_loss_coef
            self.losses = [self.pol_surr, self.pol_entpen, self.vf_loss, self.meankl, self.meanent]
            self.losses_c = [self.vf_c0_loss]

            loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        else:
            self.vf_loss = tf.reduce_mean(tf.square(self.pi.vpred - self.ret))
            self.total_loss = self.pol_surr + self.pol_entpen + self.vf_loss
            self.losses = [self.pol_surr, self.pol_entpen, self.vf_loss, self.meankl, self.meanent]
            loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        if(self.parameters.use_multi_vpred):
            self.var_list_pol_vf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/pol') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/vf/')
            self.var_list_vf_c0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/vf_c0')

        else:
            # self.var_list = self.pi.pol.get_trainable_variables()
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/pol') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/vf/')

        if(self.parameters.use_multi_vpred):
            self.lossandgrad = U.function([self.ob, self.ac, self.atarg, self.ret, self.portion, self.lrmult],
            [self.losses + [U.flatgrad(self.total_loss, self.var_list_pol_vf,clip_norm=self.parameters.max_grad_norm)],
            self.ratio_clipped, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew, self.surr1, self.surr2, self.pol_surr, self.atarg])

            self.lossandgrad_c = U.function([self.ob, self.ac, self.atarg, self.ret_c0, self.portion, self.lrmult],
            [self.losses_c + [U.flatgrad(self.total_loss_c, self.var_list_vf_c0,clip_norm=self.parameters.max_grad_norm)],
            self.ratio_clipped, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew, self.surr1, self.surr2, self.pol_surr, self.atarg])

        else:
            self.lossandgrad = U.function([self.ob, self.ac, self.atarg, self.ret, self.portion, self.lrmult],
            [self.losses + [U.flatgrad(self.total_loss, self.var_list,clip_norm=self.parameters.max_grad_norm)],
            self.ratio_clipped, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew, self.surr1, self.surr2, self.pol_surr, self.atarg])

        if(self.parameters.use_multi_vpred):
            self.adam = MpiAdam(self.var_list_pol_vf, epsilon=adam_epsilon)
            self.adam_c = MpiAdam(self.var_list_vf_c0, epsilon=adam_epsilon)

        else:
            self.adam = MpiAdam(self.var_list, epsilon=adam_epsilon)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])

        if(self.parameters.use_multi_vpred):
            self.compute_losses = U.function([self.ob, self.ac, self.atarg, self.ret, self.portion, self.lrmult], self.losses)
            self.compute_losses_c = U.function([self.ob, self.ac, self.ret_c0, self.portion, self.lrmult], self.losses_c)

        else:
            self.compute_losses = U.function([self.ob, self.ac, self.atarg, self.ret, self.portion, self.lrmult], self.losses)

        self.is_stochastic=True
        # if(self.parameters.is_testing_unit):
        #     self.is_stochastic=False

        self.t = 0
        self.ac_sample = self.a_action_space.sample()

        self.new = True
        self.rew = 0.0

        if(self.parameters.deterministic_initiation):
            np.random.seed(1)
            random.seed(1)
            tf.set_random_seed(1)
            tf.compat.v1.set_random_seed(1)

        self.ob_sample= self.a_observation_space.sample()
        horizon=self.parameters.timesteps_per_batch #_low_level
        self.obs = np.array([self.ob_sample for _ in range(horizon)])

        self.rews = np.zeros(horizon, 'float32')
        self.vpreds = np.zeros(horizon, 'float32')
        self.news = np.zeros(horizon, 'int32')
        self.acs = np.array([self.ac_sample for _ in range(horizon)])

        s = np.random.randn(10)
        self.portions = np.array([self.ac_sample for _ in range(horizon)])

        self.tbtss = np.zeros(horizon, 'int32')

        self.prevacs = self.acs.copy()

        self.new=0
        self.portion_val=1.0


    def add_vtarg_and_adv(self, seg, gamma, lam):
        # last element is only used for last vtarg, but we already zeroed it if last new = 1

        new = np.append(seg["new"], 0)                      # just add one 0 to the end of seg["new"] array
        vpred = np.append(seg["vpred"], seg["nextvpred"])   # just add one nextvpred to the end of vpred array
        T = len(seg["rew"])                                 # T = Horizon
        seg["adv"] = gaelam = np.empty(T, 'float32')        # create seg["adv"] and equal it to an array with the size of Horizon
        rew = seg["rew"]

        positions=seg["positions"]
        angles=seg["angles"]
        tbts=seg["tbts"]

        lastgaelam = 0

        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]

            if(tbts[t]):
                delta = rew[t]

            else:
                delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]

            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            # print(colorize('t: {}, delta: {:.2f}, gaelam: {:.2f}'.format(t, delta, gaelam[t]), color='green'))

        seg["tdlamret"] = seg["adv"] + seg["vpred"]         # tdlamret = seg["adv"] = gaelam[t]

    def add_vtarg_and_adv_multi_vpred(self, seg, gamma, lam):
        new = np.append(seg["new"], 0)                      # just add one 0 to the end of seg["new"] array

        vpred = np.append(seg["vpred"], seg["nextvpred"])   # just add one nextvpred to the end of vpred array
        vpred_c0 = np.append(seg["vpred_c0"], seg["nextvpred_c0"])   # just add one nextvpred to the end of vpred array
        vpred_c1 = np.append(seg["vpred_c1"], seg["nextvpred_c1"])   # just add one nextvpred to the end of vpred array

        T = len(seg["rew"])                                 # T = Horizon

        seg["adv"] = gaelam = np.empty(T, 'float32')        # create seg["adv"] and equal it to an array with the size of Horizon
        seg["adv_c0"] = gaelam_c0 = np.empty(T, 'float32')        # create seg["adv"] and equal it to an array with the size of Horizon
        seg["adv_c1"] = gaelam_c1 = np.empty(T, 'float32')        # create seg["adv"] and equal it to an array with the size of Horizon
        seg["adv_total"] = gaelam_total = np.empty(T, 'float32')        # create seg["adv"] and equal it to an array with the size of Horizon

        rew = seg["rew"]
        rew_c0 = seg["rew_c0"]
        rew_c1 = seg["rew_c1"]

        positions=seg["positions"]
        angles=seg["angles"]
        tbts=seg["tbts"]

        lastgaelam = 0
        lastgaelam_c0 = 0
        lastgaelam_c1 = 0
        lastgaelam_total = 0

        for t in reversed(range(T)):

            nonterminal = 1-new[t+1]

            if(tbts[t]):
                    delta = rew[t]
                    delta_c0 = rew_c0[t]
                    delta_c1 = rew_c1[t]
                    delta_total = (rew[t] + rew_c0[t])

            else:
                delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
                delta_c0 = rew_c0[t] + gamma * vpred_c0[t+1] * nonterminal - vpred_c0[t]
                delta_c1 = rew_c1[t] + gamma * vpred_c1[t+1] * nonterminal - vpred_c1[t]

                delta_total = (rew[t] + rew_c0[t]) + gamma * (vpred[t+1] + vpred_c0[t+1]) * nonterminal - (vpred[t] + vpred_c0[t])

            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            gaelam_c0[t] = lastgaelam_c0 = delta_c0 + gamma * lam * nonterminal * lastgaelam_c0
            gaelam_c1[t] = lastgaelam_c1 = delta_c1 + gamma * lam * nonterminal * lastgaelam_c1

            gaelam_total[t] = lastgaelam_total =  delta_total + gamma * lam * nonterminal * lastgaelam_total #gaelam[t] + gaelam_c0[t]

            # print(colorize('t: {}, delta_total: {:.2f}, gaelam_total: {:.2f}'.format(t, delta_total, gaelam_total[t]), color='green'))

        seg["tdlamret"] = seg["adv"] + seg["vpred"]         # tdlamret = seg["adv"] = gaelam[t]
        seg["tdlamret_c0"] = seg["adv_c0"] + seg["vpred_c0"]         # tdlamret = seg["adv"] = gaelam[t]
        seg["tdlamret_c1"] = seg["adv_c1"] + seg["vpred_c1"]         # tdlamret = seg["adv"] = gaelam[t]

    #### Amir
    def train(self, seg, timesteps_so_far, current_policy_layer=0):
        print('Training...')

        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        max_timesteps=1e8
        timesteps_so_far=0

        if self.parameters.schedule == 'constant':
            self.cur_lrmult = 1.0
        elif self.parameters.schedule == 'linear':
            self.cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        if(self.parameters.use_multi_vpred):
            self.add_vtarg_and_adv_multi_vpred(seg, self.gamma, self.lam)
            self.ob, self.ac, self.atarg, tdlamret, tdlamret_c0, tdlamret_c1, self.portion = seg["ob"], seg["ac"], seg["adv_total"], seg["tdlamret"], seg["tdlamret_c0"], seg["tdlamret_c1"], seg["portions"]

        else:
            self.add_vtarg_and_adv(seg, self.gamma, self.lam)
            self.ob, self.ac, self.atarg, tdlamret, self.portion = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["portions"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        self.atarg = (self.atarg - self.atarg.mean()) / self.atarg.std() # standardized advantage function estimate

        if(self.parameters.use_multi_vpred):
            d = Dataset(dict(ob=self.ob, ac=self.ac, atarg=self.atarg, vtarg=tdlamret, vtarg_c0=tdlamret_c0, vtarg_c1=tdlamret_c1, portion=self.portion), shuffle=not self.pi.recurrent)

        else:
            d = Dataset(dict(ob=self.ob, ac=self.ac, atarg=self.atarg, vtarg=tdlamret, portion=self.portion), shuffle=not self.pi.recurrent)

        self.parameters.optim_batch = self.parameters.optim_batch or self.ob.shape[0]

        if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(self.ob) # update running mean/std for policy

        self.assign_old_eq_new() # set old parameter values to new parameter values

        start=0

        # if(self.parameters.use_new_training_method==False):
        for oec in range(self.parameters.optim_epoch):
            self.losses = [] # list of tuples, each of which gives the loss for a minibatch
            self.losses_c = [] # list of tuples, each of which gives the loss for a minibatch

            batch_counter=0
            for batch in d.iterate_once(self.parameters.optim_batch):
                if(self.parameters.use_multi_vpred):
                    [*newlosses, g], self.ratio, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew_val, self.surr1, self.surr2, self.pol_surr, self.atarg = self.lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], batch["portion"], self.cur_lrmult)
                    [*newlosses_c, g_c], self.ratio, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew_val, self.surr1, self.surr2, self.pol_surr, self.atarg = self.lossandgrad_c(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg_c0"], batch["portion"], self.cur_lrmult)

                else:
                    [*newlosses, g], self.ratio, self.oldpilogpi, self.pilogpi, self.explogpioldpi, self.ac, self.portion, self.oldpi_mean, self.oldpi_std, self.oldpi_logstd, self.pi_mean, self.pi_std, self.pi_logstd, self.pi_x_ret, self.kloldnew_val, self.surr1, self.surr2, self.pol_surr, self.atarg = self.lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], batch["portion"], self.cur_lrmult)


                batch_counter+=1

                self.adam.update(g, self.parameters.vf_stpsize * self.cur_lrmult)

                if(self.parameters.use_multi_vpred):
                    self.adam_c.update(g_c, self.parameters.vf_stpsize_c * self.cur_lrmult)

                self.losses.append(newlosses)

                if(self.parameters.use_multi_vpred):
                    self.losses_c.append(newlosses_c)


        self.losses = []
        self.losses_c = []

        for batch in d.iterate_once(self.parameters.optim_batch):
            if(self.parameters.use_multi_vpred):
                newlosses = self.compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], batch["portion"], self.cur_lrmult)
                newlosses_c = self.compute_losses_c(batch["ob"], batch["ac"], batch["vtarg_c0"], batch["portion"], self.cur_lrmult)

            else:
                newlosses = self.compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], batch["portion"], self.cur_lrmult)


            self.losses.append(newlosses)
            if(self.parameters.use_multi_vpred):
                self.losses_c.append(newlosses_c)

        meanlosses,_,_ = mpi_moments(self.losses, axis=0)
        if(self.parameters.use_multi_vpred):
            meanlosses_c,_,_ = mpi_moments(self.losses_c, axis=0)

        print('Training Finished.')

        save_every_n_episode=31

        if(self.parameters.episode_counter % save_every_n_episode == 0):
            start_time=time.time()
            U.save_state(osp.expanduser(os.getcwd() + '/model/' + self.parameters.title_precise + '.save'), sess=U.get_session())
            print(colorize("consumed saving time: %.4f seconds"%(time.time() - start_time), color='red'))

        self.parameters.episode_counter+=1
        self.parameters.episode_counter_2+=1

        return True
