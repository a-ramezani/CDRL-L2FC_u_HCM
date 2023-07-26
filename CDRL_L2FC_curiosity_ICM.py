import numpy as np
import tensorflow as tf
import math
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

class Predictor(object):
    def __init__(self, sess, ob_space, ac_space, hidden_layer_size, hidden_layer_number, designHead='universe', ac_pdtype=''):
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = phi2_org = tf.placeholder(tf.float32, input_shape)

        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])
        self.sess = sess

        self.ac_pdtype=ac_pdtype

        size = hidden_layer_size

        if(hidden_layer_number==1):
            if(isinstance(ob_space, int)==False):
                phi1 = flatten(phi1)
                phi2 = flatten(phi2)

            phi1 = tf.nn.relu(linear(phi1, size, "phi11", normalized_columns_initializer(0.01)))
            phi2 = tf.nn.relu(linear(phi2, size, "phi21", normalized_columns_initializer(0.01)))

        elif(hidden_layer_number==2):
            phi1 = tf.nn.relu(linear(phi1, size, "phi11", normalized_columns_initializer(0.01)))
            phi1 = tf.nn.relu(linear(phi1, size, "phi12", normalized_columns_initializer(0.01)))

            phi2 = tf.nn.relu(linear(phi2, size, "phi21", normalized_columns_initializer(0.01)))
            phi2 = tf.nn.relu(linear(phi2, size, "phi22", normalized_columns_initializer(0.01)))

        g = tf.concat([phi1, phi2], 1)
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))

        import time

        action_dimx=ac_space
        self.ac_mean=np.zeros(action_dimx)
        self.ac_std=np.ones(action_dimx)

        self.prac_mean=np.zeros(action_dimx)
        self.prac_std=np.ones(action_dimx)

        self.averaged_prac_raw=np.zeros(action_dimx)
        self.averaged_ac_raw=np.zeros(action_dimx)

        normalize=True

        if(normalize==False):
            predicted_actions = linear(g, ac_space, "g2", normalized_columns_initializer(0.01))
            self.invloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(predicted_actions, asample)), name='invloss')
        else:
            predicted_actions = linear(g, ac_space, "g2", normalized_columns_initializer(0.01))
            predicted_actions=(predicted_actions-self.prac_mean)/self.prac_std
            self.invloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(predicted_actions, asample)), name='invloss')

        self.ainvprobs = predicted_actions

        f = tf.concat([phi1, asample], 1)
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))

        if(len(ob_space)>1):
            f = tf.nn.relu(linear(f, size, "inv_phif", normalized_columns_initializer(0.01)))
            f_full=tf.reshape(f, [-1,ob_space[0],ob_space[1]])

        else:
            f_full = tf.nn.relu(linear(f, phi2_org.get_shape()[1].value, "inv_phif", normalized_columns_initializer(0.01)))

        self.regenerate_loss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f_full, phi2_org)), name='regenerate_loss')
        self.regenerate_image=f_full

        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        self.forwardloss = (self.forwardloss + self.regenerate_loss)

        self.state_phi = phi1
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        result=self.sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})

        return result[0, :]

    def update_mean_std(self, actions, predicted_actions):
        accumulated_ac_raw=actions[0]
        for i in range(len(actions)-1):
            accumulated_ac_raw+=actions[i+1]

        self.averaged_ac_raw=((self.averaged_ac_raw + (accumulated_ac_raw / len(actions)))/2.0)
        total_std_dev=0.0
        counter=0

        for counter in range(len(actions)):
            std_dev=actions[counter] - self.averaged_ac_raw
            std_dev=std_dev*std_dev
            total_std_dev+=std_dev

        total_std_dev=total_std_dev / float(len(actions))
        total_std_dev=np.sqrt(total_std_dev)

        self.ac_mean=self.averaged_ac_raw
        self.ac_std=total_std_dev

        accumulated_prac_raw=predicted_actions[0]
        for i in range(len(predicted_actions)-1):
            accumulated_prac_raw+=predicted_actions[i+1]

        self.averaged_prac_raw=((self.averaged_prac_raw + (accumulated_prac_raw / len(predicted_actions)))/2.0)
        total_std_dev=0.0
        counter=0

        for counter in range(len(predicted_actions)):
            std_dev=predicted_actions[counter] - self.averaged_prac_raw
            std_dev=std_dev*std_dev
            total_std_dev+=std_dev

        total_std_dev=total_std_dev / float(len(predicted_actions))
        total_std_dev=np.sqrt(total_std_dev)

        self.prac_mean=self.averaged_prac_raw
        self.prac_std=total_std_dev

    def pred_bonus(self, s1, s2, asample):
        error = self.sess.run([self.forwardloss, self.invloss],
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})

        return error[0], error[1]

    def regenerate_state(self, s1, asample):
        return self.sess.run(self.regenerate_image, {self.s1: [s1],
                                            self.asample: [asample]})#[0, :]

class Curiosity(object):
    def __init__(self, sess, uid, hidden_layer_size, hidden_layer_number, env_state_space_dim=28, env_action_space='', task=0, visualise=False, unsupType='action', envWrap=False, designHead='universe', noReward=False, parameters='', ac_pdtype=''):
        self.parameters=parameters

        self.batch_states=[]
        self.batch_next_states=[]
        self.batch_actions=[]

        self.feed_dict = {}

        self.curiosity_reward_forward_loss_all=0.0
        self.curiosity_reward_inv_loss=0.0

        self.curiosity_reward_forward_loss_all_stat=0.0
        self.curiosity_reward_inv_loss_stat=0.0

        self.accumulated_curiosity_reward_raw_stat=0.0
        self.accumulated_curiosity_reward_stat=0.0

        self.mean_ig=0
        self.stddev_ig=1.0
        self.total_std_dev=0.0

        if(isinstance(env_state_space_dim, int)):
            self.a_curiosity_observation_space=((Box(0.0, 10.0, shape=(env_state_space_dim, ), dtype=np.float32)))

        else:
            self.a_curiosity_observation_space=((Box(0.0, 10.0, shape=(env_state_space_dim[0],env_state_space_dim[1], ), dtype=np.float32)))

        if(self.parameters.curiosity_trajectory_action=='waypoint'):
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(3,), dtype=np.float32)))

        elif(self.parameters.curiosity_trajectory_action=='waypoint_posatt'):
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(7,), dtype=np.float32)))

        else:
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))

        import time
        self.previous_state=self.a_curiosity_observation_space.sample()
        # print('1062 self.previous_state: {}'.format(self.previous_state.shape))

        self.curiosity_rewards_raw=[]

        self.curiosity_rewards_episode_averaged=[]

        self.curiosity_state=np.zeros((env_state_space_dim),np.float32)

        self.our_curiosity_reward_count_list=[]
        self.min_curiosity_reward=100000.0
        self.max_curiosity_reward=0.0

        self.min_our_curiosity_reward=100000.0
        self.max_our_curiosity_reward=0.0

        horizon=int(8 * 1024)

        self.rews_minus_curiosity = np.zeros(horizon, 'float32')

        self.accumulated_curiosity_reward_raw=0.0
        self.accumulated_curiosity_reward=0.0

        self.Curiosity_just_trained=False
        self.Curiosity_ready=False

        self.curiosity_episode=0

        self.curiosity_state=np.zeros((env_state_space_dim),np.float32)

        self.curiosity_next_state=np.zeros((env_state_space_dim),np.float32)
        numaction = self.a_action_space_low_level.shape[0]

        self.ACTION_SIZE=numaction

        self.curiosity_action=np.zeros((self.ACTION_SIZE),np.float32)#[0.0,0.0,0.0]
        self.curiosity_action2=np.zeros((self.ACTION_SIZE),np.float32)#[0.0,0.0,0.0]

        self.curiosity_current_states = np.array([self.curiosity_state for _ in range(self.parameters.Curiosity_database_size * 1)])
        self.curiosity_next_states = np.array([self.curiosity_state for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.batch_states = np.array([self.curiosity_state for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_next_states = np.array([self.curiosity_state for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_actions = np.array([self.curiosity_action for _ in range(self.parameters.curiosity_batch_size * 1)])

        self.curiosity_trajectory_current_states = np.array([self.curiosity_state for _ in range(self.parameters.Curiosity_database_size * 1)])
        self.curiosity_trajectory_next_states = np.array([self.curiosity_state for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.rewards_without_curiosity = np.zeros((self.parameters.Curiosity_database_size * 1), 'float32')

        self.curiosity_bonuses = np.zeros((self.parameters.Curiosity_database_size * 1), 'float32')
        self.curiosity_actions = np.array([self.curiosity_action for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.curiosity_rewards=np.zeros((self.parameters.Curiosity_database_size * 1), 'float32')
        self.curiosity_episode_actions=np.array([self.curiosity_action for _ in range(self.parameters.Curiosity_database_size * 1)])
        self.curiosity_episode_predicted_actions=np.array([self.curiosity_action for _ in range(self.parameters.Curiosity_database_size * 1)])


        self.curiosity_actions2 = np.array([self.curiosity_action2 for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.episode_counter=0

        self.episode_accumulated_main_curiosity_reward=0
        self.episode_accumulated_forward_curiosity_reward=0
        self.episode_accumulated_inverse_curiosity_reward=0

        self.episode_accumulated_main_curiosity_reward_raw=0
        self.episode_accumulated_forward_curiosity_reward_raw=0
        self.episode_accumulated_inverse_curiosity_reward_raw=0

        designHead=self.parameters.curiosity_head

        self.task = task

        self.envWrap = envWrap
        self.sess = sess

        self.predictor = None

        with tf.variable_scope("global"):
            self.global_step = tf.get_variable(str(uid) + "global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                               trainable=False)

            with tf.variable_scope(str(uid) + "predictor"):
                self.ap_network = self.predictor = Predictor(self.sess, self.a_curiosity_observation_space.shape, numaction, hidden_layer_size, hidden_layer_number, designHead, ac_pdtype)

        self.predloss = (self.predictor.invloss * (1-self.parameters.curiosity_FORWARD_LOSS_WT) + self.predictor.forwardloss * self.parameters.curiosity_FORWARD_LOSS_WT)

        predgrads = tf.gradients(self.predloss, self.predictor.var_list)

        predgrads, _ = tf.clip_by_global_norm(predgrads, self.parameters.curiosity_GRAD_NORM_CLIP)
        pred_grads_and_vars = list(zip(predgrads, self.ap_network.var_list))

        grads_and_vars = pred_grads_and_vars

        opt = tf.train.AdamOptimizer(self.parameters.curiosity_LEARNING_RATE)

        self.train_op = opt.apply_gradients(grads_and_vars)

        sync_var_list = [v1.assign(v2) for v1, v2 in zip(self.predictor.var_list, self.ap_network.var_list)]

        self.sync = tf.group(*sync_var_list)

        self.summary_writer = None
        self.local_steps = 0

    def process(self, sess, states, next_states, actions, bonuses):
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        should_compute_summary = False

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        states=np.round(states,decimals = self.parameters.curiosity_round_float_to)
        next_states=np.round(next_states,decimals = self.parameters.curiosity_round_float_to)

        episode_steps=states.shape[0]-1
        batch_size=self.parameters.curiosity_batch_size

        if(self.parameters.curiosity_trained_for_the_first_time):
            curiosity_epoch=self.parameters.curiosity_epoch
        else:
            curiosity_epoch=self.parameters.curiosity_starting_training_epoch
            self.parameters.curiosity_trained_for_the_first_time=True

        for _ in range(curiosity_epoch):
            random_numbers=np.random.randint(episode_steps, size=batch_size)
            batch_counter=0
            for random_numbers_counter in range(batch_size):
                final_random_number=random_numbers[random_numbers_counter]

                self.batch_states[batch_counter]=states[final_random_number]
                self.batch_next_states[batch_counter]=next_states[final_random_number]
                self.batch_actions[batch_counter]=actions[final_random_number]

                batch_counter+=1

            self.feed_dict = {}

            self.feed_dict[self.ap_network.s1] = self.batch_states
            self.feed_dict[self.ap_network.s2] = self.batch_next_states
            self.feed_dict[self.ap_network.asample] = self.batch_actions

            fetched = sess.run(fetches, feed_dict=self.feed_dict)

        del episode_steps
        del batch_size
