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

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def deconv2d(x, out_shape, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, prevNumFeat=None):
    with tf.variable_scope(name):
        num_filters = out_shape[-1]
        prevNumFeat = int(x.get_shape()[3]) if prevNumFeat is None else prevNumFeat
        stride_shape = [1, stride[0], stride[1], 1]
        # transpose_filter : [height, width, out_channels, in_channels]
        filter_shape = [filter_size[0], filter_size[1], num_filters, prevNumFeat]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:2]) * prevNumFeat
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = np.prod(filter_shape[:3])
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)


        deconv2d = tf.nn.conv2d_transpose(x, w, tf.stack(out_shape), stride_shape, pad)

        return deconv2d

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def smallNipsHead(x):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    '''
    x = tf.nn.relu(conv2d(x, 8, "l2", [2, 2], [2, 2]))
    x = flatten(x)

    return x

def inversesmallNipsHead(x, final_shape):
    ''' universe agent example
        input: [None, 288]; output: [None, 42, 42, 1];
    '''
    # print('Using small inverse-universe head design')
    bs = tf.shape(x)[0]

    deconv_shape1 = [final_shape[1]]
    deconv_shape2 = [final_shape[2]]

    deconv_shape1.append(int((deconv_shape1[-1]+2*0-2 + 2)/2 + 1))
    deconv_shape2.append(int((deconv_shape2[-1]+2*0-2 + 2)/2))

    inshapeprod = np.prod(x.get_shape().as_list()[1:]) / 8.0

    x = tf.reshape(x, [-1, deconv_shape1[-1], deconv_shape2[-1], 8])
    deconv_shape1 = deconv_shape1[:-1]
    deconv_shape2 = deconv_shape2[:-1]

    # i=0

    x = deconv2d(x, [bs] + final_shape[1:], "dl4", [2, 2], [2, 2], prevNumFeat=8)

    return x

class Predictor(object):
    def __init__(self, sess, ob_space, ob_space_2, ac_space, reward_space, designHead='universe', parameters='', ac_pdtype=''):
        input_shape = [None]  + [1] + list(ob_space)

        input_shape = [None]    + list(ob_space)
        input_shape_2 = [None]  + list(ob_space_2)

        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = phi2_org = tf.placeholder(tf.float32, input_shape_2)

        # action_input_shape = [None] + [1] + list(ac_space)
        action_input_shape = [None]  + list(ac_space)
        reward_input_shape = [None]  + list(reward_space)

        self.asample = asample = tf.placeholder(tf.float32, action_input_shape)
        self.rscsample = rscsample = tf.placeholder(tf.float32, reward_input_shape)
        self.rsample = rsample = tf.placeholder(tf.float32, [None] + [1])

        self.sess = sess
        self.ac_pdtype=ac_pdtype

        size = 256

        designHead = 'smallNips'

        if designHead == 'smallNips':
            phi1 = smallNipsHead(phi1)
            phi2 = flatten(phi2)
            phi2 = tf.nn.relu(linear(phi2, phi1.get_shape()[1].value, "phi12_nips", normalized_columns_initializer(0.01)))

        g = tf.concat([phi1, phi2], 1)#(1,[phi1, phi2])
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))

        action_dimx=ac_space
        self.ac_mean=np.zeros(action_dimx)
        self.ac_std=np.ones(action_dimx)

        self.prac_mean=np.zeros(action_dimx)
        self.prac_std=np.ones(action_dimx)

        self.averaged_prac_raw=np.zeros(action_dimx)
        self.averaged_ac_raw=np.zeros(action_dimx)

        normalize=True

        if(parameters.curiosity_trajectory_action=='waypoint' or parameters.curiosity_trajectory_action=='waypoint_posatt' or parameters.curiosity_trajectory_action=='waypoint_3section' ):
            predicted_action = linear(g, asample.get_shape()[1].value, "g2", normalized_columns_initializer(0.01))
            predicted_action=(predicted_action-self.prac_mean)/self.prac_std

            self.invloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(predicted_action, asample)), name='invloss') #/ predicted_actions_f.get_shape()[1].value #* self.rewards_change_loss

            self.ainvprobs = predicted_action#tf.nn.softmax(logits, dim=-1)
            f = tf.concat([phi1, asample], 1)#(1, [phi1, asample])

        else:
            asample_flat = flatten(asample)

            predicted_actions_flat = linear(g, asample_flat.get_shape()[1].value, "g2", normalized_columns_initializer(0.01))

            def extract_actions_feature(actions):
                actions_feature = linear(actions, 7, "action_f", normalized_columns_initializer(0.01))
                return actions_feature

            actions_f = extract_actions_feature(asample_flat)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                predicted_actions_f = extract_actions_feature(predicted_actions_flat)

            self.invloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(predicted_actions_f, actions_f)), name='invloss') #/ predicted_actions_f.get_shape()[1].value #* self.rewards_change_loss
            self.ainvprobs = predicted_actions_f#tf.nn.softmax(logits, dim=-1)
            f = tf.concat([phi1, actions_f], 1)#(1, [phi1, asample])

        self.rewards_change_loss=tf.reduce_mean(rscsample, name='rewardloss')
        self.reward_change_loss=rsample

        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))

        if designHead == 'smallNips':
            phi2_flatten_org_size = phi2.get_shape()[0].value
            f_full = tf.nn.relu(linear(f, phi2_org.get_shape()[2].value, "phi12_reverse", normalized_columns_initializer(0.01)))
            phi2_org_shape=phi2_org.get_shape()
            f_full = tf.reshape(f_full, [-1, input_shape_2[1], input_shape_2[2], input_shape_2[3]])

        self.regenerate_loss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f_full, phi2_org)), name='regenerate_loss')
        # self.regenerate_loss = 0.0
        self.regenerate_image=f_full
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss') #/ phi2.get_shape()[1].value #* self.rewards_change_loss
        self.forwardloss = (self.forwardloss + self.regenerate_loss) #* phi1.get_shape()[1].value
        self.state_phi = phi1
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class Curiosity_HCM_SAR(object):
    def __init__(self, sess, uid, hidden_layer_size, hidden_layer_number, env_state_space_dim=28, env_state_space_2_dim=12, env_action_space='', task=0, visualise=False, unsupType='', envWrap=False, designHead='universe', noReward=False, parameters='', ac_pdtype=''):
        self.parameters=parameters

        self.batch_states=[]
        self.batch_next_states=[]
        self.batch_actions=[]
        self.batch_rewards_change=[]

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

        self.a_curiosity_observation_space=((Box(0.0, 10.0, shape=(1,)+env_state_space_dim, dtype=np.float32)))
        self.a_curiosity_observation_space_2=((Box(0.0, 10.0, shape=(1,)+env_state_space_2_dim, dtype=np.float32)))

        if(self.parameters.curiosity_trajectory_action=='waypoint'):
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(3,), dtype=np.float32)))

        elif(self.parameters.curiosity_trajectory_action=='waypoint_posatt'):
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(7,), dtype=np.float32)))

        elif(self.parameters.curiosity_trajectory_action=='waypoint_3section'):
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(9,), dtype=np.float32)))

        else:
            self.a_action_space_low_level=((Box(-1.0, 1.0, shape=(1,self.parameters.curiosity_trajectory_length,4,), dtype=np.float32)))

        self.a_reward_space_low_level=((Box(-1.0, 1.0, shape=(1,self.parameters.curiosity_trajectory_length,1,), dtype=np.float32)))

        self.previous_state=self.a_curiosity_observation_space.sample()
        self.curiosity_rewards_raw=[]
        self.curiosity_rewards_episode_averaged=[]
        self.curiosity_state=np.zeros((100, 28),np.float32)
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

        self.curiosity_state=self.a_curiosity_observation_space.sample()#np.zeros((100,28),np.float32)#np.zeros((12),np.float32)#get_state(False,False)

        self.curiosity_next_state=self.a_curiosity_observation_space_2.sample()#np.zeros((100,28),np.float32)#np.zeros((12),np.float32)#get_state(False,False)

        self.ACTION_SIZE=self.a_curiosity_observation_space.sample()#4

        self.curiosity_action=self.a_action_space_low_level.sample()#np.zeros((self.ACTION_SIZE),np.float32)#[0.0,0.0,0.0]
        self.curiosity_action2=self.a_action_space_low_level.sample()#np.zeros((self.ACTION_SIZE),np.float32)#[0.0,0.0,0.0]

        self.curiosity_reward=self.a_reward_space_low_level.sample()#np.zeros((self.ACTION_SIZE),np.float32)#[0.0,0.0,0.0]

        self.curiosity_current_states = np.array([self.curiosity_state for _ in range(self.parameters.Curiosity_database_size * 1)])
        self.curiosity_next_states = np.array([self.curiosity_next_state for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.curiosity_rewards_changes = np.array([self.curiosity_reward for _ in range(self.parameters.Curiosity_database_size * 1)])

        self.batch_states = np.array([self.curiosity_state for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_next_states = np.array([self.curiosity_next_state for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_actions = np.array([self.curiosity_action for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_rewards = np.array([self.curiosity_reward for _ in range(self.parameters.curiosity_batch_size * 1)])
        self.batch_rewards_change = np.array([self.curiosity_reward for _ in range(self.parameters.curiosity_batch_size * 1)])

        self.curiosity_bonuses = np.zeros((self.parameters.Curiosity_database_size * 1), 'float32')
        self.curiosity_actions = np.array([self.curiosity_action for _ in range(self.parameters.Curiosity_database_size * 1)])
        self.curiosity_rewards2 = np.array([self.curiosity_reward for _ in range(self.parameters.Curiosity_database_size * 1)])

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
        numaction = self.a_action_space_low_level.shape[0]


        self.ap_network=[]
        self.global_step=[]
        self.train_op=[]
        self.sync=[]
        self.predloss=[]

        for CuH in range(self.parameters.number_of_curiosity_heads):
            with tf.variable_scope("global"):
                self.global_step.append(tf.get_variable(str(uid) + "global_step" + str(CuH), [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False))

                with tf.variable_scope(str(uid) + "predictor" + str(CuH)):
                    self.ap_network.append(Predictor(self.sess, self.a_curiosity_observation_space.shape, self.a_curiosity_observation_space_2.shape, self.a_action_space_low_level.shape, self.a_reward_space_low_level.shape, designHead, self.parameters, ac_pdtype))

            self.predictor = self.ap_network#[CuH]

            self.predloss.append((self.predictor[CuH].invloss * (1-self.parameters.curiosity_FORWARD_LOSS_WT) + self.predictor[CuH].forwardloss * self.parameters.curiosity_FORWARD_LOSS_WT))
            predgrads = tf.gradients(self.predloss[CuH], self.predictor[CuH].var_list)

            predgrads, _ = tf.clip_by_global_norm(predgrads, self.parameters.curiosity_GRAD_NORM_CLIP)
            pred_grads_and_vars = list(zip(predgrads, self.ap_network[CuH].var_list))
            grads_and_vars = pred_grads_and_vars


            opt = tf.train.AdamOptimizer(self.parameters.curiosity_LEARNING_RATE)
            self.train_op.append(opt.apply_gradients(grads_and_vars))


            sync_var_list = [v1.assign(v2) for v1, v2 in zip(self.predictor[CuH].var_list, self.ap_network[CuH].var_list)]

            self.sync.append(tf.group(*sync_var_list))


            self.summary_writer = None
            self.local_steps = 0

        print('FINISHED initiation')

    def pred_bonus(self, sess, s1, s2, asample, rscsample):
        error = None
        for CuH in range(self.parameters.number_of_curiosity_heads):
            if(CuH==0):
                error = sess.run([self.predictor[CuH].forwardloss, self.predictor[CuH].invloss, self.predictor[CuH].rewards_change_loss],
                    {self.predictor[CuH].s1: [s1], self.predictor[CuH].s2: [s2], self.predictor[CuH].asample: [asample], self.predictor[CuH].rscsample: [rscsample]})
            else:
                error += sess.run([self.predictor[CuH].forwardloss, self.predictor[CuH].invloss, self.predictor[CuH].rewards_change_loss],
                    {self.predictor[CuH].s1: [s1], self.predictor[CuH].s2: [s2], self.predictor[CuH].asample: [asample], self.predictor[CuH].rscsample: [rscsample]})

        return error[0]/ float(self.parameters.number_of_curiosity_heads), error[1]/ float(self.parameters.number_of_curiosity_heads), error[2]/ float(self.parameters.number_of_curiosity_heads)

    def pred_act(self, sess, s1, s2):
        result = None
        for CuH in range(self.parameters.number_of_curiosity_heads):
            if(CuH==0):
                result=self.sess.run(self.predictor[CuH].ainvprobs, {self.predictor[CuH].s1: [s1], self.predictor[CuH].s2: [s2]})
            else:
                result+=self.sess.run(self.predictor[CuH].ainvprobs, {self.predictor[CuH].s1: [s1], self.predictor[CuH].s2: [s2]})

        result = result / float(self.parameters.number_of_curiosity_heads)

        return result[0, :]

    def process(self, sess, states, next_states, actions, rewards_change, bonuses):
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        should_compute_summary = False

        states=np.round(states,decimals = self.parameters.curiosity_round_float_to)
        next_states=np.round(next_states,decimals = self.parameters.curiosity_round_float_to)

        episode_steps=states.shape[0]-1
        batch_size=self.parameters.curiosity_batch_size


        if(self.parameters.curiosity_trained_for_the_first_time):
            curiosity_epoch=self.parameters.curiosity_epoch
        else:
            curiosity_epoch=self.parameters.curiosity_starting_training_epoch
            self.parameters.curiosity_trained_for_the_first_time=True

        for CuH in range(self.parameters.number_of_curiosity_heads):
            fetches = [self.train_op[CuH], self.global_step[CuH]]
            for _ in range(curiosity_epoch):

                batch_counter=0

                random_numbers=np.random.randint(episode_steps, size=batch_size)
                for random_numbers_counter in range(batch_size):
                    final_random_number=random_numbers[random_numbers_counter]

                    self.batch_states[batch_counter]=states[final_random_number]
                    self.batch_next_states[batch_counter]=next_states[final_random_number]
                    self.batch_actions[batch_counter]=actions[final_random_number]
                    self.batch_rewards_change[batch_counter]=rewards_change[final_random_number]

                    batch_counter+=1

                self.feed_dict = {}

                self.feed_dict[self.ap_network[CuH].s1] = self.batch_states #batch.si[:-1]
                self.feed_dict[self.ap_network[CuH].s2] = self.batch_next_states #batch.si[1:]
                self.feed_dict[self.ap_network[CuH].asample] = self.batch_actions #batch.a
                self.feed_dict[self.ap_network[CuH].rscsample] = self.batch_rewards_change #batch.a

                fetched = sess.run(fetches, feed_dict=self.feed_dict)

            print(f'TRAINING SAR HEAD {CuH}')


        del episode_steps
        del batch_size
