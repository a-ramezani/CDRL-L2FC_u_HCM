import math
import sys
import time
import numpy as np
import random
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from baselines.common import colorize
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict
import gym

import numpy.random as nr

import rospy

import transforms3d as tf3d

from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from CDRL_L2FC_parameters import CL2FC_parameters

from L2FC_sensors import amir_rotors_sensors
from mav_msgs.msg import Actuators

from L2FC_rewards import amir_rotors_bioinspired_rewards
from L2FC_functions import rotors_stop, rotors_move, rotors_stop_engine

import cv2

from numba import jit

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import tensorflow as tf

rospy.init_node('L2FC', anonymous=True)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

#### functions to calculate the distance between two points

def d(p0, p1):
    return math.sqrt(((p1[0]-p0[0]) * (p1[0]-p0[0])) +
            ((p1[1]-p0[1]) * (p1[1]-p0[1])) +
            ((p1[2]-p0[2]) * (p1[2]-p0[2])))

def point_to_line(A1,A2,C):

    V=np.array((3),np.float32)
    W=np.array((3),np.float32)

    V=A2 - A1
    W=C - A1

    WV=np.dot(W,V)
    VV=np.dot(V,V)

    b = WV/VV

    Ab = A1 + b * V
    #return d(C, Ab)

    return d(Ab, C)

class L2FC_env:

    silent_mode=False

    header=[]

    x_angle=[]
    y_angle=[]
    z_angle=[]

    x_position=[]
    y_position=[]
    z_position=[]

    linear_velocity_x=[]
    linear_velocity_y=[]
    linear_velocity_z=[]

    angular_velocity_x=[]
    angular_velocity_y=[]
    angular_velocity_z=[]

    x_new_position_difference=[]
    y_new_position_difference=[]
    z_new_position_difference=[]

    yaw_new_angle_difference=[]

    desired_x_pose=[]
    desired_y_pose=[]
    desired_z_pose=[]
    desired_yaw_degree=[]

    robot_model_state=[]
    robot_state = []

    obstacle_model_state=[]
    obstacle_state = []

    rotors_sensors=[]
    x_position_raw=[]

    n_env=1
    active_env=0

    action_space = None

    observation_space=None

    accumulated_x_error=0.0
    accumulated_y_error=0.0
    accumulated_z_error=0.0

    accumulated_distance_error=0.0

    reach_time=0.0

    accumulated_xa_error=0.0
    accumulated_ya_error=0.0
    accumulated_za_error=0.0

    total_accumulated_reward=0.0
    episode_accumulated_reward=0.0
    average_reward=0.0

    episode_accumulated_reward_org=0.0

    episode_accumulated_move_in_yaw_direction_reward=0.0

    total_ang_vel_reward=0.0
    total_linvel_reward=0.0
    episode_accumulated_linvel_mag_reward=0.0
    episode_accumulated_ang_vel_mag_reward=0.0

    UN_COUNTER=0
    episode_counter=0
    real_move=False

    reset_timer=[]
    reset_flag=[]

    prev_callback_header_seq=[]

    start_resetting_counter=0#1500000#0#1000000
    annealing_ground_fly=1000000

    velocity_publisher=[]
    give_accurates=[]

    get_model_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    set_model_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    # map axes strings to/from tuples of inner axis, parity, repetition, frame
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

    previous_action_raw=np.zeros((4),np.float32)

    def __init__(self, algo_name, silent_mode):
        self.all_obstacle_points=[]
        self.all_obstacle_distances=[]

        self.set_the_first_spr=False

        self.policy_i=0

        self.heatmap_mat1=np.zeros((1000,1000))
        self.heatmap_image1=np.zeros((1000,1000,3))

        self.heatmap_mat=np.zeros((200,200))
        self.heatmap_image=np.zeros((200,200,3))

        self.znorm_r_number=0
        self.znorm_std_number=0

        self.rews_c0_raw_all=[]

        self.kpp=0
        self.timing_err=0

        self.data_dict = {}

        self.acc_original_r_all=[]

        self.acc_px_err_all=[]
        self.acc_py_err_all=[]
        self.acc_pz_err_all=[]

        self.acc_ax_err_all=[]
        self.acc_ay_err_all=[]
        self.acc_az_err_all=[]

        self.acc_don_all=[]

        self.acc_dwo_all=[]
        self.acc_dwg_all=[]
        self.acc_toh_all=[]

        self.acc_ydr_all=[]

        self.rotor_reset_starting_point=np.ones(3)
        self.reset_counter=0

        self.final_curiosity_reward_total=0.0

        self.max_average_curiosity=(0.0)
        self.min_average_curiosity=(0.0)

        self.check_near_ground_condition=False

        self.last_100_dones=np.zeros((100),np.float32)

        self.total_dones=0
        self.total_obstacle_hit=0
        self.rotor_to_obstacle_distance=0
        self.rotor_to_obstacle_distance_min=0

        self.total_obstacle_distance=0.0
        self.total_goal_distance=0.0
        self.total_yaw_direction_reward=0.0

        self.max_obstacle_reward=0.0
        self.min_obstacle_reward=0.0
        self.final_obstacle_reward_total=0.0

        self.all_total_dones=[]

        self.local_active_motors=4

        self.parameters=CL2FC_parameters(algo_name)

        if(self.parameters.deterministic_initiation):
            np.random.seed(1)
            random.seed(1)
            tf.set_random_seed(1)

        plt.pause(0.01)

        self.curiosity_current_observation_bulk = np.zeros((1, self.parameters.curiosity_trajectory_length,12))
        self.curiosity_next_observation_bulk  = np.zeros((1, self.parameters.curiosity_trajectory_length,12))
        self.curiosity_action_bulk = np.zeros((1, self.parameters.curiosity_trajectory_length,4))
        self.curiosity_reward_bulk = np.zeros((1, self.parameters.curiosity_trajectory_length,1))
        self.curiosity_reward_change_bulk = np.zeros((1, self.parameters.curiosity_trajectory_length,1))

        if(self.parameters.number_of_obstacles==1):
            self.obstacle_pose=np.zeros(2)
        else:
            self.obstacle_pose=np.zeros((self.parameters.number_of_obstacles,2))

        self.thrust_noise = OUNoise(4, sigma=0.2*0.05)
        self.thrust_cmds_damp = 0.0
        self.thrust_rot_damp = 0.0

        state_size=np.array((self.parameters.state_size))
        self.STATE_SIZE=np.array((self.parameters.state_size))
        self.observation_space=((Box(0.0, 1.0, shape=(self.STATE_SIZE,), dtype=np.float32)))

        self.n_env=1
        self.silent_mode=silent_mode

        self.ACTION_SIZE=4

        self.action_space = ((Box(-1, +1, (self.ACTION_SIZE,), dtype=np.float32)))#[0]

        self.all_rewards=[]

        for env_counter in range(self.n_env):
            self.x_position_raw.append(-100.0)

            self.robot_model_state.append(self.get_model_state_client(self.parameters.robot_name +'','world'))

            self.robot_state.append(ModelState())
            self.robot_state[env_counter].model_name=self.parameters.robot_name  + ''

            if(self.parameters.train_with_obstacle):
                if(self.parameters.number_of_obstacles==1):
                    self.obstacle_model_state.append(self.get_model_state_client('cylinder' + str(self.parameters.ind)+ str(0),'world'))
                    self.obstacle_state.append(ModelState())
                    self.obstacle_state[env_counter].model_name='cylinder' + str(self.parameters.ind)+ str(0)
                    self.obstacle_state[env_counter].pose = self.obstacle_model_state[env_counter].pose
                    self.obstacle_state[env_counter].twist = self.obstacle_model_state[env_counter].twist
                    self.obstacle_state[env_counter].reference_frame = 'world'
                else:
                    self.all_obstacle_state=[]
                    self.all_obstacle_model_state=[]

                    for ObI in range(self.parameters.number_of_obstacles):
                        self.obstacle_model_state=[]
                        self.obstacle_state = []

                        self.obstacle_model_state.append(self.get_model_state_client('cylinder' + str(self.parameters.ind)+ str(ObI),'world'))
                        self.obstacle_state.append(ModelState())

                        self.obstacle_state[env_counter].model_name='cylinder' + str(self.parameters.ind) + str(ObI)
                        self.obstacle_state[env_counter].pose = self.obstacle_model_state[env_counter].pose
                        self.obstacle_state[env_counter].twist = self.obstacle_model_state[env_counter].twist
                        self.obstacle_state[env_counter].reference_frame = 'world'

                        self.all_obstacle_state.append(self.obstacle_state)
                        self.all_obstacle_model_state.append(self.obstacle_model_state)

            self.robot_state[env_counter].pose = self.robot_model_state[env_counter].pose
            self.robot_state[env_counter].twist = self.robot_model_state[env_counter].twist
            self.robot_state[env_counter].reference_frame = 'world'

            x_new_position_difference = 0.0
            y_new_position_difference = 0.0
            z_new_position_difference = 0.0

            yaw_new_angle_difference=0.0

            self.x_new_position_difference.append(x_new_position_difference)
            self.y_new_position_difference.append(y_new_position_difference)
            self.z_new_position_difference.append(z_new_position_difference)

            self.yaw_new_angle_difference.append(yaw_new_angle_difference)

            self.rotors_sensors.append(amir_rotors_sensors(env_counter, self.silent_mode, True, x_new_position_difference, y_new_position_difference, z_new_position_difference, yaw_new_angle_difference, self.parameters))#self.parameters.sensor_noise))

            rospy.Subscriber('/' + self.parameters.robot_name + '' + "/ground_truth/odometry", Odometry, self.rotors_sensors[env_counter].odometry, queue_size=100, tcp_nodelay=True)#, queue_size=1, buff_size=1)

            self.velocity_publisher.append(rospy.Publisher('/' + self.parameters.robot_name + '' + '/command/motor_speed', Actuators, queue_size=10))

            desired_x_pose=0.0
            desired_y_pose=0.0
            desired_z_pose=150.0

            desired_yaw=0.0

            self.desired_x_pose.append(desired_x_pose)
            self.desired_y_pose.append(desired_y_pose)
            self.desired_z_pose.append(desired_z_pose)

            self.desired_pose_vector=np.array((desired_x_pose,desired_y_pose,desired_z_pose))

            self.desired_yaw_degree.append(desired_yaw)

            self.prev_callback_header_seq.append(0.0)
            self.give_accurates.append(give_accurate(env_counter))

            self.reset_timer.append(time.time())
            self.reset_flag.append(False)

        self.waiting_threshold=self.parameters.waiting_threshold

        self.pisition_x_difference_threshold=self.parameters.pisition_x_difference_threshold
        self.pisition_y_difference_threshold=self.parameters.pisition_y_difference_threshold
        self.pisition_z_difference_threshold=self.parameters.pisition_z_difference_threshold

        self.annealing_ground_fly=self.parameters.annealing_ground_fly

        self.L2FC_rewards=amir_rotors_bioinspired_rewards(parameters=self.parameters)

        self.legend_drawn=False

        self.min_reward=0.0
        self.max_reward=0.0

        self.all_rewards_accumulated=0.0

        if(self.parameters.curious):
            num_cpu=None

            config = tf.ConfigProto(
                allow_soft_placement=True,
                inter_op_parallelism_threads=num_cpu,
                intra_op_parallelism_threads=num_cpu)

            graph=None

            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.2

            self.curiosity_session=tf.Session(config=config, graph=graph)

            from CDRL_L2FC_curiosity_ICM import Curiosity

            from CDRL_L2FC_curiosity_HCM_SAS import Curiosity_HCM_SAS

            from CDRL_L2FC_curiosity_HCM_SAR import Curiosity_HCM_SAR

            if(self.parameters.curiosity_type=='ICM'):
                self.curiosity= Curiosity(self.curiosity_session,'ICM', self.parameters.curiosity_hl_s,self.parameters.curiosity_hl_n, env_state_space_dim=self.parameters.state_size, parameters=self.parameters, unsupType='regenerate')

            elif(self.parameters.curiosity_type=='HCM'):
                self.curiosity= Curiosity_HCM_SAS(self.curiosity_session,'HCM_SAS', self.parameters.curiosity_hl_s,self.parameters.curiosity_hl_n, env_state_space_dim=(self.parameters.curiosity_trajectory_length,12), parameters=self.parameters, unsupType='regenerate')

                if(self.parameters.curiosity_use_reward_head):
                    self.curiosity_reward_head= Curiosity_HCM_SAR(self.curiosity_session,'HCM_SAR', self.parameters.curiosity_hl_s,self.parameters.curiosity_hl_n, env_state_space_dim=(self.parameters.curiosity_trajectory_length,12),env_state_space_2_dim=(self.parameters.curiosity_trajectory_length,1), parameters=self.parameters, unsupType='regenerate')

            self.init_op = tf.global_variables_initializer()#initialize_all_variables()
            self.curiosity_session.run(self.init_op)

        #### all the lists

        horizon = self.parameters.timesteps_per_batch

        self.ac = ac = self.action_space.sample()
        self.ob = ob= self.observation_space.sample()

        position=[]
        position.append(0.0)
        position.append(0.0)
        position.append(0.0)

        angle=[]
        angle.append(0.0)
        angle.append(0.0)
        angle.append(0.0)

        self.obs = np.array([ob for _ in range(horizon)])
        self.rews = np.zeros(horizon, 'float32')
        self.rews_c0 = np.zeros(horizon, 'float32')
        self.rews_c1 = np.zeros(horizon, 'float32')

        self.rews_c0_raw = np.zeros(horizon, 'float32')
        self.rews_c0_normalized = np.zeros(horizon, 'float32')

        self.c0_mean = 0.0
        self.c0_std = 0.0
        self.c0_vari = 0.0

        self.vpreds = np.zeros(horizon, 'float32')
        self.vpreds_c0 = np.zeros(horizon, 'float32')
        self.vpreds_c1 = np.zeros(horizon, 'float32')


        self.news = np.zeros(horizon, 'int32')
        self.acs = np.array([ac for _ in range(horizon)])
        self.portions = np.array([ac for _ in range(horizon)])

        self.positions = np.array([position for _ in range(horizon)])
        self.angles = np.array([angle for _ in range(horizon)])
        self.tbtss = np.zeros(horizon, 'int32')

        self.prevacs = self.acs.copy()

        self.rews_sparse = np.zeros(horizon, 'float32')
        self.rews_negative = np.zeros(horizon, 'float32')

    def rotors_bc_do_things(self, action_c):

        reward = self.L2FC_rewards.rotors_reward(action_c,
        self.rotors_sensors[self.active_env].z_position,
        self.rotors_sensors[self.active_env].x_position,
        self.rotors_sensors[self.active_env].y_position,
        self.rotors_sensors[self.active_env].linear_velocity_z,
        self.rotors_sensors[self.active_env].x_angle,
        self.rotors_sensors[self.active_env].y_angle,
        self.rotors_sensors[self.active_env].z_angle,
        self.rotors_sensors[self.active_env].angular_velocity_x,
        self.rotors_sensors[self.active_env].angular_velocity_y,
        (self.desired_x_pose[self.active_env]),
        (self.desired_y_pose[self.active_env]),
        (self.desired_z_pose[self.active_env]),
        self.active_env)

        done=False

        if (self.rotors_sensors[self.active_env].z_position >= self.parameters.reset_pose_thresh_z):
            reward=-10.0
            done=True

        if ((self.rotors_sensors[self.active_env].z_position <= self.parameters.near_ground_terminal_height) and (self.check_near_ground_condition)):
            reward=-10.0
            done=True

        if ((self.rotors_sensors[self.active_env].x_position >= self.parameters.reset_pose_thresh_xy or self.rotors_sensors[self.active_env].y_position >= self.parameters.reset_pose_thresh_xy)
            or (self.rotors_sensors[self.active_env].x_position <= -self.parameters.reset_pose_thresh_xy or self.rotors_sensors[self.active_env].y_position <= -self.parameters.reset_pose_thresh_xy)):
            reward=-10.0
            done=True

        x_angle_difference = 0 - self.rotors_sensors[self.active_env].x_angle
        if (x_angle_difference < 0):
            x_angle_difference = x_angle_difference * -1.0

        y_angle_difference = 0 - self.rotors_sensors[self.active_env].y_angle
        if (y_angle_difference < 0):
            y_angle_difference = y_angle_difference * -1.0

        #### adding this to allow special manuovers
        if(self.rotors_sensors[self.active_env].z_position<=40):
            angle_threshold=80#45#90

        else:
            angle_threshold=360#45#90

        if ((x_angle_difference > angle_threshold or
    	      y_angle_difference > angle_threshold) ):
            reward=-10.0
            done=True

        return reward, done

    def rotors_bc_get_state(self,Multi=False,New=False):
        if(New):
            while(self.rotors_sensors[self.active_env].header.seq<=self.prev_callback_header_seq[self.active_env]):
                 time.sleep(0.0005)

        if(New):
            if(self.rotors_sensors[self.active_env].header.seq-self.prev_callback_header_seq[self.active_env]>1):
                self.timing_err+=1

        x_position_normalized=round(self.rotors_sensors[self.active_env].x_position_raw, self.parameters.round_float_to)
        y_position_normalized=round(self.rotors_sensors[self.active_env].y_position_raw, self.parameters.round_float_to)
        z_position_normalized=round(self.rotors_sensors[self.active_env].z_position_raw, self.parameters.round_float_to)

        x_linear_velocity_normalized=self.rotors_sensors[self.active_env].linear_velocity_x #/ 10.0
        y_linear_velocity_normalized=self.rotors_sensors[self.active_env].linear_velocity_y #/ 10.0
        z_linear_velocity_normalized=self.rotors_sensors[self.active_env].linear_velocity_z #/ 10.0

        x_angular_velocity_normalized = self.rotors_sensors[self.active_env].angular_velocity_x #/ 10.0
        y_angular_velocity_normalized = self.rotors_sensors[self.active_env].angular_velocity_y #/ 10.0
        z_angular_velocity_normalized = self.rotors_sensors[self.active_env].angular_velocity_z #/ 10.0

        linear_acceleration_x_normalized =self.rotors_sensors[self.active_env].linear_acceleration_x #/ 90.0
        linear_acceleration_y_normalized =self.rotors_sensors[self.active_env].linear_acceleration_y #/ 90.0
        linear_acceleration_z_normalized =self.rotors_sensors[self.active_env].linear_acceleration_z #/ 90.0

        angular_acceleration_x_normalized = self.rotors_sensors[self.active_env].angular_acceleration_x #/ 10.0
        angular_acceleration_y_normalized = self.rotors_sensors[self.active_env].angular_acceleration_y #/ 10.0
        angular_acceleration_z_normalized = self.rotors_sensors[self.active_env].angular_acceleration_z #/ 10.0

        x_euler=self.rotors_sensors[self.active_env].x_angle_rad
        y_euler=self.rotors_sensors[self.active_env].y_angle_rad
        z_euler=self.rotors_sensors[self.active_env].z_angle_rad

        rotmat=(tf3d.euler.euler2mat(x_euler, y_euler, z_euler)).reshape(9)

        x_error=self.rotors_sensors[self.active_env].x_position_raw
        y_error=self.rotors_sensors[self.active_env].y_position_raw
        z_error=self.rotors_sensors[self.active_env].z_position_raw

        ####

        cur_state_full = np.array([
        x_position_normalized, y_position_normalized, z_position_normalized,
        x_linear_velocity_normalized, y_linear_velocity_normalized, z_linear_velocity_normalized,
        linear_acceleration_x_normalized, linear_acceleration_y_normalized, linear_acceleration_z_normalized,
        rotmat[0], rotmat[1], rotmat[2],
        rotmat[3], rotmat[4], rotmat[5],
        rotmat[6], rotmat[7], rotmat[8],
        x_angular_velocity_normalized, y_angular_velocity_normalized, z_angular_velocity_normalized,
        angular_acceleration_x_normalized, angular_acceleration_y_normalized, angular_acceleration_z_normalized,
        self.previous_action_raw[0],self.previous_action_raw[1],self.previous_action_raw[2],self.previous_action_raw[3]])

        rotor_point=np.array((x_error,y_error, 0),np.float32)

        for obst_pose in range(self.parameters.number_of_obstacles):
            obstacle_point=np.array((self.obstacle_pose[obst_pose][0], self.obstacle_pose[obst_pose][1], 0),np.float32)
            rotor_to_obstacle_distance=abs(d(rotor_point, obstacle_point))
            obst_info=([self.obstacle_pose[obst_pose][0]-x_position_normalized,self.obstacle_pose[obst_pose][1]-y_position_normalized, rotor_to_obstacle_distance])

            cur_state_full=np.concatenate((cur_state_full, obst_info), axis=0)

        rotor_point=np.array((x_error,y_error,z_error),np.float32)
        desired_point=np.array((0,0,1.5),np.float32)
        rotor_to_goal_distance=abs(d(rotor_point, desired_point))

        goal_info=([rotor_to_goal_distance]) # 32

        cur_state_full=np.concatenate((cur_state_full, goal_info), axis=0)

        ####

        ob=cur_state_full

        self.prev_callback_header_seq[self.active_env]=self.rotors_sensors[self.active_env].header.seq
        # print('main env) seq: ' + str(self.rotors_sensors[self.active_env].header.seq))

        return ob

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def rotors_bc_run_action(self,action):
        EPS = 1e-6

        motor_damp_time_down=motor_damp_time_up=0.25

        torque_to_thrust = 0.05
        motor_linearity= 0.424

        dt=0.01

        action=(action + 1.0) / 2.0
        motor_tau_up = 4*dt/(motor_damp_time_up + EPS)
        motor_tau_down = 4*dt/(motor_damp_time_down + EPS)
        motor_tau = motor_tau_up * np.ones([4,])

        motor_tau[action < self.thrust_cmds_damp] = motor_tau_down
        motor_tau[motor_tau > 1.] = 1. #today

        thrust_rot = action**0.5

        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        thrust_noise = action * self.thrust_noise.noise()
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        final_action=self.thrust_cmds_damp
        action_c= final_action * self.parameters.action_multiplied_by * 2.0

        Speeds = rotors_move(np.clip(action_c, self.parameters.action_c_min, self.parameters.action_c_max), self.ACTION_SIZE, self.velocity_publisher[self.active_env], self.local_active_motors)

        self.previous_action_raw = action

        return action_c

    def rotors_bc_continuous_step(self, action, action_repeatation=1):
        portion=1.0
        action=np.clip(action, -1.0, 1.0)
        print('[{}] counter: {}, ROBOT: {}, Algorithm: {}'.format('GENERAL', self.UN_COUNTER, self.parameters.robot_name, self.parameters.algo_name))

        not_inside_the_area=True

        for _ in range(self.parameters.action_repeatation):
            total_reward=0.0

            total_x_error=0.0
            total_y_error=0.0
            total_z_error=0.0

            total_distance_error=0.0

            total_xa_error=0.0
            total_ya_error=0.0
            total_za_error=0.0

            obs=[]
            rewards=[]
            dones=[]
            rewards_sparse=[]
            rewards_negative=[]

            rewards_sparse.append(0)
            rewards_negative.append(0)


            for action_counter in range(len(action)):
                self.active_env=action_counter
                a=np.array(action)[self.active_env]


                if(np.isnan(a).any()):
                    print('[{}] Problematic Action: {}'.format(str(self.active_env),a))
                    time.sleep(1000000)

                else:
                    start_time = time.time()
                    action_c=self.rotors_bc_run_action(a)

            self.active_env=0

            ob=self.rotors_bc_get_state(False,True) #### -------> very important

            for action_counter in range(len(action)):

                self.active_env=action_counter

                if(True):
                    rew, new=self.rotors_bc_do_things(action_c)

                    start_time = time.time()

                    self.give_accurates[self.active_env].accumulated_rewards_so_far=self.give_accurates[self.active_env].accumulated_rewards_so_far+rew
                    self.give_accurates[self.active_env].accumulated_steps_so_far=self.give_accurates[self.active_env].accumulated_steps_so_far+1

                    if(self.give_accurates[self.active_env].ground_accurate_counter_used):

                        self.give_accurates[self.active_env].cumulative_reward_per_episode_flight=self.give_accurates[self.active_env].cumulative_reward_per_episode_flight+rew
                        self.give_accurates[self.active_env].cumulative_step_per_episode_flight=self.give_accurates[self.active_env].cumulative_step_per_episode_flight+1

                    is_tbts=False

                    if(self.parameters.enh):
                        position_z_difference = abs(self.rotors_sensors[self.active_env].z_position - self.desired_z_pose[self.active_env])
                        position_x_difference = abs(self.rotors_sensors[self.active_env].x_position - self.desired_x_pose[self.active_env])
                        position_y_difference = abs(self.rotors_sensors[self.active_env].y_position - self.desired_y_pose[self.active_env])

                        yaw_is_fine=False
                        if(self.parameters.move_in_yaw_direction):
                            if(self.rotors_sensors[self.active_env].z_angle_raw_eu < self.parameters.yaw_acceptable_error):
                                yaw_is_fine = True
                        else:
                            yaw_is_fine=True

                        if(position_z_difference < self.pisition_z_difference_threshold and
                            position_x_difference < self.pisition_x_difference_threshold and
                            position_y_difference < self.pisition_y_difference_threshold and
                            (self.UN_COUNTER)>self.start_resetting_counter and
                            self.parameters.time_based_terminal_state==True and
                            yaw_is_fine):

                            if(self.reset_flag[self.active_env]==False):
                                self.reset_timer[self.active_env]=0
                                self.reset_flag[self.active_env]=True
                                print('[{}] start resetting counter...'.format(str(self.active_env)))

                            self.reset_timer[self.active_env]+=1
                            not_inside_the_area=False

                            spr=0

                            print(colorize("[{}] counting resetting counter: {} seconds".format((self.active_env),(time.time() - self.reset_timer[self.active_env])), color='blue'))

                            if(spr>0):
                                self.set_the_first_spr=True

                        if(((self.reset_timer[self.active_env])>self.waiting_threshold) and
                            self.reset_flag[self.active_env]==True and
                            (self.UN_COUNTER)>self.start_resetting_counter):

                            self.reset_flag[self.active_env]=False
                            self.reset_timer[self.active_env]=0

                            is_tbts=True

                            rew=self.statistics.tbts_reward

                            rewards_sparse[0]=self.statistics.tbts_reward

                            self.reach_time=0.0

                            new=True

                        else:
                            self.reach_time+=1

                        if(self.reach_time>=self.parameters.activate_ground_reset_on and self.parameters.activate_ground_reset):
                            self.check_near_ground_condition=True


                    real_reward=0.0

                    real_reward=rew

                    rewards.append(real_reward)
                    dones.append(new)

                    if(new):
                        rewards_negative[0]+=rew

                    if(new):
                        if(is_tbts):
                            # self.total_finishes+=1
                            obs.append((self.rotors_bc_reset(type='good_done')))
                            print(colorize("[{}] resetting rotors".format((self.active_env)), color='green'))
                        else:
                            self.total_dones+=1
                            obs.append((self.rotors_bc_reset(type='done')))
                            print(colorize("[{}] resetting rotors".format((self.active_env)), color='red'))

                    else:
                        if(is_tbts==False):
                            if(self.active_env==0):
                                obs.append((self.rotors_bc_get_state(False,False)))
                            else:
                                obs.append((self.rotors_bc_get_state(False,False)))
                        else:
                            obs.append((self.rotors_bc_reset(type='good_done')))
                            print(colorize("[{}] resetting rotors".format((self.active_env)), color='green'))

                    x_error=self.rotors_sensors[self.active_env].x_position_raw
                    y_error=self.rotors_sensors[self.active_env].y_position_raw
                    z_error=self.rotors_sensors[self.active_env].z_position_raw

                    xa_error=self.rotors_sensors[self.active_env].x_angle_raw_eu
                    ya_error=self.rotors_sensors[self.active_env].y_angle_raw_eu
                    za_error=self.rotors_sensors[self.active_env].z_angle_raw_eu

                total_x_error+=abs(x_error)
                total_y_error+=abs(y_error)
                total_z_error+=abs(z_error)

                rotor_point=np.array((x_error,y_error,z_error),np.float32)
                desired_point=np.array((0,0,1.5),np.float32)

                distance_error=abs(d(rotor_point, desired_point))

                total_distance_error+=distance_error

                total_xa_error+=abs(xa_error)
                total_ya_error+=abs(ya_error)
                total_za_error+=abs(za_error)

                total_reward+=rew

        actual_distance_error=total_distance_error/float(self.n_env)

        self.accumulated_x_error+=total_x_error
        self.accumulated_y_error+=total_y_error
        self.accumulated_z_error+=total_z_error

        self.accumulated_distance_error+=actual_distance_error

        self.accumulated_xa_error+=total_xa_error
        self.accumulated_ya_error+=total_ya_error
        self.accumulated_za_error+=total_za_error

        self.average_reward=self.total_accumulated_reward/(self.UN_COUNTER + 1)

        self.UN_COUNTER+=1

        position=[]
        position.append(self.rotors_sensors[self.active_env].x_position)
        position.append(self.rotors_sensors[self.active_env].y_position)
        position.append(self.rotors_sensors[self.active_env].z_position)

        angle=[]
        angle.append(self.rotors_sensors[self.active_env].x_angle)
        angle.append(self.rotors_sensors[self.active_env].y_angle)
        angle.append(self.rotors_sensors[self.active_env].z_angle)

        talkative=True
        talkative=False

        orginal_reward=rewards[0]
        self.episode_accumulated_reward_org+=rewards[0]

        if(rewards[0]<self.min_reward):
            self.min_reward=rewards[0]

        if(rewards[0]>self.max_reward):
            self.max_reward=rewards[0]

        self.all_rewards_accumulated+=rewards[0]

        x_error=self.rotors_sensors[self.active_env].x_position_raw
        y_error=self.rotors_sensors[self.active_env].y_position_raw
        z_error=self.rotors_sensors[self.active_env].z_position_raw

        rotor_point=np.array((x_error,y_error,z_error),np.float32)
        desired_point=np.array((0,0,1.5),np.float32)

        max_distance=np.max(abs(rotor_point-desired_point))

        distance_error=abs(d(rotor_point, desired_point))

        x_error=self.rotors_sensors[self.active_env].x_position_raw
        y_error=self.rotors_sensors[self.active_env].y_position_raw
        z_error=self.rotors_sensors[self.active_env].z_position_raw

        if(self.parameters.train_with_obstacle):
            final_obstacle_reward=0.0

            rotor_point=np.array((x_error,y_error, 0),np.float32)

            self.rotor_to_obstacle_distance_min=100
            for obsti in range(self.parameters.number_of_obstacles):
                obstacle_point=np.array((self.obstacle_pose[obsti][0], self.obstacle_pose[obsti][1], 0),np.float32)
                distance=abs(d(rotor_point, obstacle_point))
                if(distance<self.rotor_to_obstacle_distance_min):
                    self.rotor_to_obstacle_distance_min=distance

                self.rotor_to_obstacle_distance += distance

            obstacle_reward = self.rotor_to_obstacle_distance * -1.0

            self.rotor_to_obstacle_distance = self.rotor_to_obstacle_distance / float(self.parameters.number_of_obstacles)

            self.total_goal_distance+=distance_error

            if((obstacle_reward)>self.max_obstacle_reward):
                self.max_obstacle_reward=(obstacle_reward)

            if((obstacle_reward)<self.min_obstacle_reward):
                self.min_obstacle_reward=(obstacle_reward)

            self.final_obstacle_reward_total+=obstacle_reward
            final_obstacle_reward_average=self.final_obstacle_reward_total/self.UN_COUNTER
            if(final_obstacle_reward_average==0.0):
                final_obstacle_reward_average=0.000001

            final_obstacle_reward=obstacle_reward/final_obstacle_reward_average
            final_obstacle_reward= final_obstacle_reward * self.parameters.obstacle_reward_coef

        else:
            obstacle_reward=0.0
            final_obstacle_reward=0.0

        move_in_yaw_direction_reward=0.0

        if(self.parameters.move_in_yaw_direction):

            move_in_yaw_direction_reward=-1 * abs(self.rotors_sensors[self.active_env].z_angle_raw_eu) * self.parameters.yaw_direction_reward_coef
            self.total_yaw_direction_reward+=move_in_yaw_direction_reward
            rewards[0]+=move_in_yaw_direction_reward

        linvel_mag_reward=0.0
        if(self.parameters.increase_lingvel):
            linvel_mag=np.linalg.norm([self.rotors_sensors[self.active_env].linear_velocity_x, self.rotors_sensors[self.active_env].linear_velocity_y, self.rotors_sensors[self.active_env].linear_velocity_z])

            linvel_mag_reward=linvel_mag * self.parameters.linvel_reward_coef
            self.total_linvel_reward+=linvel_mag_reward
            rewards[0]+=linvel_mag_reward

        ang_vel_mag_reward=0.0
        if(self.parameters.increase_angvel):
            ang_vel_mag=np.linalg.norm([self.rotors_sensors[self.active_env].angular_velocity_x, self.rotors_sensors[self.active_env].angular_velocity_y, self.rotors_sensors[self.active_env].angular_velocity_z])

            ang_vel_mag_reward=ang_vel_mag * self.parameters.ang_vel_reward_coef
            self.total_ang_vel_reward+=ang_vel_mag_reward
            rewards[0]+=ang_vel_mag_reward

        self.episode_accumulated_linvel_mag_reward+=linvel_mag_reward
        self.episode_accumulated_ang_vel_mag_reward+=ang_vel_mag_reward

        self.episode_accumulated_move_in_yaw_direction_reward+=move_in_yaw_direction_reward

        x_error_m=self.rotors_sensors[self.active_env].x_position_raw# * 100.0
        y_error_m=self.rotors_sensors[self.active_env].y_position_raw# * 100.0

        x_error_m_rounded1=round(x_error_m, 2)
        y_error_m_rounded1=round(y_error_m, 2)

        x_error_c_m_rounded11=int(x_error_m_rounded1 * 100)
        y_error_c_m_rounded11=int(y_error_m_rounded1 * 100)

        x_error_c_m_rounded21= x_error_c_m_rounded11 + 500
        y_error_c_m_rounded21= y_error_c_m_rounded11 + 500

        x_error_m_rounded=round(x_error_m, 1)
        y_error_m_rounded=round(y_error_m, 1)

        x_error_c_m_rounded1=int(x_error_m_rounded * 10)
        y_error_c_m_rounded1=int(y_error_m_rounded * 10)

        x_error_c_m_rounded2= x_error_c_m_rounded1 + 100
        y_error_c_m_rounded2= y_error_c_m_rounded1 + 100

        if(x_error_c_m_rounded2>0 and x_error_c_m_rounded2<200 and y_error_c_m_rounded2>=0 and y_error_c_m_rounded2<200):
            self.heatmap_mat[x_error_c_m_rounded2,y_error_c_m_rounded2]+=1

        if(x_error_c_m_rounded21>0 and x_error_c_m_rounded21<1000 and y_error_c_m_rounded21>=0 and y_error_c_m_rounded21<1000):
            self.heatmap_mat1[x_error_c_m_rounded21,y_error_c_m_rounded21]+=1

        if(self.parameters.train_with_obstacle):
            obstacle_radius=25/100
            rotors_radius=40/100

            minimum_possible_distance_between_rotors_and_obstacle=obstacle_radius + rotors_radius

            if(self.rotor_to_obstacle_distance_min<minimum_possible_distance_between_rotors_and_obstacle):
                self.total_dones+=1
                self.total_obstacle_hit+=1

                obs=[]
                rewards=[]
                dones=[]

                print(colorize("[{}] resetting rotors for obstacles".format((self.active_env)), color='red'))

                obs.append((self.rotors_bc_reset(type='done')))
                new=True
                reward=self.parameters.hit_obstacle_negative_reward
                rewards.append(reward)
                dones.append(new)

                rewards_negative=[]
                rewards_negative.append(reward)

            else:
                rewards[0]+=final_obstacle_reward
                self.total_obstacle_distance+=self.rotor_to_obstacle_distance

        self.reset_counter+=1

        sparse_reward=0.0

        curiosity_reward = 0.0

        # print('self.timing_err: {}'.format(self.timing_err))

        return obs, rewards, np.array(dones), position, angle, is_tbts, portion, rewards_sparse, rewards_negative

    def draw_diagrams(self):
        self.parameters.ext_reward_plot.plot(self.parameters.episode_counter, np.clip((self.episode_accumulated_reward_org/float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)),-15,+15), marker='.', color='black', label='original')

        self.acc_original_r_all.append((self.episode_accumulated_reward_org/float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)))

        self.last_100_dones=np.roll(self.last_100_dones, -1)

        self.last_100_dones[99]=self.total_dones

        self.all_total_dones.append(self.total_dones)

        self.parameters.aux_reward_plot.plot(self.parameters.episode_counter, (self.episode_accumulated_move_in_yaw_direction_reward/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for)), marker='.', color='blue', label='yaw rew')

        self.parameters.aux_reward_plot.plot(self.parameters.episode_counter, (self.episode_accumulated_linvel_mag_reward/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for)), marker='.', color='black', label='linvel mag')
        self.parameters.aux_reward_plot.plot(self.parameters.episode_counter, (self.episode_accumulated_ang_vel_mag_reward/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for)), marker='.', color='maroon', label='angvel mag')

        self.parameters.failed_flight_plot.plot(self.parameters.episode_counter, self.total_dones/self.parameters.repeat_action_for, marker='.', color='black', label='done')

        self.parameters.total_obstacle_hit_plot.plot(self.parameters.episode_counter,self.total_obstacle_hit/self.parameters.repeat_action_for, marker='.', color='red', label='hit_obstacle')
        self.parameters.obstacle_goal_distance_plot.plot(self.parameters.episode_counter,self.total_obstacle_distance/float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), marker='.', color='blue', label='distance_with_obstacle')
        self.parameters.obstacle_goal_distance_plot.plot(self.parameters.episode_counter,self.total_goal_distance/float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), marker='.', color='green', label='distance_with_goal')

        self.acc_dwo_all.append(self.total_obstacle_distance/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for))
        self.acc_dwg_all.append(self.total_goal_distance/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for))

        self.acc_toh_all.append(self.total_obstacle_hit)

        self.acc_ydr_all.append((self.episode_accumulated_move_in_yaw_direction_reward/float((self.parameters.timesteps_per_batch-1)*self.parameters.repeat_action_for)))

        self.acc_don_all.append(self.total_dones)


        self.parameters.odom_pos_err_plot.plot(self.parameters.episode_counter, self.accumulated_x_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), marker='.', color='red', label='px')
        self.parameters.odom_pos_err_plot.plot(self.parameters.episode_counter, self.accumulated_y_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), marker='.', color='green', label='py')
        self.parameters.odom_pos_err_plot.plot(self.parameters.episode_counter, (self.accumulated_z_error - 1.5)/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), marker='.', color='blue', label='pz')

        self.parameters.odom_att_err_plot.plot(self.parameters.episode_counter, np.clip(self.accumulated_xa_error / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), 0, 0.5), marker='.', color='red', label='ax')
        self.parameters.odom_att_err_plot.plot(self.parameters.episode_counter, np.clip(self.accumulated_ya_error / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), 0, 0.5), marker='.', color='green', label='ay')
        self.parameters.odom_att_err_plot.plot(self.parameters.episode_counter, np.clip(self.accumulated_za_error / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for), 0, 0.5), marker='.', color='blue', label='az')

        self.acc_px_err_all.append(self.accumulated_x_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))
        self.acc_py_err_all.append(self.accumulated_y_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))
        self.acc_pz_err_all.append(self.accumulated_z_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))

        self.acc_ax_err_all.append(self.accumulated_xa_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))
        self.acc_ay_err_all.append(self.accumulated_ya_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))
        self.acc_az_err_all.append(self.accumulated_za_error/ float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for))

        total_accumulated_reward_per_episode_flight=0.0
        total_accumulated_step_per_episode_flight=0.0

        total_accumulated_done_counter=0.0

        for env_counter in range(self.n_env):
            total_accumulated_reward_per_episode_flight+=self.give_accurates[env_counter].cumulative_reward_per_episode_flight
            total_accumulated_step_per_episode_flight+=self.give_accurates[env_counter].cumulative_step_per_episode_flight

            self.give_accurates[env_counter].cumulative_reward_per_episode_flight=0.0
            self.give_accurates[env_counter].cumulative_step_per_episode_flight=0.0

            total_accumulated_done_counter+=self.give_accurates[env_counter].done_counter

            self.give_accurates[env_counter].done_counter = 0.0

        average_accumulated_done_counter=total_accumulated_done_counter/float(self.n_env)

        averaged_accumulated_reward_per_episode_flight=total_accumulated_reward_per_episode_flight/float(self.n_env)
        averaged_accumulated_step_per_episode_flight=total_accumulated_step_per_episode_flight/float(self.n_env)

        averaged_accumulated_reward_per_episode_flight = averaged_accumulated_reward_per_episode_flight / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)
        averaged_accumulated_step_per_episode_flight = averaged_accumulated_step_per_episode_flight / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)

        self.parameters.last_100_failed_flight_plot.clear()

        self.parameters.last_100_failed_flight_plot.set_xlabel("Step", fontsize=10)
        self.parameters.last_100_failed_flight_plot.set_ylabel("Reward", fontsize=10)
        self.parameters.last_100_failed_flight_plot.set_title("", fontsize=10)
        self.parameters.last_100_failed_flight_plot.axis('auto')
        self.parameters.last_100_failed_flight_plot.grid(which='both')

        self.parameters.last_100_failed_flight_plot.plot(self.last_100_dones, linestyle='-', linewidth=2, color='black', label='done')

        average_distance_error=self.accumulated_distance_error / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)

        average_angle_error=(self.accumulated_xa_error + self.accumulated_ya_error + self.accumulated_za_error) / float(self.parameters.timesteps_per_batch*self.parameters.repeat_action_for)

        start_time=time.time()

        save_outs=False
        save_outs=True

        if(save_outs):
            np.savetxt('data/' + self.parameters.title_precise + '_acc_original_r_all.out', np.array(self.acc_original_r_all))

            np.savetxt('data/' + self.parameters.title_precise + '_acc_px_err_all.out', np.array(self.acc_px_err_all))
            np.savetxt('data/' + self.parameters.title_precise + '_acc_py_err_all.out', np.array(self.acc_py_err_all))
            np.savetxt('data/' + self.parameters.title_precise + '_acc_pz_err_all.out', np.array(self.acc_pz_err_all))

            np.savetxt('data/' + self.parameters.title_precise + '_acc_ax_err_all.out', np.array(self.acc_ax_err_all))
            np.savetxt('data/' + self.parameters.title_precise + '_acc_ay_err_all.out', np.array(self.acc_ay_err_all))
            np.savetxt('data/' + self.parameters.title_precise + '_acc_az_err_all.out', np.array(self.acc_az_err_all))

            np.savetxt('data/' + self.parameters.title_precise + '_acc_don_all.out', np.array(self.acc_don_all))

            if(self.parameters.train_with_obstacle):
                np.savetxt('data/' + self.parameters.title_precise + '_acc_dwo_all.out', np.array(self.acc_dwo_all))
                np.savetxt('data/' + self.parameters.title_precise + '_acc_dwg_all.out', np.array(self.acc_dwg_all))
                np.savetxt('data/' + self.parameters.title_precise + '_acc_toh_all.out', np.array(self.acc_toh_all))

            if(self.parameters.move_in_yaw_direction):
                np.savetxt('data/' + self.parameters.title_precise + '_acc_ydr_all.out', np.array(self.acc_ydr_all))

        print(colorize("consumed time for saving the info: %.4f seconds"%(time.time() - start_time), color='red'))

        self.parameters.save_pdf()

    def train_and_update_with_curiosity(self):
        action2=None
        talkative=False

        curiosity_reward_rewards_change_loss=0

        if(self.parameters.curiosity_type=='HCM' or self.parameters.curiosity_type=='pos_to_pos_trajectory_bulk'):
            k_range=self.parameters.timesteps_per_batch-1-self.parameters.curiosity_trajectory_length

        else:
            k_range=self.parameters.timesteps_per_batch-1

        for k in range(k_range):
            self.kpp+=1
            if(self.parameters.curious):
                curiosity_reward=0.0
                our_curiosity_reward=0.0
                bonus = 0.0

                if(k>(0+self.parameters.curiosity_trajectory_length)):
                    if(1==1):
                        curiosity_action=np.copy(self.acs[k])
                        curiosity_next_observation=np.copy(self.obs[k+1])
                        curiosity_current_observation=np.copy(self.obs[k - (self.parameters.curiosity_trajectory_length)])

                        if(self.parameters.reward_type_for_curiosity=='all'):
                            reward_without_curiosity=np.copy(self.rews[k - (self.parameters.curiosity_trajectory_length)])

                            if(self.parameters.reward_type_for_curiosity_only_change):
                                reward_without_curiosity=np.copy(self.rews[k] - self.rews[k - (self.parameters.curiosity_trajectory_length)])

                        elif(self.parameters.reward_type_for_curiosity=='sparse'):
                            reward_without_curiosity=np.copy(self.rews_sparse[k - (self.parameters.curiosity_trajectory_length)])
                            if(self.parameters.reward_type_for_curiosity_only_change):
                                reward_without_curiosity=np.copy(self.rews_sparse[k] - self.rews_sparse[k - (self.parameters.curiosity_trajectory_length)])

                        elif(self.parameters.reward_type_for_curiosity=='negative'):
                            reward_without_curiosity=np.copy(self.rews_negative[k - (self.parameters.curiosity_trajectory_length)])
                            if(self.parameters.reward_type_for_curiosity_only_change):
                                reward_without_curiosity=np.copy(self.rews_negative[k] - self.rews_negative[k - (self.parameters.curiosity_trajectory_length)])

                        elif(self.parameters.reward_type_for_curiosity=='sparse_negative'):
                            if(self.rews_negative[k - (self.parameters.curiosity_trajectory_length)]<0):
                                reward_without_curiosity=np.copy(self.rews_negative[k - (self.parameters.curiosity_trajectory_length)])
                                if(self.parameters.reward_type_for_curiosity_only_change):
                                    reward_without_curiosity=np.copy(self.rews_negative[k] - self.rews_negative[k - (self.parameters.curiosity_trajectory_length)])

                            else:
                                reward_without_curiosity=np.copy(self.rews_sparse[k - (self.parameters.curiosity_trajectory_length)])
                                if(self.parameters.reward_type_for_curiosity_only_change):
                                    reward_without_curiosity=np.copy(self.rews_sparse[k] - self.rews_sparse[k - (self.parameters.curiosity_trajectory_length)])

                        reward_without_curiosity_array=np.zeros((1))
                        reward_without_curiosity_array[0] = reward_without_curiosity

                        if(self.parameters.curiosity_type=='HCM'):
                            self.curiosity_current_observation_bulk = (self.obs[k-(self.parameters.curiosity_trajectory_length):k, np.r_[0:3,9:18]]).reshape(1, self.parameters.curiosity_trajectory_length,12)
                            self.curiosity_next_observation_bulk = (self.obs[k: k+(self.parameters.curiosity_trajectory_length), np.r_[0:3,9:18]]).reshape(1, self.parameters.curiosity_trajectory_length,12)
                            self.curiosity_action_bulk = (self.acs[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2)),:]).reshape(1, self.parameters.curiosity_trajectory_length,4)

                            if(self.parameters.reward_type_for_curiosity=='all'):
                                self.curiosity_reward_bulk = (self.rews[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1)
                                self.curiosity_reward_change_bulk = self.curiosity_reward_bulk
                                if(self.parameters.reward_type_for_curiosity_only_change):
                                    self.curiosity_reward_change_bulk = (self.rews[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1) - (self.rews[k-(int(self.parameters.curiosity_trajectory_length/2))-1: k+(int(self.parameters.curiosity_trajectory_length/2))-1]).reshape(1, self.parameters.curiosity_trajectory_length,1)

                            elif(self.parameters.reward_type_for_curiosity=='sparse'):
                                self.curiosity_reward_bulk = (self.rews_sparse[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1)
                                self.curiosity_reward_change_bulk = self.curiosity_reward_bulk
                                if(self.parameters.reward_type_for_curiosity_only_change):
                                    self.curiosity_reward_change_bulk = (self.rews_sparse[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1) - (self.rews_sparse[k-(int(self.parameters.curiosity_trajectory_length/2))-1: k+(int(self.parameters.curiosity_trajectory_length/2))-1]).reshape(1, self.parameters.curiosity_trajectory_length,1)

                            elif(self.parameters.reward_type_for_curiosity=='negative'):
                                self.curiosity_reward_bulk = (self.rews_negative[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1)
                                self.curiosity_reward_change_bulk = self.curiosity_reward_bulk
                                if(self.parameters.reward_type_for_curiosity_only_change):
                                    self.curiosity_reward_change_bulk = (self.rews_negative[k-(int(self.parameters.curiosity_trajectory_length/2)): k+(int(self.parameters.curiosity_trajectory_length/2))]).reshape(1, self.parameters.curiosity_trajectory_length,1) - (self.rews_negative[k-(int(self.parameters.curiosity_trajectory_length/2))-1: k+(int(self.parameters.curiosity_trajectory_length/2))-1]).reshape(1, self.parameters.curiosity_trajectory_length,1)

                            curiosity_reward_2_bulk = self.rews[k-1] - self.rews[k+self.parameters.curiosity_trajectory_length]

                        if(self.parameters.curiosity_trajectory_length>0):
                            if(self.parameters.curiosity_trajectory_action=='waypoint_posatt'):
                                curiosity_waypoint_action=np.zeros(7)
                                curiosity_waypoint_action[0:3]=curiosity_next_observation[np.r_[0:3]] - curiosity_current_observation[np.r_[0:3]]

                                curiosity_waypoint_action[3:7]=(tf3d.quaternions.mat2quat(curiosity_next_observation[np.r_[9:18]])).reshape(4) - (tf3d.quaternions.mat2quat(curiosity_current_observation[np.r_[9:18]])).reshape(4)

                                curiosity_action=curiosity_waypoint_action
                            elif(self.parameters.curiosity_trajectory_action=='waypoint'):
                                curiosity_waypoint_action=curiosity_next_observation[np.r_[0:3]] - curiosity_current_observation[np.r_[0:3]]
                                curiosity_action=curiosity_waypoint_action

                            elif(self.parameters.curiosity_trajectory_action=='waypoint_3section'):
                                section1=self.curiosity_current_observation_bulk[0,0,np.r_[0:3]] - self.curiosity_next_observation_bulk[0,0,np.r_[0:3]]
                                section2=self.curiosity_current_observation_bulk[0,int(self.parameters.curiosity_trajectory_length/2),np.r_[0:3]] - self.curiosity_next_observation_bulk[0,int(self.parameters.curiosity_trajectory_length/2),np.r_[0:3]]
                                section3=self.curiosity_current_observation_bulk[0,-1,np.r_[0:3]] - self.curiosity_next_observation_bulk[0,-1,np.r_[0:3]]

                                curiosity_waypoint_action=np.concatenate((section1, section2, section3), axis=0)
                                curiosity_action=curiosity_waypoint_action

                            else:
                                curiosity_waypoint_action=curiosity_action

                    if(self.parameters.use_curiosity_pathak):

                        if(self.parameters.curiosity_type=='ICM'):
                            curiosity_reward_forward_loss, curiosity_reward_inv_loss= self.curiosity.predictor.pred_bonus((curiosity_current_observation), ((curiosity_next_observation)/1.0), (curiosity_action))
                            predicted_action = self.curiosity.predictor.pred_act((curiosity_current_observation), ((curiosity_next_observation)/1.0))

                        elif(self.parameters.curiosity_type=='HCM'):
                            if(self.parameters.curiosity_trajectory_action=='waypoint_posatt' or self.parameters.curiosity_trajectory_action=='waypoint_3section'):
                                curiosity_reward_forward_loss, curiosity_reward_inv_loss, curiosity_reward_rewards_change_loss= self.curiosity.pred_bonus(self.curiosity_session, self.curiosity_current_observation_bulk, self.curiosity_next_observation_bulk, curiosity_waypoint_action, self.curiosity_reward_change_bulk)
                                if(self.parameters.curiosity_use_reward_head):
                                    curiosity_reward_forward_loss_reward_head, curiosity_reward_inv_loss_reward_head, curiosity_reward_rewards_change_loss_reward_head= self.curiosity_reward_head.pred_bonus(self.curiosity_session,self.curiosity_current_observation_bulk, self.curiosity_reward_change_bulk, curiosity_waypoint_action, self.curiosity_reward_change_bulk)

                                    curiosity_reward_forward_loss = (curiosity_reward_forward_loss + curiosity_reward_forward_loss_reward_head)/2.0
                                    curiosity_reward_inv_loss = (curiosity_reward_inv_loss + curiosity_reward_inv_loss_reward_head)/2.0
                                    curiosity_reward_rewards_change_loss = (curiosity_reward_rewards_change_loss + curiosity_reward_rewards_change_loss_reward_head)/2.0

                            else:
                                curiosity_reward_forward_loss, curiosity_reward_inv_loss, curiosity_reward_rewards_change_loss= self.curiosity.pred_bonus(self.curiosity_session,self.curiosity_current_observation_bulk, self.curiosity_next_observation_bulk, self.curiosity_action_bulk, self.curiosity_reward_change_bulk)
                                if(self.parameters.curiosity_use_reward_head):
                                    curiosity_reward_forward_loss_reward_head, curiosity_reward_inv_loss_reward_head, curiosity_reward_rewards_change_loss_reward_head= self.curiosity_reward_head.pred_bonus(self.curiosity_session,self.curiosity_current_observation_bulk, self.curiosity_reward_change_bulk, self.curiosity_action_bulk, self.curiosity_reward_change_bulk)

                                    curiosity_reward_forward_loss = (curiosity_reward_forward_loss + curiosity_reward_forward_loss_reward_head)/2.0
                                    curiosity_reward_inv_loss = (curiosity_reward_inv_loss + curiosity_reward_inv_loss_reward_head)/2.0
                                    curiosity_reward_rewards_change_loss = (curiosity_reward_rewards_change_loss + curiosity_reward_rewards_change_loss_reward_head)/2.0

                            predicted_action = self.curiosity.pred_act(self.curiosity_session,self.curiosity_current_observation_bulk, self.curiosity_next_observation_bulk)

                            if(self.parameters.curiosity_use_reward_head):
                                predicted_action_reward_head = self.curiosity_reward_head.pred_act(self.curiosity_session,self.curiosity_current_observation_bulk, self.curiosity_reward_change_bulk)

                        if((self.parameters.episode_counter_2 % self.parameters.use_curiosity_every_n_episodes==0) or self.parameters.use_intermittent_curiosity==False):

                            if(self.parameters.Curiosity_dynamics_type=='fwd'):
                                bonus = curiosity_reward_forward_loss #+ curiosity_reward_inv_loss

                            elif(self.parameters.Curiosity_dynamics_type=='inv'):
                                bonus = curiosity_reward_inv_loss

                            else:
                                bonus = curiosity_reward_forward_loss + curiosity_reward_inv_loss

                        else:
                            bonus=0.0


                        curiosity_reward = bonus

                        self.parameters.episode_accumulated_forward_curiosity_reward=0
                        self.parameters.episode_accumulated_inverse_curiosity_reward=0

                        curiosity_reward=self.parameters.ALPHA1 * curiosity_reward

                        ####

                        # #### for parameters
                        ASSIGN_P1=True
                        if(ASSIGN_P1):
                            if(self.parameters.curiosity_trajectory_length>0):
                                self.curiosity.curiosity_rewards[self.parameters.Curiosity_counter]=(curiosity_reward)
                                self.curiosity.curiosity_episode_actions[self.parameters.Curiosity_counter]=(curiosity_action)
                                self.curiosity.curiosity_episode_predicted_actions[self.parameters.Curiosity_counter]=(predicted_action)

                            else:

                                self.curiosity.curiosity_rewards[self.parameters.Curiosity_counter]=(curiosity_reward)
                                self.curiosity.curiosity_episode_actions[self.parameters.Curiosity_counter]=(curiosity_action)
                                self.curiosity.curiosity_episode_predicted_actions[self.parameters.Curiosity_counter]=(predicted_action)

                                ####
                ASSIGN_P2=True
                if(ASSIGN_P2):
                    # print('assign p2')
                    if(k>(0+self.parameters.curiosity_trajectory_length)):
                        if(self.parameters.curiosity_trajectory_length==0):
                            self.curiosity.curiosity_current_states[self.parameters.Curiosity_counter]=(curiosity_current_observation)
                            if(self.parameters.curiosity_type=='normal_to_reward'):
                                self.curiosity.curiosity_next_states[self.parameters.Curiosity_counter]=(reward_without_curiosity_array)
                            else:
                                self.curiosity.curiosity_next_states[self.parameters.Curiosity_counter]=(curiosity_next_observation)

                            self.curiosity.curiosity_actions[self.parameters.Curiosity_counter]=curiosity_action
                            self.curiosity.curiosity_actions2[self.parameters.Curiosity_counter]=action2
                            self.curiosity.curiosity_bonuses[self.parameters.Curiosity_counter]=bonus

                            self.curiosity.rewards_without_curiosity[self.parameters.Curiosity_counter]=reward_without_curiosity_array

                        elif(self.parameters.curiosity_trajectory_length>0):


                            if(self.parameters.curiosity_type=='HCM' or self.parameters.curiosity_type=='pos_to_pos_trajectory_bulk'):
                                self.curiosity.curiosity_current_states[self.parameters.Curiosity_counter]=(self.curiosity_current_observation_bulk)
                                self.curiosity.curiosity_next_states[self.parameters.Curiosity_counter]=(self.curiosity_next_observation_bulk)
                                self.curiosity.curiosity_rewards_changes[self.parameters.Curiosity_counter]=(self.curiosity_reward_change_bulk)
                                if(self.parameters.curiosity_trajectory_action=='waypoint_posatt' or self.parameters.curiosity_trajectory_action=='waypoint_3section' ):
                                    self.curiosity.curiosity_actions[self.parameters.Curiosity_counter]=curiosity_waypoint_action

                                else:
                                    self.curiosity.curiosity_actions[self.parameters.Curiosity_counter]=self.curiosity_action_bulk

                                if(self.parameters.curiosity_use_reward_head):
                                    self.curiosity_reward_head.curiosity_current_states[self.parameters.Curiosity_counter]=(self.curiosity_current_observation_bulk)
                                    self.curiosity_reward_head.curiosity_next_states[self.parameters.Curiosity_counter]=self.curiosity_reward_change_bulk#(self.curiosity_next_observation_bulk)
                                    self.curiosity_reward_head.curiosity_rewards_changes[self.parameters.Curiosity_counter]=(self.curiosity_reward_change_bulk)
                                    if(self.parameters.curiosity_trajectory_action=='waypoint_posatt' or self.parameters.curiosity_trajectory_action=='waypoint_3section' ):
                                        self.curiosity_reward_head.curiosity_actions[self.parameters.Curiosity_counter]=curiosity_waypoint_action

                                    else:
                                        self.curiosity_reward_head.curiosity_actions[self.parameters.Curiosity_counter]=self.curiosity_action_bulk

                            else:
                                self.curiosity.curiosity_trajectory_current_states[self.parameters.Curiosity_counter]=(curiosity_current_observation[np.r_[0:3]])
                                self.curiosity.curiosity_trajectory_next_states[self.parameters.Curiosity_counter]=(curiosity_next_observation[np.r_[0:3]])
                                self.curiosity.curiosity_actions[self.parameters.Curiosity_counter]=curiosity_waypoint_action
                                self.curiosity.rewards_without_curiosity[self.parameters.Curiosity_counter]=reward_without_curiosity_array

                            self.curiosity.curiosity_actions2[self.parameters.Curiosity_counter]=action2
                            self.curiosity.curiosity_bonuses[self.parameters.Curiosity_counter]=bonus


                        self.parameters.Curiosity_counter+=1

                        if(self.parameters.Curiosity_counter>=self.parameters.Curiosity_database_size):
                            self.parameters.Curiosity_counter=0

                        if(self.parameters.CuriosityCurrentlyFilled_counter<self.parameters.Curiosity_database_size):
                            self.parameters.CuriosityCurrentlyFilled_counter+=1

                curiosity_reward=curiosity_reward * self.parameters.pathak_curiosity_coef

                if(curiosity_reward<self.parameters.min_curiosity_reward):
                    self.parameters.min_curiosity_reward=curiosity_reward

                if(curiosity_reward>self.parameters.max_curiosity_reward):
                    self.parameters.max_curiosity_reward=curiosity_reward

            else:
                curiosity_reward=0.0

            self.rews_c0_raw[k]=curiosity_reward
            self.rews_c0_raw_all.append(curiosity_reward)

        self.rews_c0_normalized=self.rews_c0_raw

        if(self.UN_COUNTER>=self.parameters.curiosity_starting_training_episode):
            for k in range(k_range):
                if(self.parameters.curious):
                    curiosity_reward=self.rews_c0_normalized[k]

                    curiosity_reward=curiosity_reward*self.parameters.curiosity_second_coef

                    if(self.parameters.curiosity_trajectory_length>0):
                        linear_alpha=0.75
                        if(self.parameters.curiosity_trajectory_type=='single'):
                            old_rew=self.rews[k]
                            if(self.parameters.use_multi_vpred):
                                self.rews_c0[k-self.parameters.curiosity_trajectory_length]+=curiosity_reward

                            else:
                                self.rews[k-self.parameters.curiosity_trajectory_length]+=curiosity_reward

                        else:
                            curiosity_reward_linear=0
                            if(k>self.parameters.curiosity_trajectory_length):

                                for po in range(self.parameters.curiosity_trajectory_length):
                                    po_reverse = self.parameters.curiosity_trajectory_length - po
                                    old_rew=self.rews[k]
                                    if(self.parameters.curiosity_trajectory_type=='constant'):
                                        if(self.parameters.use_multi_vpred):
                                            self.rews_c0[k-self.parameters.curiosity_trajectory_length+po]+=((curiosity_reward/self.parameters.curiosity_trajectory_length)*1)

                                        else:
                                            self.rews[k-self.parameters.curiosity_trajectory_length+po]+=((curiosity_reward/self.parameters.curiosity_trajectory_length)*1)

                                    elif(self.parameters.curiosity_trajectory_type=='linear'):
                                        if(self.parameters.curiosity_type=='HCM'):
                                            if(po==0):
                                                curiosity_reward_linear = curiosity_reward

                                                curiosity_reward_linear_final = curiosity_reward_linear * (self.parameters.linear_beta)

                                                if(self.parameters.use_multi_vpred):
                                                    self.rews_c0[k]+=curiosity_reward_linear_final
                                                else:
                                                    self.rews[k]+=curiosity_reward_linear_final

                                            else:
                                                curiosity_reward_linear = curiosity_reward_linear * (self.parameters.linear_alpha)

                                                curiosity_reward_linear_final = curiosity_reward_linear * (self.parameters.linear_beta)

                                                if(self.parameters.use_multi_vpred):
                                                    self.rews_c0[k+po]+=curiosity_reward_linear_final
                                                    self.rews_c0[k-po]+=curiosity_reward_linear_final
                                                else:
                                                    self.rews[k+po]+=curiosity_reward_linear_final
                                                    self.rews[k-po]+=curiosity_reward_linear_final

                                        else:
                                            if(po==0):
                                                curiosity_reward_linear = curiosity_reward

                                            else:
                                                curiosity_reward_linear = curiosity_reward_linear * (self.parameters.linear_alpha)

                                            curiosity_reward_linear_final = curiosity_reward_linear * (self.parameters.linear_beta)

                                            if(self.parameters.use_multi_vpred):
                                                self.rews_c0[k-po]+=curiosity_reward_linear_final

                                            else:
                                                self.rews[k-po]+=curiosity_reward_linear_final

                    else:
                        old_rew=self.rews[k]

                        if(self.parameters.use_multi_vpred):
                            self.rews_c0[k-self.parameters.curiosity_trajectory_length]+=curiosity_reward

                        else:
                            self.rews[k-self.parameters.curiosity_trajectory_length]+=curiosity_reward

        TRAIN=True
        if(TRAIN):
            if(self.parameters.episode_counter%self.parameters.TRAIN_CURIOSITY_EVERY==0 and self.UN_COUNTER>=self.parameters.curiosity_starting_training_episode):
                ##### training the curiosity

                print('start training the curiosity module...')

                if(self.parameters.curiosity_trajectory_length==0):

                    self.curiosity.predictor.update_mean_std(self.curiosity.curiosity_episode_actions, self.curiosity.curiosity_episode_predicted_actions)
                    self.curiosity.process(self.curiosity_session, self.curiosity.curiosity_current_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_next_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_actions[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_bonuses[0:self.parameters.CuriosityCurrentlyFilled_counter])

                elif(self.parameters.curiosity_trajectory_length>0):
                    if(self.parameters.curiosity_type=='HCM'):
                        self.curiosity.process(self.curiosity_session, self.curiosity.curiosity_current_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_next_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_actions[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_rewards_changes[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_bonuses[0:self.parameters.CuriosityCurrentlyFilled_counter])

                        if(self.parameters.curiosity_use_reward_head):
                            self.curiosity_reward_head.process(self.curiosity_session, self.curiosity_reward_head.curiosity_current_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity_reward_head.curiosity_next_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity_reward_head.curiosity_actions[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity_reward_head.curiosity_rewards_changes[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity_reward_head.curiosity_bonuses[0:self.parameters.CuriosityCurrentlyFilled_counter])

                    else:
                        self.curiosity.predictor.update_mean_std(self.curiosity.curiosity_episode_actions, self.curiosity.curiosity_episode_predicted_actions)
                        self.curiosity.process(self.curiosity_session, self.curiosity.curiosity_trajectory_current_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_trajectory_next_states[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_actions[0:self.parameters.CuriosityCurrentlyFilled_counter], self.curiosity.curiosity_bonuses[0:self.parameters.CuriosityCurrentlyFilled_counter])

                print('curiosity module trained...')

        return True

    def rotors_bc_reset(self, type):
        self.set_the_first_spr=False

        self.check_near_ground_condition=False

        self.reach_time=0.0

        if(type=='start'):
            print('start')
            use_ground_accurate_counter=True
            start=True
            done=False
            horizon=False
        elif(type=='done'):
            print('done')
            use_ground_accurate_counter=False
            start=False
            done=True
            horizon=False
            self.give_accurates[self.active_env].done_counter = self.give_accurates[self.active_env].done_counter + 1

        elif(type=='good_done'):
            print('good done')
            use_ground_accurate_counter=False
            start=False
            done=True
            horizon=False

        else:
            use_ground_accurate_counter=True
            start=False
            done=False
            horizon=True

        rotors_stop_engine(self.velocity_publisher[self.active_env])

        discrete_random_pose=False
        self.robot_state[self.active_env].pose.orientation.x=0.0
        self.robot_state[self.active_env].pose.orientation.y=0.0
        self.robot_state[self.active_env].pose.orientation.z=0.0
        self.robot_state[self.active_env].pose.orientation.w=0.0

        self.robot_state[self.active_env].twist.linear.x=0.0
        self.robot_state[self.active_env].twist.linear.y=0.0
        self.robot_state[self.active_env].twist.linear.z=0.0

        self.robot_state[self.active_env].twist.angular.x=0.0
        self.robot_state[self.active_env].twist.angular.y=0.0
        self.robot_state[self.active_env].twist.angular.z=0.0

        if(self.UN_COUNTER>0):
            rewards_to_steps=0.0
            if(self.give_accurates[self.active_env].accumulated_steps_so_far>0):
                rewards_to_steps=self.give_accurates[self.active_env].accumulated_rewards_so_far/self.give_accurates[self.active_env].accumulated_steps_so_far

            self.give_accurates[self.active_env].accumulated_steps_so_far=0.0
            self.give_accurates[self.active_env].accumulated_rewards_so_far=0.0

            if(self.give_accurates[self.active_env].ground_accurate_counter_used):
                if(self.give_accurates[self.active_env].accurate_counter_point_ground[self.give_accurates[self.active_env].accurate_counter_ground-1]==-1000000.0):
                    self.give_accurates[self.active_env].accurate_counter_point_ground[self.give_accurates[self.active_env].accurate_counter_ground-1]=rewards_to_steps
                else:
                    self.give_accurates[self.active_env].accurate_counter_point_ground[self.give_accurates[self.active_env].accurate_counter_ground-1]=(self.give_accurates[self.active_env].accurate_counter_point_ground[self.give_accurates[self.active_env].accurate_counter_ground-1]+rewards_to_steps)/2.0

            else:
                if(self.give_accurates[self.active_env].accurate_counter_point[self.give_accurates[self.active_env].accurate_counter-1]==-1000000.0):
                    self.give_accurates[self.active_env].accurate_counter_point[self.give_accurates[self.active_env].accurate_counter-1]=rewards_to_steps
                else:
                    self.give_accurates[self.active_env].accurate_counter_point[self.give_accurates[self.active_env].accurate_counter-1]=(self.give_accurates[self.active_env].accurate_counter_point[self.give_accurates[self.active_env].accurate_counter-1]+rewards_to_steps)/2.0


        if(discrete_random_pose):
            x_random_pose=np.random.randint(-2,3, size=1) # x value [-3,3] discrete
            y_random_pose=np.random.randint(-2,3, size=1) # y value [-3,3] discrete
            z_random_pose=np.random.randint(0,3, size=1) # z value [0,2] discrete

        else:
            x_random_pose=random.uniform(-2, 2) # x value [-3,3] continuous
            y_random_pose=random.uniform(-2, 2) # y value [-3,3] continuous
            z_random_pose=0.10

            if(self.parameters.enh):
                if(use_ground_accurate_counter):
                    if(self.parameters.train_with_obstacle):
                        if(self.parameters.obstacle_initiation_method=='method1'):
                            x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate_ground_obstacle()

                        else:
                            space=self.parameters.inner_obstacle_space+self.parameters.obstacle_space+self.parameters.obstacle_space_with_drone_space_distance
                            x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate_ground_obstacle(space=space,method=self.parameters.obstacle_initiation_method)

                    else:
                        x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate_ground()

                else:
                    if(self.parameters.train_with_obstacle):
                        if(self.parameters.obstacle_initiation_method=='method1'):
                            x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate_obstacle()

                        else:
                            space=self.parameters.inner_obstacle_space+self.parameters.obstacle_space+self.parameters.obstacle_space_with_drone_space_distance
                            x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate_obstacle(space=space,method=self.parameters.obstacle_initiation_method)
                    else:
                        x_random_pose, y_random_pose = self.give_accurates[self.active_env].give_x_y_accurate()

                    if(self.parameters.aggressive_maneuver):
                        x_random_orientation=np.random.randint(-180,180, size=1) # x value [-3,3] discrete
                        y_random_orientation=np.random.randint(-180,180, size=1) # y value [-3,3] discrete
                        z_random_orientation=np.random.randint(-180,180, size=1)

                        quaternion = quaternion_from_euler(math.radians(x_random_orientation), math.radians(y_random_orientation), math.radians(z_random_orientation))

                        self.robot_state[self.active_env].pose.orientation.x=quaternion[0]
                        self.robot_state[self.active_env].pose.orientation.y=quaternion[1]
                        self.robot_state[self.active_env].pose.orientation.z=quaternion[2]
                        self.robot_state[self.active_env].pose.orientation.w=quaternion[3]

            x_random_pose_org=x_random_pose
            y_random_pose_org=y_random_pose

            x_random_pose=x_random_pose+self.x_new_position_difference[self.active_env]
            y_random_pose=y_random_pose+self.y_new_position_difference[self.active_env]

        self.give_accurates[self.active_env].ground_accurate_counter_used=use_ground_accurate_counter

        if(done):
            if(self.UN_COUNTER<=self.annealing_ground_fly):
                epsilon_fly=self.UN_COUNTER/self.annealing_ground_fly
            else:
                epsilon_fly=1.0

            if(epsilon_fly>1.0):
                epsilon_fly=1.0

            ground_fly_chance=random.uniform(0.0,1.0)

            if(self.parameters.aggressive_maneuver):
                epsilon_fly=self.parameters.epsilon_fly

            epsilon_fly=self.parameters.epsilon_fly

            if(ground_fly_chance<=epsilon_fly):
                z_random_pose=0.10

            else:
                z_random_pose=random.uniform(0.10, 3)

            if(self.parameters.move_in_yaw_direction):
                quaternion = quaternion_from_euler(math.radians(0.0), math.radians(0.0), math.radians(np.random.randint(-180,180, size=1)))

                self.robot_state[self.active_env].pose.orientation.x=quaternion[0]
                self.robot_state[self.active_env].pose.orientation.y=quaternion[1]
                self.robot_state[self.active_env].pose.orientation.z=quaternion[2]
                self.robot_state[self.active_env].pose.orientation.w=quaternion[3]

            else:
                self.robot_state[self.active_env].pose.orientation.x=0.0
                self.robot_state[self.active_env].pose.orientation.y=0.0
                self.robot_state[self.active_env].pose.orientation.z=0.0
                self.robot_state[self.active_env].pose.orientation.w=0.0

            if(self.parameters.enh==False):
                z_random_pose=0.10

        self.robot_state[self.active_env].pose.position.x=x_random_pose
        self.robot_state[self.active_env].pose.position.y=y_random_pose
        self.robot_state[self.active_env].pose.position.z=z_random_pose

        self.rotor_reset_starting_point=np.array((self.robot_state[self.active_env].pose.position.x,self.robot_state[self.active_env].pose.position.y,self.robot_state[self.active_env].pose.position.z),np.float32)
        self.reset_counter=0

        self.robot_model_state[self.active_env] = self.set_model_state_client(self.robot_state[self.active_env])

        not_pass=True
        not_pass_counter=0
        while(not_pass):

            if(round(x_random_pose_org,2)==round(self.rotors_sensors[self.active_env].x_position_raw,2) and
            round(y_random_pose_org,2)==round(self.rotors_sensors[self.active_env].y_position_raw,2)):
                not_pass=False

            self.robot_model_state[self.active_env] = self.set_model_state_client(self.robot_state[self.active_env])

            not_pass_counter+=1

            if(not_pass_counter>20):
                if(abs(round(x_random_pose_org,2)-round(self.rotors_sensors[self.active_env].x_position_raw,2))<=0.02 and
                abs(round(y_random_pose_org,2)-round(self.rotors_sensors[self.active_env].y_position_raw,2))<=0.02):
                    not_pass=False

            time.sleep(0.001)

        if(self.parameters.train_with_obstacle):
            if(self.parameters.number_of_obstacles>1):
                obstacle_poses_x=[]
                obstacle_poses_y=[]
                constant_obstacle_number=float(np.random.randint(0,self.parameters.number_of_obstacles, size=1))
                constant_obstacle_number=0
                print('ADDING SEVERAL OBSTACLES')
                for NoO in range(self.parameters.number_of_obstacles):
                    #method 2

                    if(self.parameters.obstacle_initiation_method=='method2'):
                        not_pass=True
                        while(not_pass):
                            x_random_pose_portion=x_random_pose_org/50.0
                            y_random_pose_portion=y_random_pose_org/50.0

                            rnd_portion=float(np.random.randint(0,50, size=1))
                            rnd_portion=float(np.random.randint(10,45, size=1))

                            x_obst_pose_max=x_random_pose_portion * rnd_portion
                            y_obst_pose_max=y_random_pose_portion * rnd_portion

                            x_rnd=random.uniform(-2,2)
                            y_rnd=random.uniform(-2,2)

                            if(constant_obstacle_number!=NoO):
                                x_obst_pose_max+=x_rnd
                                y_obst_pose_max+=y_rnd

                            x_random_pose=self.x_new_position_difference[self.active_env] + x_obst_pose_max
                            y_random_pose=self.y_new_position_difference[self.active_env] + y_obst_pose_max
                            current_obst_pose=[x_random_pose, y_random_pose, 0]
                            obstacle_radius=25/100

                            if(NoO==0):
                                okay_to_add_the_obstacle=True

                            else:
                                okay_to_add_the_obstacle=True
                                for ObS in range(len(obstacle_poses_x)):
                                    current_obst_pose=[x_random_pose, y_random_pose, 0]
                                    previous_obst_pose=[obstacle_poses_x[ObS], obstacle_poses_y[ObS], 0]

                                    drone_pose=[x_random_pose_org, y_random_pose_org,0]
                                    center_pose=[self.x_new_position_difference[self.active_env],self.y_new_position_difference[self.active_env],0]

                                    distance=d(current_obst_pose, previous_obst_pose)

                                    distance_drone=d(current_obst_pose, drone_pose)
                                    distance_center=d(current_obst_pose, center_pose)

                                    if(distance<(obstacle_radius*3.0) or
                                        distance_drone<(obstacle_radius*4.0) or
                                        distance_center<(obstacle_radius*4.0)):
                                        okay_to_add_the_obstacle=False

                            if(okay_to_add_the_obstacle):
                                obstacle_poses_x.append(x_random_pose)
                                obstacle_poses_y.append(y_random_pose)

                                self.obstacle_pose[NoO][0]=x_random_pose-self.x_new_position_difference[self.active_env]
                                self.obstacle_pose[NoO][1]=y_random_pose-self.y_new_position_difference[self.active_env]

                                not_pass=False

                        if(x_random_pose_org>0):
                            x_random_pose=random.uniform(self.x_new_position_difference[self.active_env]+1, x_random_pose - 0.5) # x value [-3,3] continuous
                        elif(x_random_pose_org<=0):
                            x_random_pose=random.uniform(self.x_new_position_difference[self.active_env]-1, x_random_pose + 0.5) # x value [-3,3] continuous

                        if(y_random_pose_org>0):
                            y_random_pose=random.uniform(self.y_new_position_difference[self.active_env]+1, y_random_pose - 0.5) # y value [-3,3] continuous
                        elif(y_random_pose_org<=0):
                            y_random_pose=random.uniform(self.y_new_position_difference[self.active_env]-1, y_random_pose + 0.5) # y value [-3,3] continuous

                    self.all_obstacle_state[NoO][self.active_env].pose.position.x=obstacle_poses_x[NoO]#self.obstacle_pose[0]
                    self.all_obstacle_state[NoO][self.active_env].pose.position.y=obstacle_poses_y[NoO]#self.obstacle_pose[1]

                    self.all_obstacle_state[NoO][self.active_env].pose.position.z=1.616197

                    self.all_obstacle_state[NoO][self.active_env].pose.orientation.x=0.0
                    self.all_obstacle_state[NoO][self.active_env].pose.orientation.y=0.0
                    self.all_obstacle_state[NoO][self.active_env].pose.orientation.z=0.0
                    self.all_obstacle_state[NoO][self.active_env].pose.orientation.w=0.0

                    self.all_obstacle_model_state[NoO][self.active_env] = self.set_model_state_client(self.all_obstacle_state[NoO][self.active_env])

                    #### making sure rotors pose is accurate
                    not_pass=True
                    not_pass_counter=0
                    while(not_pass or not_pass_counter>5):

                        if(round(obstacle_poses_x[NoO],2)==round(self.all_obstacle_state[NoO][self.active_env].pose.position.x,2) and
                        round(obstacle_poses_y[NoO],2)==round(self.all_obstacle_state[NoO][self.active_env].pose.position.y,2)):
                            not_pass=False

                        print('*')

                        self.all_obstacle_model_state[NoO][self.active_env] = self.set_model_state_client(self.all_obstacle_state[NoO][self.active_env])



                        time.sleep(0.001)


        return self.rotors_bc_get_state(False,True)#ob
        ####

class give_accurate:
    env_id=100

    #### Normal Accurate Counter
    done_counter=0
    cumulative_step_per_episode_flight=0
    cumulative_reward_per_episode_flight=0.0

    accumulated_steps_so_far=0
    accumulated_rewards_so_far=0.0

    accurate_counter_point=[]
    for i in range(16):
        accurate_counter_point.append(-1000000.0)

    accurate_counter=16
    saved_accurate_counter=16
    previous_was_min=False

    #### Ground Accurate Counter
    ground_accurate_counter_used=True

    accurate_counter_point_ground=[]
    for i in range(16):
        accurate_counter_point_ground.append(-1000000.0)

    accurate_counter_ground=16
    saved_accurate_counter_ground=16
    previous_was_min_ground=False



    def __init__(self,env_id):
        self.env_id=env_id

    def give_x_y_accurate_obstacle(self,space=1.0,method='method1'):
        min_first=False
        epsilon_greedy_min_sequencial=False#True#False#True one error

        if(epsilon_greedy_min_sequencial):
            min_first_epsilon=0.5

            chance=random.uniform(0,1.0)

            if(chance<min_first_epsilon):
                min_first=True

            else:
                min_first=False


        if(min_first):
            if(self.previous_was_min==False):
                self.saved_accurate_counter=self.accurate_counter

            self.previous_was_min=True
            min_point=1000000.0

            for i in range(12):
                if(self.accurate_counter_point[i]<min_point):

                    min_point=self.accurate_counter_point[i]
                    self.accurate_counter=i+1

            if(self.accurate_counter==1): #a2
                x_random_pose=random.uniform(-1.0*space,0*space) #(-1.0, 3) -> error
                y_random_pose=random.uniform(1.0*space,2.0*space)
            if(self.accurate_counter==2): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==3): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

            if(self.accurate_counter==4): #b2
                x_random_pose=random.uniform(0*space, -1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==5): #b3
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==6): #b4
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter==7): #c2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==8): #c3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==9): #c4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter==10): #d2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==11): #d3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==12): #d4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

        else:
            if(self.previous_was_min):
                self.accurate_counter=self.saved_accurate_counter

            self.previous_was_min=False

            self.accurate_counter+=1

            if(self.accurate_counter>12):
                self.accurate_counter=1

            if(self.accurate_counter==1): #a2
                x_random_pose=random.uniform(-1.0*space,0*space) #(-1.0, 2.0) -> error
                y_random_pose=random.uniform(1.0*space,2.0*space)
            if(self.accurate_counter==2): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==3): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

            if(self.accurate_counter==4): #b2
                x_random_pose=random.uniform(0*space, -1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==5): #b3
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==6): #b4
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter==7): #c2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==8): #c3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter==9): #c4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter==10): #d2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==11): #d3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter==12): #d4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

        return x_random_pose, y_random_pose

    def give_x_y_accurate_ground_obstacle(self,space=1.0,method='method1'):
        min_first_ground=False
        epsilon_greedy_min_sequencial_ground=True

        if(epsilon_greedy_min_sequencial_ground):
            min_first_epsilon_ground=0.2
            chance_ground=random.uniform(0,1.0)

            if(chance_ground<min_first_epsilon_ground):
                min_first_ground=True

                all_equal=True
                for i in range(12):
                    if(self.accurate_counter_point_ground[i]==8192):
                        print('.')
                    else:
                        all_equal=False
                        break

                if(all_equal):
                    min_first_ground=False

            else:
                min_first_ground=False

        if(min_first_ground):
            if(self.previous_was_min_ground==False):
                self.saved_accurate_counter_ground=self.accurate_counter_ground

            self.previous_was_min_ground=True
            min_point_ground=1000000.0

            for i in range(12):
                if(self.accurate_counter_point_ground[i]<min_point_ground):
                    min_point_ground=self.accurate_counter_point_ground[i]
                    self.accurate_counter_ground=i+1

            if(self.accurate_counter_ground==1): #a2
                x_random_pose=random.uniform(-1.0*space,0*space) #(-1.0, 2.0) -> error
                y_random_pose=random.uniform(1.0*space,2.0*space)
            if(self.accurate_counter_ground==2): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==3): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

            if(self.accurate_counter_ground==4): #b2
                x_random_pose=random.uniform(0*space, -1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==5): #b3
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==6): #b4
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter_ground==7): #c2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==8): #c3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==9): #c4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter_ground==10): #d2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==11): #d3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==12): #d4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

        else:
            if(self.previous_was_min_ground):
                self.accurate_counter_ground=self.saved_accurate_counter_ground

            self.previous_was_min_ground=False

            self.accurate_counter_ground+=1

            if(self.accurate_counter_ground>12):
                self.accurate_counter_ground=1

            if(self.accurate_counter_ground==1): #a2
                x_random_pose=random.uniform(-1.0*space,0*space) #(-1.0, 2.0) -> error
                y_random_pose=random.uniform(1.0*space,2.0*space)
            if(self.accurate_counter_ground==2): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==3): #a1
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)

            if(self.accurate_counter_ground==4): #b2
                x_random_pose=random.uniform(0*space, -1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==5): #b3
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==6): #b4
                x_random_pose=random.uniform(-1.0*space, -2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter_ground==7): #c2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==8): #c3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(-1.0*space, -2.0*space)
            if(self.accurate_counter_ground==9): #c4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, -1.0*space)

            if(self.accurate_counter_ground==10): #d2
                x_random_pose=random.uniform(0*space, 1.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==11): #d3
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(1.0*space, 2.0*space)
            if(self.accurate_counter_ground==12): #d4
                x_random_pose=random.uniform(1.0*space, 2.0*space)
                y_random_pose=random.uniform(0*space, 1.0*space)


        return x_random_pose, y_random_pose

class OUNoise:
    """Ornstein–Uhlenbeck process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state
