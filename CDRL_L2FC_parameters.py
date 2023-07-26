
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from baselines.common.models import mlp, cnn_small, cnn

from time import gmtime, strftime

class CL2FC_parameters:

    uid=1

    gamma=0.99
    lam=0.95

    state_normalization=True

    aggressive_maneuver=True
    enh = True

    mx_kl=0.01

    reset_flag=False
    reset_timer=0.0

    optim_batch=512

    network_name='N_MLP'

    neural_network_hwomany_layers=2
    neural_network_hwomany_neural_each_layer=256

    clip_param=0.2

    max_grad_norm=0.05

    action_c_max=800.0
    action_c_min=150.0

    action_repeatation=1

    entcoeff=0.01

    pisition_z_difference_threshold=25.0
    pisition_x_difference_threshold=50.0
    pisition_y_difference_threshold=50.0

    angle_x_multiplier=1.0
    angle_y_multiplier=1.0
    angle_z_multiplier=1.0

    annealing_ground_fly=1e6

    normalized_position_z_reward_multiplier=10.0

    normalized_position_x_reward_multiplier=12.0
    normalized_position_y_reward_multiplier=12.0

    normalized_angle_reward_multiplier=float(1.0/25.0)

    time_based_terminal_state=True

    tbts_reward=500


    optim_epoch=15

    vf_stpsize=3e-4

    activation_function='relu'#'leaky_relu'#'tanh'#'leaky_relu'
    activation_function='tanh'#'leaky_relu' CUDA_VISIBLE_DEVICES=0 python a_rotors_bc_ppo.py

    action_multiplied_by=500.0

    stop_counter_starting_spisode=0

    epsilon_fly=0.5

    normalized_position_z_reward_multiplier=10.0
    normalized_position_x_reward_multiplier=10.0
    normalized_position_y_reward_multiplier=10.0

    gaussian_fixed_var=True#False

    activate_ground_reset_on = 20 #300
    activate_ground_reset = True

    no_velocity_and_acceleration=False

    norm_state=True
    norm_state=False

    norm_reward=True
    norm_reward=False

    near_ground_terminal_height=20

    schedule='linear'
    schedule='constant'

    neural_net='normal'

    train_with_obstacle=False
    train_with_obstacle=True

    state_type = "pos_linvel_linacc_rotmat_angvel_angacc_prevaction_obstaclepose_dist" #obstaclepose
    state_size=32
    neural_net='obstacle'
    neural_net='obstacle2'

    timesteps_per_batch=1024*16
    timesteps_per_batch=1024*8
    timesteps_per_batch=1024*4

    timesteps_per_batch=512*1

    repeat_action_for=1
    # repeat_action_for=2
    # repeat_action_for=5

    curiosity_trained_for_the_first_time=False
    curiosity_trained_for_the_first_time=True
    curiosity_starting_training_episode=int(1 * (timesteps_per_batch))

    curiosity_starting_training_epoch=100

    curiosity_round_float_to=1
    curiosity_round_float_to=2
    curiosity_round_float_to=3

    use_multi_vpred=False
    use_multi_vpred=True

    vf_loss_coef=1.0
    vf_c0_loss_coef=1.0

    obstacle_reward_coef=1.0
    obstacle_reward_coef=0.01

    hit_obstacle_negative_reward=-1
    hit_obstacle_negative_reward=-10

    move_in_yaw_direction=False
    move_in_yaw_direction=True

    yaw_acceptable_error = 0.2 # radian == 10 degrees
    yaw_acceptable_error = 0.1 # radian == 5 degrees

    yaw_direction_reward_coef=1.0

    linvel_reward_coef=0.3
    ang_vel_reward_coef=0.06

    reset_pose_thresh=5000
    reset_pose_thresh=10000
    reset_pose_thresh=750

    reset_pose_thresh_xy=500
    reset_pose_thresh_z=500


    number_of_obstacles=3

    obstacle_initiation_method='method2'
    # state_size=32
    state_size=(29 + (3 * number_of_obstacles))

    if(obstacle_initiation_method=='method2'):
        reset_pose_thresh_xy=1000
        reset_pose_thresh_z=500

    ind=1

    waiting_threshold=100

    Curiosity_database_size=int(16*4*1024) # 32 K

    curiosity_use_reward_head=False
    curiosity_use_reward_head=True

    obstacle_space=1.0
    inner_obstacle_space=1.0
    obstacle_with_drone_distance=0.5
    outer_obstacle_space_for_drone=1.0

    inner_obstacle_space=2.0
    obstacle_space=2.0
    obstacle_space_with_drone_space_distance=1.0

    curiosity_deterministic_initiation=False
    curiosity_deterministic_initiation=True

    deterministic_initiation=False
    # deterministic_initiation=True

    curiosity_trajectory_length=100
    reward_type_for_curiosity_only_change=False

    vf_stpsize_c=3e-5
    vf_stpsize_c=3e-4


    curiosity_hl_n=2
    curiosity_hl_s=128
    TRAIN_CURIOSITY_EVERY=4
    curiosity_epoch=100

    curiosity_type_limited_type='none'
    curiosity_FORWARD_LOSS_WT=0.7

    curiosity_second_coef=6.0

    curiosity_round_float_to=2

    repeat_action_for=2

    waiting_threshold=waiting_threshold * repeat_action_for

    consider_rewards_change_loss=False

    linear_alpha=0.9

    round_float_to=2

    curiosity_batch_size=512#128

    use_curiosity_every_n_episodes=1

    clip_action=True
    clip_action=False

    episode_counter_2=0

    use_curiosity_pathak=True

    accumulated_curiosity_reward_raw=0.0

    curiosity_mean_ig=0
    curiosity_stddev_ig=1.0
    curiosity_total_std_dev=0.0

    min_curiosity_reward=100000.0
    max_curiosity_reward=0.0

    curiosity_episode_counter=0

    limited_att_roll_pitch=0.785398163 # 45 degree
    limited_pos_xy=1.5 # centimeter

    curiosity_engine='PATHAK-STATE-ACTION-PREDICTOR'
    Z_NORMALIZATION_TYPE='EPISODE_DATA'

    Curiosity_training_step=int(2*1024)
    Curiosity_training_step=int(4*1024)
    Curiosity_training_step=int(8*1024)

    Curiosity_training_step=timesteps_per_batch

    curiosity_head='nature'

    Curiosity_counter=0
    pathak_curiosity_coef=1.0
    pathak_curiosity_coef=2.0

    algo_mission=''

    embodiment=True#False#True
    curiosity_engine='PATHAK-STATE-ACTION-PREDICTOR'
    curiosity_engine='PATHAK-STATE-ACTION-PREDICTOR-Z-NORMALIZED'

    ALPHA1=1.0 * 1.0#1.0 #curiosity_reward
    ALPHA2=1.0 #- ALPHA1
    ABRIL_ALPHA=0.5
    Z_NORMALIZATION_TYPE='EPISODE_DATA'

    RENEW_CURIOSITY=False
    RENEW_CURIOSITY_EVERY=100
    learning_algo='curious_BC'

    curiosity_ROLLOUT_MAXLEN=64
    curiosity_GRAD_NORM_CLIP=40.0
    curiosity_LEARNING_RATE=1e-4
    curiosity_PREDICTION_LR_SCALE=1.0

    CuriosityCurrentlyFilled_counter=0

    robot_name='hummingbird' + str(ind)

    x_init=0
    x_init=0

    filename_str=""

    loading=True
    loading=False

    def __init__(self, algo_name):
        self.algo_name=algo_name
        if(algo_name=='PPO'):
            self.curious=False
            print('PPO')
            self.curiosity_use_reward_head=False
            self.number_of_curiosity_heads=0
            self.Curiosity_dynamics_type=''
            self.curiosity_trajectory_type=''
            self.curiosity_type='none'
            self.curiosity_trajectory_length=0
            self.curiosity_trajectory_action=''
            self.reward_type_for_curiosity=''

            self.increase_lingvel=True
            self.increase_angvel=True

            self.use_multi_vpred=False

        elif(algo_name=='PPO_ICM'):
            self.curious=True

            self.curiosity_use_reward_head=False
            self.number_of_curiosity_heads=1
            self.Curiosity_dynamics_type='fwd'
            self.curiosity_trajectory_type='single'
            self.curiosity_type='ICM'
            self.curiosity_trajectory_length=0
            self.curiosity_trajectory_action='motorspeeds'
            self.reward_type_for_curiosity='negative'

            print('PPO_ICM')
            self.increase_lingvel=True
            self.increase_angvel=True
            self.use_multi_vpred=True


        elif(algo_name=='PPO_HCM'):
            self.curious=True
            self.curiosity_use_reward_head=True
            self.number_of_curiosity_heads=5
            self.Curiosity_dynamics_type='fwd_inv'
            self.curiosity_trajectory_type='linear'
            self.curiosity_type='HCM'
            self.curiosity_trajectory_length=100
            self.curiosity_trajectory_action='waypoint_3section'
            self.reward_type_for_curiosity='negative'
            self.waiting_threshold=100
            self.increase_lingvel=True
            self.increase_angvel=True

            print('PPO_HCM')
            self.use_multi_vpred=True

        if(self.curiosity_trajectory_type=='linear'):
            self.linear_beta=1.0/self.curiosity_trajectory_length
        else:
            self.linear_beta=1.0

        ####
        ####
        ####
        ####



        if(True):
            self.title='GENERAL: ' + str(self.uid) + 'robot: ' + self.robot_name + \
            ', steps: ' + str(self.timesteps_per_batch) + \
            ', step size: ' + str(self.vf_stpsize) + \
            ', entcoeff: ' + str(self.entcoeff) + \
            ', schedule: ' + str(self.schedule) + \
            ', o_batch: ' + str(self.optim_batch) + \
            ', o_epoch: ' + str(self.optim_epoch) + \
            ', am: ' + str(self.action_multiplied_by) + \
            ', enh: ' + str(self.enh) + \
            ', ag: ' + str(self.aggressive_maneuver) + \
            ', ep_fly: ' + str(self.epsilon_fly) + \
            ', loading: ' + str(self.loading) + \
            ', nn: ' + str(self.neural_net) + \
            ', wt: ' + str(self.waiting_threshold) + \
            ', RAf: ' + str(self.repeat_action_for) + \
            ', UMV: ' + str(self.use_multi_vpred) + \
            ', vflc: ' + str(self.vf_loss_coef) + \
            ', vfc0lc: ' + str(self.vf_c0_loss_coef) + \
            '\nCURIOUS: ' + str(self.curious) + \
            ', CT: ' + str(self.curiosity_type) + \
            ', CSC: ' + str(self.curiosity_second_coef) + \
            ', db_Size: ' + str(self.Curiosity_database_size) + \
            ', batch_Size: ' + str(self.curiosity_batch_size) + \
            ', CE: ' + str(self.curiosity_epoch) + \
            ', hl_s: ' + str(self.curiosity_hl_s) + \
            ', hl_n: ' + str(self.curiosity_hl_n) + \
            ', n_eps: ' + str(self.use_curiosity_every_n_episodes) + \
            ', dynamics: ' + str(self.Curiosity_dynamics_type) + \
            ', TS: ' + str(self.Curiosity_training_step) + \
            ', coef: ' + str(self.pathak_curiosity_coef) + \
            ', CTL: ' + str(self.curiosity_trajectory_length) + \
            ', CTT: ' + str(self.curiosity_trajectory_type) + \
            ', CTA: ' + str(self.curiosity_trajectory_action) + \
            ', FL_c: ' + str(self.curiosity_FORWARD_LOSS_WT) + \
            ', lt: ' + str(self.curiosity_type_limited_type) + \
            ', TCE: ' + str(self.TRAIN_CURIOSITY_EVERY) + \
            ', RTfC: ' + str(self.reward_type_for_curiosity) + \
            ', URH: ' + str(self.curiosity_use_reward_head) + \
            ', OC: ' + str(self.reward_type_for_curiosity_only_change) + \
            '\nAUX REWARD, VF_stp: ' + str(self.vf_stpsize_c) + \
            ', O_IM: ' + str(self.obstacle_initiation_method) + \
            ', O_N: ' + str(self.number_of_obstacles) + \
            ', C_Hs: ' + str(self.number_of_curiosity_heads) + \
            ', DI: ' + str(self.deterministic_initiation) + \
            ', O_IS: ' + str(self.inner_obstacle_space) + \
            ', O_S: ' + str(self.obstacle_space) + \
            ', O_SDS: ' + str(self.obstacle_space_with_drone_space_distance) + \
            '\nMYD_RC: ' + str(self.yaw_direction_reward_coef) + \
            ', Ilinvel: ' + str(self.increase_lingvel) + \
            ', linvel_RC: ' + str(self.linvel_reward_coef) + \
            ', Iangvel: ' + str(self.increase_angvel) + \
            ', angvel_RC: ' + str(self.ang_vel_reward_coef) + \
            '\nREWARD type: ' + \
            ', GR: ' + str(self.activate_ground_reset) + \
            ', GR_on: ' + str(self.activate_ground_reset_on) + \
            ', n_angle_r_m: ' + str(self.normalized_angle_reward_multiplier) + \
            ', RPT_xy: ' + str(self.reset_pose_thresh_xy) + \
            ', RPT_z: ' + str(self.reset_pose_thresh_z) + \
            ', ng: ' + str(self.near_ground_terminal_height) + \
            ', RFt: ' + str(self.round_float_to) + \
            ', CRFt: ' + str(self.curiosity_round_float_to) + \
            ', CST_epsd: ' + str(self.curiosity_starting_training_episode) + \
            ', CST_epch: ' + str(self.curiosity_starting_training_epoch) + \
            ', l_alpha: ' + str(self.linear_alpha) + \
            ', l_beta: ' + str(self.linear_beta) + \
            ', C_RC_loss: ' + str(self.consider_rewards_change_loss) + \
            ', O: ' + str(self.train_with_obstacle) + \
            ', O_rc: ' + str(self.obstacle_reward_coef) + \
            ', O_nr: ' + str(self.hit_obstacle_negative_reward) + \
            ', MYD: ' + str(self.move_in_yaw_direction) + \
            ', MYD_AE: ' + str(self.yaw_acceptable_error)



            self.title_precise=str(self.uid) + self.robot_name + \
            '_' + str(self.timesteps_per_batch) + \
            '_' + str(self.network_name) + \
            '_C' + str(self.curious) + \
            '_' + str(self.schedule) + \
            '_' + str(self.curiosity_type) + \
            '_O' + str(self.train_with_obstacle) + \
            '_NO' + str(self.number_of_obstacles) + \
            '_' + str(self.action_multiplied_by) + \
            '_' + str(self.state_size) + \
            '_' + str(self.enh) + \
            '_A' + str(self.aggressive_maneuver) + \
            '_RAf' + str(self.repeat_action_for) + \
            '_MYD' + str(self.move_in_yaw_direction) + \
            '_nCH' + str(self.number_of_curiosity_heads) + \
            '_DI' + str(self.deterministic_initiation)





        self.fix_cum_reward = 0
        self.cum_reward_average_counter = 0
        self.cum_reward_average = 0
        self.done_counter=0
        self.cum_reward = 0

        self.episode_total_reward=0.0
        self.episode_counter=0

        self.running_reward = 0
        self.running_reward_mean = 0
        self.done_counter = 0.0

        self.running_reward = 0
        self.running_reward_mean = 0
        self.done_counter = 0.0


        self.fig = plt.figure()
        self.fig.suptitle(self.title, fontsize=10)

        size=10



        self.ext_reward_plot = self.fig.add_subplot(4,2,1)
        self.ext_reward_plot.set_xlabel("episode", fontsize=size)
        self.ext_reward_plot.set_ylabel("External Reward", fontsize=size)
        self.ext_reward_plot.set_title("", fontsize=size)
        self.ext_reward_plot.axis('auto')
        self.ext_reward_plot.grid(which='both')

        self.aux_reward_plot = self.fig.add_subplot(4,2,2)
        self.aux_reward_plot.set_xlabel("episode", fontsize=size)
        self.aux_reward_plot.set_ylabel("Auxiliary Reward", fontsize=size)
        self.aux_reward_plot.set_title("", fontsize=size)
        self.aux_reward_plot.axis('auto')
        self.aux_reward_plot.grid(which='both')

        self.last_100_failed_flight_plot = self.fig.add_subplot(4,2,3)
        self.last_100_failed_flight_plot.set_xlabel("Step", fontsize=size)
        self.last_100_failed_flight_plot.set_ylabel("LAst 100 Failed Fly", fontsize=size)
        self.last_100_failed_flight_plot.set_title("", fontsize=size)
        self.last_100_failed_flight_plot.axis('auto')
        self.last_100_failed_flight_plot.grid(which='both')


        self.failed_flight_plot = self.fig.add_subplot(4,2,4)
        self.failed_flight_plot.set_xlabel("episode", fontsize=size)
        self.failed_flight_plot.set_ylabel("Failed Fly", fontsize=size)
        self.failed_flight_plot.set_title("", fontsize=size)
        self.failed_flight_plot.axis('auto')
        self.failed_flight_plot.grid(which='both')

        self.odom_pos_err_plot = self.fig.add_subplot(4,2,5)
        self.odom_pos_err_plot.set_xlabel("episode", fontsize=size)
        self.odom_pos_err_plot.set_ylabel("Position Error", fontsize=size)
        self.odom_pos_err_plot.set_title("", fontsize=size)
        self.odom_pos_err_plot.axis('auto')
        self.odom_pos_err_plot.grid(which='both')

        self.odom_att_err_plot = self.fig.add_subplot(4,2,6)
        self.odom_att_err_plot.set_xlabel("episode", fontsize=size)
        self.odom_att_err_plot.set_ylabel("Attitude Error", fontsize=size)
        self.odom_att_err_plot.set_title("", fontsize=size)
        self.odom_att_err_plot.axis('auto')
        self.odom_att_err_plot.grid(which='both')

        self.total_obstacle_hit_plot = self.fig.add_subplot(4,2,7)
        self.total_obstacle_hit_plot.set_xlabel("Obstacle Hit", fontsize=size)
        self.total_obstacle_hit_plot.set_ylabel("", fontsize=size)
        self.total_obstacle_hit_plot.set_title("", fontsize=size)
        self.total_obstacle_hit_plot.axis('auto')
        self.total_obstacle_hit_plot.grid(which='both')

        self.obstacle_goal_distance_plot = self.fig.add_subplot(4,2,8)
        self.obstacle_goal_distance_plot.set_xlabel("Obstacle & Goal Distance", fontsize=size)
        self.obstacle_goal_distance_plot.set_ylabel("", fontsize=size)
        self.obstacle_goal_distance_plot.set_title("", fontsize=size)
        self.obstacle_goal_distance_plot.axis('auto')
        self.obstacle_goal_distance_plot.grid(which='both')


        self.fix_cum_reward = 0
        self.cum_reward_average_counter = 0
        self.cum_reward_average = 0
        self.done_counter=0
        self.cum_reward = 0

        self.episode_total_reward=0.0

        self.pdf_file_name='pdf/' + self.title_precise + '.pdf'


        self.loading_path=''
        self.save_path=''


        # time.sleep(5)
    def save_pdf(self):
        with PdfPages(self.pdf_file_name) as pdf:
            pdf.savefig(self.fig)
