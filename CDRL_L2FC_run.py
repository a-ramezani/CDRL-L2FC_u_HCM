
# MIT License
import tensorflow as tf
import rospy
import numpy as np
import argparse
import matplotlib.pyplot as plt

import random

from PIL import Image
import img2pdf
import os
import sys
import time

import logging

from collections import deque
import pickle

import baselines.common.tf_util as U
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict

from baselines.common import set_global_seeds

import os.path as osp

from L2FC_env import L2FC_env

import cv2

def main(args):

    algo_name=args[1]

    environment=L2FC_env(algo_name, silent_mode=True)

    from CDRL_L2FC_policy import CDRL_policy

    sess = U.single_threaded_session()
    sess.__enter__()

    seed=1

    set_global_seeds(1001)

    if(environment.parameters.deterministic_initiation):
        np.random.seed(1)
        random.seed(1)
        tf.set_random_seed(1)

    a_action_space_low_level=Box(-1.0, 1.0, shape=(4,), dtype=np.float32,seed=1)
    a_observation_space_low_level=Box(-1.0, 1.0, shape=(environment.parameters.state_size,), dtype=np.float32,seed=1)
    policy_low_level=CDRL_policy(a_action_space_low_level,a_observation_space_low_level,environment.parameters,policy_number=str(0),policy_name='0',number_of_output_actions=4)
    U.initialize()

    if(environment.parameters.loading):
        U.load_state(environment.parameters.filename_str + '.save', sess=U.get_session())
        print('model loaded')

    policy_low_level.adam.sync()

    is_stochastic=True
    t = 0
    horizon = environment.parameters.timesteps_per_batch
    rew = 0.0

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    new=0
    portion=1.0
    vpred_c1=0

    ob = environment.rotors_bc_reset('start')

    while True:

        if t > 0 and t % horizon == 0:

            ob=environment.rotors_bc_reset(type='horizon')

            print('Model reset.')
            if(environment.UN_COUNTER>=environment.parameters.curiosity_starting_training_episode):
                environment.draw_diagrams()

            environment.total_obstacle_hit=0
            environment.total_obstacle_distance=0
            environment.total_goal_distance=0
            environment.total_yaw_direction_reward=0

            environment.accumulated_predicted_pos_error=0.0
            environment.accumulated_predicted_att_error=0.0
            environment.accumulated_predicted_linvel_error=0.0
            environment.accumulated_predicted_angacc_error=0.0

            environment.total_finishes=0
            environment.total_dones=0

            environment.accumulated_reach_time=0.0
            environment.number_of_reach_per_episode=0.0
            environment.reach_time=0.0
            environment.current_time_step=0

            environment.accumulated_x_error=0.0
            environment.accumulated_y_error=0.0
            environment.accumulated_z_error=0.0

            environment.accumulated_distance_error=0.0

            environment.accumulated_xa_error=0.0
            environment.accumulated_ya_error=0.0
            environment.accumulated_za_error=0.0

            environment.total_accumulated_diff_reward=0.0

            environment.episode_accumulated_reward_org=0.0

            environment.episode_accumulated_negative_action_reward_org=0.0
            environment.episode_accumulated_negative_acceleration_reward_org=0.0
            environment.episode_accumulated_negative_point_to_line_reward_org=0.0
            environment.episode_accumulated_obstacle_distance_reward=0.0
            environment.episode_accumulated_move_in_yaw_direction_reward=0.0

            environment.total_ang_vel_reward=0.0
            environment.total_linvel_reward=0.0
            environment.episode_accumulated_linvel_mag_reward=0.0
            environment.episode_accumulated_ang_vel_mag_reward=0.0

            draw_now=False

            if(environment.parameters.curious):
                environment.train_and_update_with_curiosity()
                if(environment.UN_COUNTER>=environment.parameters.curiosity_starting_training_episode):
                    draw_now=True

            else:
                draw_now=True

            if(environment.legend_drawn==False and draw_now):
                environment.parameters.aux_reward_plot.legend(loc='upper left')
                environment.parameters.ext_reward_plot.legend(loc='upper left')
                environment.parameters.odom_pos_err_plot.legend(loc='upper left')

                environment.parameters.odom_att_err_plot.legend(loc='upper left')

                environment.parameters.total_obstacle_hit_plot.legend(loc='lower left', fontsize=8)
                environment.parameters.obstacle_goal_distance_plot.legend(loc='lower left', fontsize=8)

                environment.parameters.last_100_failed_flight_plot.legend(loc='lower left')

                environment.legend_drawn=True

            plt.pause(0.01)

            ####
            ####
            ####

            max_heatmap=np.max(environment.heatmap_mat)
            min_heatmap=np.min(environment.heatmap_mat)

            normed_heatmap_mat = ((environment.heatmap_mat - min_heatmap) / (max_heatmap - min_heatmap))

            max_heatmap=np.max(normed_heatmap_mat)
            min_heatmap=np.min(normed_heatmap_mat)

            environment.heatmap_image[:,:,0] = 0.0#int(255.0)
            environment.heatmap_image[:,:,1] = 0.0
            environment.heatmap_image[:,:,2] = normed_heatmap_mat # (normed_heatmap_mat * 255.0)

            image_file_name='image/' + environment.parameters.title_precise + '_100_' + str(environment.parameters.episode_counter) + '.png'

            if(environment.parameters.episode_counter % 50 == 0):
                cv2.imwrite(image_file_name, environment.heatmap_image * 255.0)

                normed_heatmap_mat_loaded = cv2.imread(image_file_name, 0)

                max_heatmap=int(np.max(normed_heatmap_mat_loaded))
                min_heatmap=int(np.min(normed_heatmap_mat_loaded))

                normed_heatmap_mat_loaded = ((normed_heatmap_mat_loaded - min_heatmap) / (max_heatmap - min_heatmap)) * 255
                normed_heatmap_mat_loaded = np.array(normed_heatmap_mat_loaded, np.uint8)
                rgb_heatmap_opencv = cv2.applyColorMap(normed_heatmap_mat_loaded, cv2.COLORMAP_JET)
                image_file_name='image/' + environment.parameters.title_precise + '_H_100_' + str(environment.parameters.episode_counter) + '.png'
                cv2.imwrite(image_file_name, rgb_heatmap_opencv)

                img_path = image_file_name
                pdf_path='image/' + environment.parameters.title_precise + '_H_100_' + str(environment.parameters.episode_counter) + '.pdf'
                image = Image.open(img_path)
                pdf_bytes = img2pdf.convert(image.filename)
                file = open(pdf_path, "wb")
                file.write(pdf_bytes)
                image.close()
                file.close()

            ####
            ####
            ####

            environment.parameters.episode_counter+=1

            if(environment.parameters.norm_reward):
                environment.rews=environment.rews / 10.0

            if(environment.parameters.use_multi_vpred):
                seg = {"ob" : environment.obs, "rew" : environment.rews, "rew_c0" : environment.rews_c0, "rew_c1" : environment.rews_c1, "vpred" : environment.vpreds, "vpred_c0" : environment.vpreds_c0, "vpred_c1" : environment.vpreds_c1, "new" : environment.news,
                        "ac" : environment.acs, "prevac" : environment.prevacs, "nextvpred": vpred * (1 - new), "nextvpred_c0": vpred_c0 * (1 - new), "nextvpred_c1": vpred_c1 * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens, "positions" : environment.positions, "angles" : environment.angles, "tbts" : environment.tbtss, "portions" : environment.portions}

            else:
                seg = {"ob" : environment.obs, "rew" : environment.rews, "vpred" : environment.vpreds, "new" : environment.news,
                        "ac" : environment.acs, "prevac" : environment.prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens, "positions" : environment.positions, "angles" : environment.angles, "tbts" : environment.tbtss, "portions" : environment.portions}

            if(environment.UN_COUNTER>=environment.parameters.curiosity_starting_training_episode):
                policy_low_level.train(seg,t)

            ep_rets = []
            ep_lens = []

            time.sleep(1)
            environment.rews_c0[:] = 0.0
            environment.rews_c1[:] = 0.0


            print('Resuming the next episode.')

        prevac = environment.ac
        start_time = time.time()

        if(environment.parameters.use_multi_vpred):
            ac, vpred, vpred_c0 = policy_low_level.pi.act(is_stochastic, ob, portion)

        else:
            ac, vpred = policy_low_level.pi.act(is_stochastic, ob, portion)

        ac=np.round(ac,decimals = 2)

        if(environment.parameters.clip_action):
            ac = np.clip(ac, -1.0, +1.0)

        i = t % horizon
        environment.policy_i=i

        environment.obs[i] = ob
        environment.vpreds[i] = vpred
        if(environment.parameters.use_multi_vpred):

            environment.vpreds_c0[i] = vpred_c0

        environment.news[i] = new

        start_time = time.time()

        action_repeated_for=0

        for RAf in range(environment.parameters.repeat_action_for):
            all_ob, all_rew, all_new, position, angle, tbts, portion, rewards_sparse, rewards_negative= environment.rotors_bc_continuous_step(np.array([np.copy(ac)]))
            action_repeated_for+=1

        environment.acs[i] = ac
        environment.prevacs[i] = prevac

        ob=all_ob[0]

        rew=all_rew[0]
        new=all_new[0]

        environment.portions[i][0]=portion
        environment.portions[i][1]=portion
        environment.portions[i][2]=portion
        environment.portions[i][3]=portion

        environment.positions[i]=position

        environment.angles[i]=angle
        environment.tbtss[i]=tbts

        environment.rews[i] = rew

        environment.rews_sparse[i] = rewards_sparse[0]
        environment.rews_negative[i] = rewards_negative[0]

        t += 1

if __name__ == '__main__':
    main(sys.argv)
