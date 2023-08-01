
class amir_rotors_bioinspired_rewards:
	# global ros_sleep_time
	# prev_action=np.zeros((4),np.float32)
	# prev_prev_action=np.zeros((4),np.float32)

	def __init__(self, parameters):

		self.reward=0.0
		self.distanceLeft=self.distanceRight=self.distanceCenter=500.0

		self.step_counter=1

		self.parameters=parameters


	def rotors_reward(self, action_c, z_position, x_position, y_position, linear_velocity_z, x_angle, y_angle, z_angle, angular_velocity_x, angular_velocity_y,desired_x_position, desired_y_position, desired_z_position, active_env):
		reward=0.0

		silent_mode=True

		position_z_difference = abs(z_position - desired_z_position)
		position_x_difference = abs(x_position - desired_x_position)
		position_y_difference = abs(y_position - desired_y_position)

		positive_reward=True
		position_reward=0.0
		normalized_position_reward=0.0
		if(positive_reward):

		xy_max_old=500.0
		xy_max=1000.0

		z_max_old=200.0
		z_max=500.0

		t_diff_z=position_z_difference/z_max
		t_diff_x=position_x_difference/xy_max
		t_diff_y=position_y_difference/xy_max

		t_diff_z=(1.0-t_diff_z)
		t_diff_x=(1.0-t_diff_x)
		t_diff_y=(1.0-t_diff_y)

		t_diff_z=t_diff_z * (10.0)
		t_diff_x=t_diff_x * (10.0)
		t_diff_y=t_diff_y * (10.0)

		if(abs(position_z_difference)>z_max_old):
			t_diff_z=-1.0

		if(abs(position_x_difference)>xy_max_old):
			t_diff_x=-1.0

		if(abs(position_y_difference)>xy_max_old):
			t_diff_y=-1.0

		t_diff=t_diff_z + t_diff_x + t_diff_y

		normalized_position_reward=t_diff/3.0

		normalized_position_z_reward=normalized_position_x_reward=normalized_position_y_reward=0

		normalized_position_reward=normalized_position_reward*self.parameters.coef_ext

		normalized_torcs_reward=0.0

		total_off_angle=(abs(x_angle*self.parameters.angle_x_multiplier) + abs(y_angle*self.parameters.angle_y_multiplier))# + abs(z_angle))

		total_off_angle += (abs(z_angle)/self.parameters.angle_z_multiplier)
		zed=(abs(z_angle)/self.parameters.angle_z_multiplier)

		normalized_angle_reward = (total_off_angle * -1.0) * self.parameters.normalized_angle_reward_multiplier

		normalized_total_reward = (normalized_position_reward + normalized_angle_reward) #if action size ==4

		normalization_reward=0.0

		action_negative_reward=0.0

		f_reward=(normalized_total_reward + action_negative_reward)

		return f_reward
