import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import time
from math import exp

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles a
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

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


class amir_rotors_sensors:
	silent_mode=False
	env_id=0

	print('Starting Sensor Class')

	# odometry variables
	header=0.0

	x_angle=0.0
	y_angle=0.0
	z_angle=0.0

	x_position=0.0
	y_position=0.0
	z_position=0.0

	linear_velocity_x=0.0
	linear_velocity_y=0.0
	linear_velocity_z=0.0

	angular_velocity_x=0.0
	angular_velocity_y=0.0
	angular_velocity_z=0.0

	prev_linear_velocity_x=0.0
	prev_linear_velocity_y=0.0
	prev_linear_velocity_z=0.0

	prev_angular_velocity_x=0.0
	prev_angular_velocity_y=0.0
	prev_angular_velocity_z=0.0

	linear_acceleration_x=0.0
	linear_acceleration_y=0.0
	linear_acceleration_z=0.0

	angular_acceleration_x=0.0
	angular_acceleration_y=0.0
	angular_acceleration_z=0.0

	able_to_move=True
	x_new_position_difference=0.0
	y_new_position_difference=0.0
	z_new_position_difference=0.0

	yaw_new_angle_difference=0.0

	x_angle_raw=0.0
	y_angle_raw=0.0
	z_angle_raw=0.0
	w_angle_raw=0.0

	x_angle_raw_eu = 0.0
	y_angle_raw_eu = 0.0
	z_angle_raw_eu = 0.0

	x_position_raw=0.0
	y_position_raw=0.0
	z_position_raw=0.0

	x_position_actual=0.0
	y_position_actual=0.0
	z_position_actual=0.0


	def __init__(self, env_id, silent_mode, able_to_move, x_new_position_difference, y_new_position_difference, z_new_position_difference, yaw_new_angle_difference, parameters):
		self.parameters=parameters

		self.silent_mode=silent_mode
		self.env_id=env_id

		self.able_to_move=True
		self.x_new_position_difference=x_new_position_difference
		self.y_new_position_difference=y_new_position_difference
		self.z_new_position_difference=z_new_position_difference
		self.yaw_new_angle_difference=yaw_new_angle_difference

	def angle_between_two_vector(self, V1, V2):
		V1V2=np.dot(V1,V2)
		V1_magnitude=np.linalg.norm(V1)
		V2_magnitude=np.linalg.norm(V2)

		angle=math.asin(V1V2/(V1_magnitude*V2_magnitude))

		if(self.x_position_raw>0 and self.y_position_raw>0):
		          angle += -1.5708

		if(self.x_position_raw>0 and self.y_position_raw<0):
		          angle += 1.5708

		return math.degrees(angle), angle # degrees, radian


	def odometry(self, msg):
		self.header=msg.header
		Update=True

		if(Update):
				self.x_angle_raw=msg.pose.pose.orientation.x #data->vx#; //angular velocity X
				self.y_angle_raw=msg.pose.pose.orientation.y #data->vy#; // angular velocity Y
				self.z_angle_raw = msg.pose.pose.orientation.z #+ (0.07) # data->vy#; // angular velocity Y
				self.w_angle_raw = msg.pose.pose.orientation.w

				quaternion=np.array([self.x_angle_raw, self.y_angle_raw, self.z_angle_raw, self.w_angle_raw])
				angles = euler_from_quaternion(quaternion)

				self.x_position_actual=msg.pose.pose.position.x
				self.y_position_actual=msg.pose.pose.position.y
				self.z_position_actual=msg.pose.pose.position.z

				if(self.able_to_move):
						self.x_position_raw=msg.pose.pose.position.x - self.x_new_position_difference
						self.y_position_raw=msg.pose.pose.position.y - self.y_new_position_difference
						self.z_position_raw=msg.pose.pose.position.z - self.z_new_position_difference

						self.x_position_raw_ref=msg.pose.pose.position.x - self.x_new_position_difference
						self.y_position_raw_ref=msg.pose.pose.position.y - self.y_new_position_difference
						self.z_position_raw_ref=msg.pose.pose.position.z - self.z_new_position_difference

						x_error=self.x_position_raw
						y_error=self.y_position_raw
						z_error=self.z_position_raw

						if(self.parameters.move_in_yaw_direction):
						                      if(abs(x_error)>0.5 or abs(y_error)>0.5):
						                                            rotor_point=np.array((x_error,y_error, 0),np.float32)

						                                            A2=np.array([0,0,0])

						                                            A1A2 = A2 - rotor_point #vector A1A2
						                                            # A1A2 = rotor_point - A2 #vector A1A2

						                                            XZ_plane_normal_vector=np.array([0,1,0],np.float32)

						                                            angle_with_XZ=self.angle_between_two_vector(A1A2, XZ_plane_normal_vector)[1]

						                                            self.yaw_new_angle_difference=angle_with_XZ

    						# else:
    						#                       print('x_error: {}'.format(x_error))
    						#                       print('y_error: {}'.format(y_error))

						                      self.z_angle_raw_eu = (angles[2]) - self.yaw_new_angle_difference

    						# if(abs(x_error)>0.5 or abs(y_error)>0.5):
    						#                       print('angle_with_XZ: {}, yaw: {}, self.z_angle_raw_eu: {}'.format(math.degrees(angle_with_XZ), format(math.degrees(angles[2])), math.degrees(self.z_angle_raw_eu)))
						else:
						                      self.z_angle_raw_eu = (angles[2])



				else:
						self.x_position_raw=msg.pose.pose.position.x
						self.y_position_raw=msg.pose.pose.position.y
						self.z_position_raw=msg.pose.pose.position.z

						self.x_position_raw_ref=msg.pose.pose.position.x
						self.y_position_raw_ref=msg.pose.pose.position.y
						self.z_position_raw_ref=msg.pose.pose.position.z

						self.z_angle_raw_eu = (angles[2])


				self.x_angle_raw_eu = (angles[0])
				self.y_angle_raw_eu = (angles[1])
				# self.z_angle_raw_eu = (angles[2])

				self.x_angle_rad = (self.x_angle_raw_eu)
				self.y_angle_rad = (self.y_angle_raw_eu)
				self.z_angle_rad = (self.z_angle_raw_eu)

				self.x_angle_ref = math.degrees(self.x_angle_raw_eu)
				self.y_angle_ref = math.degrees(self.y_angle_raw_eu)
				self.z_angle_ref = math.degrees(self.z_angle_raw_eu)

				self.x_angle = math.degrees(angles[0])
				self.y_angle = math.degrees(angles[1])
				self.z_angle = math.degrees(self.z_angle_raw_eu)

				self.x_position = int(self.x_position_raw * 100.0)
				self.y_position = int(self.y_position_raw * 100.0)
				self.z_position = int(self.z_position_raw * 100.0)

				self.x_position_ref = int(self.x_position_raw_ref * 100.0)
				self.y_position_ref = int(self.y_position_raw_ref * 100.0)
				self.z_position_ref = int(self.z_position_raw_ref * 100.0)

				self.linear_velocity_x = msg.twist.twist.linear.x
				self.linear_velocity_y = msg.twist.twist.linear.y
				self.linear_velocity_z = msg.twist.twist.linear.z

				self.angular_velocity_x = msg.twist.twist.angular.x
				self.angular_velocity_y = msg.twist.twist.angular.y
				self.angular_velocity_z = msg.twist.twist.angular.z

				time_unit=1.0

				self.linear_acceleration_x=(self.linear_velocity_x-self.prev_linear_velocity_x)/time_unit
				self.linear_acceleration_y=(self.linear_velocity_y-self.prev_linear_velocity_y)/time_unit
				self.linear_acceleration_z=(self.linear_velocity_z-self.prev_linear_velocity_z)/time_unit

				self.angular_acceleration_x=(self.angular_velocity_x-self.prev_angular_velocity_x)/time_unit
				self.angular_acceleration_y=(self.angular_velocity_y-self.prev_angular_velocity_y)/time_unit
				self.angular_acceleration_z=(self.angular_velocity_z-self.prev_angular_velocity_z)/time_unit

				self.prev_linear_velocity_x=self.linear_velocity_x
				self.prev_linear_velocity_y=self.linear_velocity_y
				self.prev_linear_velocity_z=self.linear_velocity_z

				self.prev_angular_velocity_x=self.angular_velocity_x
				self.prev_angular_velocity_y=self.angular_velocity_y
				self.prev_angular_velocity_z=self.angular_velocity_z
