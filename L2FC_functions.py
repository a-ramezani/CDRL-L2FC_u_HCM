import numpy as np

from mav_msgs.msg import Actuators

#### bio-inspired controller functions

def rotors_move(action, num, velocity_publisher, active_motors):
    EngineSpeed = Actuators()
    Speeds = np.array([int(action[0]), int(action[1]), int(action[2]), int(action[3])])

    EngineSpeed.angular_velocities = Speeds
    velocity_publisher.publish(EngineSpeed)

    return Speeds

def rotors_stop(velocity_publisher):
    EngineSpeed = Actuators()

    Speed_New_Value = 100
    Speeds = np.array([Speed_New_Value, Speed_New_Value, Speed_New_Value, Speed_New_Value])
    EngineSpeed.angular_velocities = Speeds
    velocity_publisher.publish(EngineSpeed)

    return True

def rotors_stop_engine(velocity_publisher):
    EngineSpeed = Actuators()
    Speeds = np.array([0, 0, 0, 0])

    EngineSpeed.angular_velocities = Speeds
    velocity_publisher.publish(EngineSpeed)

    return Speeds
