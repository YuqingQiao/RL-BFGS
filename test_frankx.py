from argparse import ArgumentParser
import time

from franky import Robot, JointWaypointMotion, JointWaypoint, LinearMotion, Affine, ReferenceType


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="192.168.178.12", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.1
    # robot.velocity_rel = 0.002
    # robot.acceleration_rel = 0.01
    # robot.jerk_rel = 0.001

    m1 = JointWaypointMotion([JointWaypoint([-1.8, 1.1, 1.7, -2.1, -1.1, 1.6, -0.4])])
    robot.move(m1, asynchronous=True)

    time.sleep(0.5)
    m1 = JointWaypointMotion([JointWaypoint([1.8, -1.1, -1.7, 2.1, 1.1, -1.6, 0.4])])
    robot.move(m1, asynchronous=True)

    # Wait for the robot to finish its motion
    robot.join_motion()


