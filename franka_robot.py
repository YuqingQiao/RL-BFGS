import numpy as np
from frankx import (
    Affine,
    Robot,
    JointMotion,
    LinearMotion,
    MotionData,
    Reaction,
    Measure,
    WaypointMotion,
    Waypoint
)


class FrankaRobot:
    def __init__(self, id = "192.168.178.12"):
        self.robot = Robot(id)
        self.gripper = self.robot.get_gripper()

        # Robot base position
        self.robot_base = [0.8, 0.75, 0.4]

        # Default values for gripper
        self.gripper.gripper_speed = 0.05    # [m/s]
        self.gripper.gripper_force = 20.0    # [N]

        # constrain to a percentage of maximum velocity, acceleration and jerk
        self.robot.velocity_rel = 0.5
        self.robot.acceleration_rel = 0.5
        self.robot.jerk_rel = 0.02

        # recover from errors TODO: why we do this here ?
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()

    def move_3d(self, pos):
        # safety constraint, if overall Force exceeds the value, the robot stops automatically
        safety = MotionData().with_reaction(Reaction(Measure.ForceXYZNorm() > 10.0))
        # move to cartesian coordinates # TODO: (from robot base?)
        self.robot.move(LinearMotion(Affine(pos)), safety)

        if safety.has_fired:
            self.robot.recover_from_errors()
            print("Overall Force exceeded 10N!")

    def move_q_async(self, q):
        # safety constraint, if overall Force exceeds the value, the robot stops automatically
        # safety_robot = MotionData().with_reaction(Reaction(Measure.ForceXYZNorm() > 10.0))
        # safety_gripper = MotionData().with_reaction(Reaction(Measure.ForceXYZNorm() > 10.0))

        # Somehow the 7. joint from the real robot is rotated by pi/4 compared to the simulation, so we always add
        # this value here
        q_ = q[:7].copy() + np.array([0., 0., 0., 0., 0., 0., np.pi/4])

        # First 7 numbers are the joint angels,
        thread_robot = self.robot.move_async(JointMotion(q_))#, safety_robot)

        # if safety_robot.has_fired:
        #     self.robot.recover_from_errors()
        #     print("Overall Force exceeded 10N!")

        # Last one is gripper control
        # TODO: The RL agent proposes actions in (-1, 1), where -1 is a closed and 1 a fully opened gripper.
        # TODO: In Mujoco, we map this to a control input in (0, 255), here it should be (0, 0.08)
        w = ((q[-1] + 1.0) / 2) * 0.079
        # Make sure the robot don't apply too much action on the cube
        if w > 0.04:
            w = 0.079
        if w <= 0.04:
            w = 0.052
        thread_gripper = self.gripper.move_async(w)#, safety_gripper)
        thread_robot.join()
        thread_gripper.join()

        # if safety_gripper.has_fired:
        #     self.robot.recover_from_errors()
        #     print("Overall Force exceeded 10N!")

    def move_q(self, q):

        # Somehow the 7. joint from the real robot is rotated by pi/4 compared to the simulation, so we always add
        # this value here
        q_ = q[:7].copy() + np.array([0., 0., 0., 0., 0., 0., np.pi/4])

        # First 7 numbers are the joint angels,
        self.robot.move(JointMotion(q_))#, safety_robot)


        # Last one is gripper control
        # TODO: The RL agent proposes actions in (-1, 1), where -1 is a closed and 1 a fully opened gripper.
        # TODO: In Mujoco, we map this to a control input in (0, 255), here it should be (0, 0.08)
        w = ((q[-1] + 1.0) / 2) * 0.079
        # Make sure the robot don't apply too much action on the cube
        if w > 0.036:
            w = 0.079
        if w <= 0.036:
            w = 0.051
        self.gripper.move(w)#, safety_gripper)

    def get_current_pose(self):
        return self.robot.current_pose().vector()

    def get_current_q(self):
        state = self.robot.get_state()
        # TODO: The RL agent expects 2 values for each finger, but they should always be equal.
        # TODO: They should lie in range (0.004  -  0.036) for when the gripper is completely closed or open respectively
        # TODO: As per documentation, the width parameter has 0 for fully closed and 0.08 for fully open, so we map it
        w = (self.gripper.width() / 0.08) * 0.032 + 0.004
        return np.append(state.q, [w]*2)

    def get_current_q_dq(self):
        state = self.robot.get_state()
        w = (self.gripper.width() / 0.08) * 0.032 + 0.004
        q = np.append(state.q, [w]*2)
        dq = np.append(state.dq, [0., 0.])

        return q, dq

    @staticmethod
    def map_width(value):
        old_min = 0.0
        old_max = 0.08
        new_min = 0.004
        new_max = 0.036

        return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

