import numpy as np
from franky import (
    Affine,
    Robot,
    LinearMotion,
    Gripper,
    Reaction,
    Measure,
    JointPositionMotion,
    Motion,
    JointWaypointMotion,
    JointWaypoint
)


class FrankaRobot:
    def __init__(self, id = "192.168.178.12"):
        self.robot = Robot(id)
        self.gripper = Gripper(id)

        # Robot base position
        self.robot_base = [0.8, 0.75, 0.4]

        # Default values for gripper
        self.gripper.gripper_speed = 0.08   # [m/s]
        self.gripper.gripper_force = 10.0    # [N]

        # constrain to a percentage of maximum velocity, acceleration and jerk
        self.robot.relative_dynamics_factor = 0.4
        # self.robot.velocity_rel = 0.2
        # self.robot.acceleration_rel = 0.1
        # self.robot.jerk_rel = 0.01

        # recover from errors TODO: why we do this here ?
        # self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.open_count = 0
        self.close_count = 0
        self.open_token = 0
        self.close_token = 0
        self.step_count = 0
        self.total_step = 0
        self.gripper_token = 0

    # def move_3d(self, pos):
        # safety constraint, if overall Force exceeds the value, the robot stops automatically
        # safety = MotionData().with_reaction(Reaction(Measure.ForceXYZNorm() > 10.0))
        # # move to cartesian coordinates # TODO: (from robot base?)
        # self.robot.move(LinearMotion(Affine(pos)), safety)
        #
        # if safety.has_fired:
        #     self.robot.recover_from_errors()
        #     print("Overall Force exceeded 10N!")

    def move_q_async(self, q):

        q_ = q[:7].copy() + np.array([0., 0., 0., 0., 0., 0., np.pi/4])
        self.robot.move(JointWaypointMotion([JointWaypoint(q_)]), asynchronous=True)


        # Last one is gripper control
        # TODO: The RL agent proposes actions in (-1, 1), where -1 is a closed and 1 a fully opened gripper.
        # TODO: In Mujoco, we map this to a control input in (0, 255), here it should be (0, 0.08)
        w = ((q[-1] + 1.0) / 2) * 0.079
        # Make sure the robot don't apply too much action on the cube
        w_ = w
        if w > 0.036:
            w = 0.079
            # w_ = 0.079
        if w <= 0.036:
            w = 0.051
            w_ = 0.051
        if w_ >= 0.1:
            w_ = 0.079
        self.total_step = self.total_step + 1
        # mannally force the gripper to open or close
        if w == 0.079 and self.open_token == 0:
            self.close_token = 1
            self.open_token = 1
        if self.close_token == 1:
            self.step_count = self.step_count + 1
            w = 0.079 # fully open
            if self.open_count == 0:
                if self.gripper_token == 0:
                    self.gripper.move_async(w)
                    self.gripper_token = 1
                self.open_count = self.open_count + 1
            if self.step_count >= 35 and self.step_count < 100:

                w = 0.051 #size of the object
                if self.close_count == 0:
                    if self.gripper_token == 1:
                        self.gripper.move_async(w)
                        self.gripper_token = 2
                    self.open_count = self.open_count + 1
        if self.step_count >= 100:

            self.close_token = 0
            if w_ == 0.079:
                self.open_token = 2
            if self.open_token == 2:
                w_ = 0.079
            if self.gripper_token == 2 and w_ == 0.079:
                self.gripper.move_async(w_)
                self.gripper_token = 3
        print(self.total_step, self.step_count, self.close_token, self.open_token, w, w_)


    def move_q(self, q):

        # Somehow the 7. joint from the real robot is rotated by pi/4 compared to the simulation, so we always add
        # this value here
        q_ = q[:7].copy() + np.array([0., 0., 0., 0., 0., 0., np.pi/4])

        # First 7 numbers are the joint angels,
        self.robot.move(JointWaypointMotion([JointWaypoint(q_)]))

        # Last one is gripper control
        # TODO: The RL agent proposes actions in (-1, 1), where -1 is a closed and 1 a fully opened gripper.
        # TODO: In Mujoco, we map this to a control input in (0, 255), here it should be (0, 0.08)
        w = ((q[-1] + 1.0) / 2) * 0.079
        # Make sure the robot don't apply too much action on the cube
        if w > 0.036:
            w = 0.079
        if w <= 0.036:
            w = 0.051
        self.gripper.move(w)

    def get_current_q(self):
        state = self.robot.state
        # TODO: The RL agent expects 2 values for each finger, but they should always be equal.
        # TODO: They should lie in range (0.004  -self.open_token == 0  0.036) for when the gripper is completely closed or open respectively
        # TODO: As per documentation, the width parameter has 0 for fully closed and 0.08 for fully open, so we map it
        w = (self.gripper.width / 0.08) * 0.032 + 0.004
        return np.append(state.q, [w]*2)

    def get_current_q_dq(self):
        state = self.robot.state
        w = (self.gripper.width / 0.08) * 0.032 + 0.004
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

