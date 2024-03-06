import os
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils import mujoco_utils

from envs.ik_controller_bfgs import IKController #replace the controller here
from envs.utils import eul2quat
from envs.custom_scenarios import scenarios as scene

DEFAULT_CAMERA_CONFIG = {
    "distance": 1,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}

DEFAULT_SIZE = 480


class FrankaObstx2Env(gym.Env, EzPickle):
    """
    ## Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The task in the environment is for a manipulator to move a block to a target position on top of a table or in mid-air. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper.
    The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the graspping operation of pick and place.
    The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning the control to the robot.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper.

    | Num | Action                                                 | Control Min | Control Max | Name (in corresponding XML file)                                | Joint | Unit         |
    | --- | ------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 3   | Positional displacement per timestep of each finger of the gripper  | -1          | 1           | robot0:r_gripper_finger_joint and robot0:l_gripper_finger_joint | hinge | position (m) |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector.
    Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
    | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 3   | Block x position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 4   | Block y position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 5   | Block z position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 6   | Relative block x position with respect to gripper x position in global coordinates. Equals to x<sub>gripper</sub> - x<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 7   | Relative block y position with respect to gripper y position in global coordinates. Equals to y<sub>gripper</sub> - y<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 8   | Relative block z position with respect to gripper z position in global coordinates. Equals to z<sub>gripper</sub> - z<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 9   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
    | 10  | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
    | 11  | Global x rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 12  | Global y rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 13  | Global z rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 14  | Relative block linear velocity in x direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 15  | Relative block linear velocity in y direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 16  | Relative block linear velocity in z direction                                                                                         | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 17  | Block angular velocity along the x axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 18  | Block angular velocity along the y axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 19  | Block angular velocity along the z axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 20  | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 21  | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 22  | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 23  | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
    | 24  | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |

    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Final goal block position in the x coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
    | 1   | Final goal block position in the y coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
    | 2   | Final goal block position in the z coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |

    * `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Current block position in the x coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
    | 1   | Current block position in the y coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
    | 2   | Current block position in the z coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |


    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target position, and `0` if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `FetchPickAndPlace-v2`. However, for `dense` reward the id must be modified to `FetchPickAndPlaceDense-v2` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('FetchPickAndPlaceDense-v2')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at `(x,y,z) = [0.405, 0.48, 0]` in global coordinates.

    The block's position has a fixed height of `(z) = [0.42] m ` (on top of the table). The initial `(x,y)` position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of `[-0.15, 0.15] m`. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than `0.1 m`.
    The initial orientation of the block is the same as for the gripper, `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

    Finally the target position where the robot has to move the block is generated. The target can be in mid-air or over the table. The random target is also generated by adding an offset to the initial grippers position `(x,y)` sampled from a uniform distribution with a range of `[-0.15, 0.15] m`.
    The height of the target is initialized at `(z) = [0.42] m ` and an offset is added to it sampled from another uniform distribution with a range of `[0, 0.45] m`.


    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100)
    ```

    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(
            self,
            n_substeps=20,
            control_mode='position',
            obj_range=0.1,
            target_range=0.1,
            num_obst=3,
            obj_goal_dist_threshold=0.03,
            obj_gripper_dist_threshold=0.02,
            max_vel=0.1,
            obj_lost_reward=-0.2,
            collision_reward=-1.,
            scenario=None,
            **kwargs
    ):
        """Initialize the hand and fetch robot superclass.

         kwargs:
             model_path (string): the path to the mjcf MuJoCo model.
             initial_qpos (np.ndarray): initial position value of the joints in the MuJoCo simulation.
             n_actions (integer): size of the action space.
             n_substeps (integer): number of MuJoCo simulation timesteps per Gymnasium step.
             render_mode (optional string): type of rendering mode, "human" for window rendering and "rgb_array" for offscreen. Defaults to None.
             width (optional integer): width of each rendered frame. Defaults to DEFAULT_SIZE.
             height (optional integer): height of each rendered frame . Defaults to DEFAULT_SIZE.
         """

        self._mujoco = mujoco
        self._utils = mujoco_utils

        if control_mode in ['position', 'position_rotation']:
            self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_obstx2.xml")
        else:
            self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_obstx2_ik.xml")

        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        # total_obst = number of obstacles declared in the xml file. (num_obst defines the number of obstacles to be randomized on the table)
        self.total_obst = 7

        self.n_substeps = n_substeps
        self.control_mode = control_mode
        self.obj_range = obj_range
        self.target_range = target_range
        self.num_obst = num_obst
        self.obj_goal_dist_threshold = obj_goal_dist_threshold
        self.obj_gripper_dist_threshold = obj_gripper_dist_threshold
        self.max_vel = max_vel
        self.obj_lost_reward = obj_lost_reward
        self.collision_reward = collision_reward
        self.scenario = scenario

        self.initial_qpos = {
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],#[1.25, 0.53, 0.4, 1., 0., 0., 0.]
            'robot:joint1': 0.0254,
            'robot:joint2': -0.2188,
            'robot:joint3': -0.0265,
            'robot:joint4': -2.6851,
            'robot:joint5': -0.0092,
            'robot:joint6': 2.4664,
            'robot:joint7': 0.0068,
        }

        self.goal = np.zeros(0)
        self.obstacles = []
        self.reward_sum = 0
        self.col_sum = 0

        if self.control_mode == 'position' or self.control_mode == 'ik_controller':
            self.n_actions = 4
        elif self.control_mode == 'position_rotation':
            self.n_actions = 7
        elif self.control_mode == 'torque':
            self.n_actions = 8
        else:
            raise ValueError("Control mode should be one of the following: ['position', 'position_rotation', 'torque']")

        self._initialize_simulation()

        obs = self._get_obs()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_actions,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64
                ),
                object_gripper_dist=spaces.Box(
                    -np.inf, np.inf, shape=obs["object_gripper_dist"].shape, dtype=np.float64
                ),
                collision=spaces.Discrete(2),
            )
        )

        # IK controller
        self.IKC = IKController(self.get_body_pos("link0"), self.data.qpos[:7], self.get_obstacle_info())

        self.render_mode = 'human'
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG
        )

        EzPickle.__init__(self, **kwargs)

    def _initialize_simulation(self):
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = DEFAULT_SIZE
        self.model.vis.global_.offheight = DEFAULT_SIZE

        for name, value in self.initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_geom_sizes = np.copy(self.model.geom_size)
        self.initial_body_pos = np.copy(self.model.body_pos)
        self.initial_site_size = np.copy(self.model.site_size)

        # Actuator ranges
        ctrlrange = self.model.actuator_ctrlrange
        self.actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        self.actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        # Obstacle IDs (for collision checking)
        self.obstacle_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'obstacle{n}') for n in
                             range(self.total_obst)]

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Typically, due to a timelimit.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
                key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, -1, 1)
        target_pos = []
        q_res = []
        if self.control_mode == 'ik_controller':
            ##########################
            # get obstacle information
            obstacles = self.get_obstacle_info()
            # compute target position
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
            target_pos = (grip_pos + np.clip(action[:3], -1, 1) * 0.04).copy()
            # to avoid hitting the table, we clip the target in z-direciton
            table_edge = (self.get_body_pos('table0') + self.get_geom_size('table0'))[2] + 0.025
            if target_pos[2] < table_edge:
                target_pos[2] = table_edge

            self.set_site_pos("target_pos", target_pos)

            qpos = self.data.qpos[:7].copy()

            # calculate forward kinematics and capsule positions for visualization
            q_res, robot_capsules, obst_capsules = self.IKC.solve(qpos, obstacles, target_pos)

            for i, caps in enumerate(robot_capsules + obst_capsules):
                pos = (caps['u'] + caps['p']) / 2
                size = np.array([caps['roh'], np.linalg.norm((caps['u'] - caps['p']) / 2), 0.005])
                quat = np.empty((4,))
                mujoco.mju_quatZ2Vec(quat, caps['u'] - caps['p'])

                self.set_site_pos("capsule" + str(i + 1), pos)
                self.set_site_size("capsule" + str(i + 1), size)
                self.set_site_quat("capsule" + str(i + 1), quat)

            ######################
            self._set_action(np.append(q_res, action[3]))
        else:
            self._set_action(action)
        self._move_obstacles()

        # self._mujoco.mj_step(self.model, self.data, nstep=1)
        # self.render()

        # forward simu
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        obs = self._get_obs()

        reward = self.compute_reward(obs["achieved_goal"], self.goal, obs['object_gripper_dist'], obs['collision'])
        self.col_sum += obs['collision']
        self.reward_sum += reward

        terminated = False
        truncated = False

        info = {
            "Success": self._is_success(obs["achieved_goal"], self.goal),
            "ExReward": self.reward_sum,
            "Collisions": self.col_sum
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render a frame of the MuJoCo simulation."""
        # visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0")
        self.model.site_pos[site_id] = self.goal - sites_offset[0]

        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def _set_action(self, action):
        # ensure that we don't change the action outside of this scope
        action = action.copy()

        if self.control_mode.startswith('pos'):
            if self.control_mode == 'position':
                pos_ctrl, gripper_ctrl = action[:3], action[3]
                rot_ctrl = [1.0, 0.0, 0.0, 0.0]  # fixed rotation of the end effector, expressed as a quaternion
            else:
                pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:6], action[6]
                # rot_ctrl *= 0.1  # limit maximum rotation
                rot_ctrl = eul2quat(rot_ctrl)

            pos_ctrl *= 0.05  # limit maximum change in position
            gripper_ctrl *= 50  # scale control action
            # normalize gripper ctrl
            ctrl_range = self.model.actuator_ctrlrange
            ctrl =  np.clip(self.data.ctrl[0] + gripper_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
            action = np.concatenate([pos_ctrl, rot_ctrl, ctrl])
            # Apply action to simulation.
            self._utils.mocap_set_action(self.model, self.data, action)
            self.data.ctrl = ctrl
        else:
            action[-1] *= 50  # scale control action
            ctrl_range = self.model.actuator_ctrlrange
            action[-1] = np.clip(self.data.ctrl[-1] + action[-1], ctrl_range[-1, 0], ctrl_range[-1, 1])
            self.data.ctrl = action

    def _move_obstacles(self):
        for obst in self.obstacles:
            margin = self.dt * self.max_vel
            pos = self._utils.get_joint_qpos(self.model, self.data, obst['name'] + ':joint')
            d = obst['direction']
            # this is to ensure that obstacles stay on their path, even if a collision happen
            # self._utils.set_joint_qpos(self.model, self.data, obst['name'] + ':joint', obst['pos'])
            obst['pos'][d] += obst['vel'][d] * self.dt
            # flip direction
            if (pos[d] - obst['size'][d]) <= obst['l_bound'][d] + margin:
                obst['vel'] = abs(obst['vel'])
            elif (pos[d] + obst['size'][d]) >= obst['r_bound'][d] - margin:
                obst['vel'] = -1 * abs(obst['vel'])
            # adjust
            self._utils.set_joint_qvel(self.model, self.data, obst['name'] + ':joint', obst['vel'])

    def _check_collisions(self):
        for i in range(self.data.ncon):

            contact = self.data.contact[i]
            if contact.exclude > 0:
                continue

            for obst_id in self.obstacle_ids:
                # skip table contacts and object
                table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'table0')
                panda_table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'panda_table')
                object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'object0')
                skip = [table_id, panda_table_id, object_id]
                if contact.geom1 not in skip and contact.geom2 not in skip:
                    if (contact.geom1 == obst_id) or (contact.geom2 == obst_id):
                        # print("COL!")
                        return 1
        return 0

    # getter for hgg
    def get_obs(self):
        return self._get_obs()

    def _get_obs(self):
        # robot
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.model, self.data, self._model_names.joint_names)
        gripper_placement = [robot_qpos[-1]]

        # gripper
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")

        # object
        object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "object0")
        object_size = self.get_geom_size("object0")

        # object-gripper (we only need this for the reward)
        object_rel_pos = object_pos - grip_pos
        object_gripper_dist = np.linalg.norm(object_rel_pos.ravel())

        # obstacles
        obstacles = []
        for n in range(self.total_obst):
            pos = self._utils.get_joint_qpos(self.model, self.data, f'obstacle{n}' + ':joint')[:3]
            vel = self.custom_get_joint_qvel(self.model, self.data, f'obstacle{n}' + ':joint')[:3]
            size = self.get_geom_size(f'obstacle{n}')
            obstacles.append(np.concatenate([pos, vel, size]))
        if not self.scenario:
            # randomize order to avoid overfitting to the first obstacle during curriculum learning
            self.np_random.shuffle(obstacles)
        obst_states = np.concatenate(obstacles)

        # achieved goal, essentially the object position
        achieved_goal = np.squeeze(object_pos.copy())

        # collisions
        collision = self._check_collisions()

        obs = np.concatenate(
            [
                # robot_qpos,
                # robot_qvel,
                gripper_placement,
                grip_pos,
                grip_velp,
                object_pos,
                object_size,
                object_velp,
                obst_states
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "object_gripper_dist": object_gripper_dist.copy(),
            "collision": collision,
        }

    def compute_reward(self, achieved_goal, desired_goal, object_gripper_dist, collision):
        # Compute distance between goal and the achieved goal.
        rew = self._is_success(achieved_goal, desired_goal) - 1
        # object lost reward
        if object_gripper_dist > self.obj_gripper_dist_threshold:
            rew += self.obj_lost_reward
        # collisions reward
        if collision:
            rew += self.collision_reward
        return rew

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.obj_goal_dist_threshold).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state. It should satisfy the `GoalEnv` :attr:`observation_space`.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """

        super().reset(seed=seed)

        did_reset_sim = False
        while not did_reset_sim:
            self.obstacles = []
            did_reset_sim = self._reset_sim()

        self.reward_sum = 0
        self.col_sum = 0
        obs = self._get_obs()

        return obs, {}

    def _reset_sim(self):

        # Reset Everything
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self.model.geom_size = np.copy(self.initial_geom_sizes)
        self.model.body_pos = np.copy(self.initial_body_pos)
        self.model.site_size = np.copy(self.initial_site_size)
        if self.model.na != 0:
            self.data.act[:] = None

        # Initialize custom env setup if given, else randomize
        if self.scenario:
            self._init_scenario(self.scenario)
        else:
            # occasionally mix in one of the custom scenarios once max obst number is reached
            # if self.num_obst == self.total_obst and self.np_random.random() < 0.2:
            #     self._init_scenario(scene[self.np_random.choice(list(scene.keys()))])
            # else:
            self._init_random()

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _init_scenario(self, scenario):
        # Object
        obj_init_space = scenario['obj_init_space']
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        object_qpos[:2] = self.np_random.uniform(obj_init_space['min'], obj_init_space['max'], size=2)
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

        # Target
        target_init_space = scenario['target_init_space']
        target = self.np_random.uniform(target_init_space['min'], target_init_space['max'], size=3)
        self.goal = target.copy()

        # Obstacles
        for obst in range(self.total_obst):
            obstacle = scenario[f'obstacle{obst}']

            # randomize starting position in moving direction
            pos = obstacle['pos'].copy()
            d = obstacle['dir']
            if obstacle['site_size'][d] - obstacle['size'][d] > 0.025:
                pos[d] = self.np_random.uniform(
                    obstacle['pos'][d] - obstacle['site_size'][d] + obstacle['size'][d] + 0.025,
                    obstacle['pos'][d] + obstacle['site_size'][d] - obstacle['size'][d] - 0.025
                )

            qpos = np.concatenate([pos, [1, 0, 0, 0]])
            # randomize direction and vel
            d = obstacle['dir']
            vel = np.zeros(6)
            vel[d] = self.np_random.uniform(obstacle['vel']['min'], obstacle['vel']['max'])

            self._utils.set_joint_qpos(self.model, self.data, f"obstacle{obst}:joint", qpos)
            self._utils.set_joint_qvel(self.model, self.data, f"obstacle{obst}:joint", vel)
            self.set_geom_size(f'obstacle{obst}', obstacle['size'])
            self.set_body_pos(f'obstacle{obst}:site', obstacle['site_pos'])
            self.set_site_size(f'obstacle{obst}', obstacle['site_size'])
            self.obstacles.append({
                'name': f'obstacle{obst}',
                'pos': qpos,
                'vel': vel,
                'size': obstacle['size'],
                'direction': d,
                'l_bound': np.subtract(obstacle['site_pos'], obstacle['site_size']),
                'r_bound': np.add(obstacle['site_pos'], obstacle['site_size'])
            })

    def _init_random(self):
        # Get workspace boundaries
        ws_pos = self.get_site_pos('workspace')
        ws_size = self.get_site_size('workspace')
        min_pos = ws_pos - ws_size
        max_pos = ws_pos + ws_size

        paths = []
        # Randomize start position and size of object
        obj_margin = 0.05  # margin from table edge
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        object_qpos[:2] = self.np_random.uniform(
            ws_pos - self.get_wrapper_attr('obj_range') * (ws_size - obj_margin),
            ws_pos + self.get_wrapper_attr('obj_range') * (ws_size - obj_margin),
            size=3
        )[:2]
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)
        self.set_geom_size("object0", np.random.uniform(0.015, 0.025, size=3))
        # ensure that the robot has space to grab the object
        obj_safe_size = self.get_geom_size('object0').copy()
        obj_safe_size += [0.1, 0.12, 0.2]
        paths.append({'pos': object_qpos[:3], 'size': obj_safe_size})

        # Sample goal
        goal = np.empty(3)
        goal[:2] = self.np_random.uniform(
            ws_pos - self.get_wrapper_attr('target_range') * ws_size,
            ws_pos + self.get_wrapper_attr('target_range') * ws_size,
            size=3
        )[:2]
        # for the target height we use sqrt, so that the robot has to learn to pick up quicker
        goal[2] = object_qpos[2] + 0.02 + self.np_random.uniform(0, np.sqrt(self.get_wrapper_attr('target_range')) * ws_size[2] * 2, size=1)
        self.goal = goal.copy()
        # ensure that the robot has space to reach goal
        target_safe_size = self.get_geom_size('target0').copy()
        target_safe_size += [0.1, 0.12, 0.2]
        paths.append({'pos': goal, 'size': target_safe_size})

        self._randomize_obstacles(max_pos, min_pos, paths)

    def _randomize_obstacles(self, max_pos, min_pos, paths):
        # obstacle size range
        min_size = np.array([0.01, 0.01, 0.01])
        max_size = np.array([0.06, 0.06, 0.06])
        for obst in range(self.num_obst):
            pos, size, vel = None, None, None
            site_pos, site_size, d = None, None, None
            new_path = {}
            overlap = True
            while overlap:
                obst_margin = self.dt * self.max_vel + 0.02
                # sample obstacle size
                size = np.array([np.random.uniform(m, x) for m, x in zip(min_size, max_size)])
                # limit maximum volume of the obstacle
                if np.prod(size) > 0.0002:
                    continue
                # calculate pos according to size
                pos = np.concatenate(
                    [self.np_random.uniform(min_pos + size + obst_margin, max_pos - size - obst_margin), [1, 0, 0, 0]])
                # generate a path and set the site
                d = self.np_random.choice(range(3))
                site_size, site_pos = size.copy(), pos[:3].copy()
                site_size[d] = (max_pos[d] - min_pos[d]) / 2
                site_pos[d] = (max_pos[d] + min_pos[d]) / 2
                # make sure that no path is colliding with each other
                new_path = {'pos': site_pos, 'size': site_size}
                overlap = self._check_overlap(paths, new_path)
                # random velocity in d direction
                vel = np.zeros(6)
                vel[d] = self.np_random.uniform(-self.max_vel, self.max_vel)
                # give object a 30% chance to be static of same size of table
                if self.np_random.uniform() <= 0.3:
                    vel = np.zeros(6)
                    pos[d] = site_pos[d]
                    size[d] = site_size[d]
            self._utils.set_joint_qpos(self.model, self.data, f"obstacle{obst}:joint", pos)
            self._utils.set_joint_qvel(self.model, self.data, f"obstacle{obst}:joint", vel)
            self.set_geom_size(f'obstacle{obst}', size)
            self.set_body_pos(f'obstacle{obst}:site', site_pos)
            self.set_site_size(f'obstacle{obst}', site_size)
            self.obstacles.append({
                'name': f'obstacle{obst}',
                'pos': pos,
                'vel': vel,
                'size': size,
                'direction': d,
                'l_bound': site_pos - site_size,
                'r_bound': site_pos + site_size
            })
            paths.append(new_path)

    @staticmethod
    def _check_overlap(paths, new_path):
        # min margin between paths
        margin = np.array([0.005, 0.005, 0.005])
        # Make sure that none of the geoms qpos are penetrating each other
        new_min_r = new_path['pos'] - new_path['size']
        new_max_r = new_path['pos'] + new_path['size']

        for path in paths:
            min_r = path['pos'] - path['size'] - margin
            max_r = path['pos'] + path['size'] + margin

            if all(np.logical_and((new_max_r > min_r), (max_r > new_min_r))):
                return True
        return False

    # --- Utils ---
    def get_body_pos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_pos[body_id]

    def set_body_pos(self, name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_pos[body_id] = pos

    def get_geom_size(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_size[geom_id]

    def get_obstacle_info(self):
        obstacles = []
        for n in range(self.total_obst):
            obstacles.append({
                'pos': self._utils.get_joint_qpos(self.model, self.data, f'obstacle{n}' + ':joint')[:3],
                'size': self.get_geom_size(f'obstacle{n}')
            })
        return obstacles

    def set_geom_size(self, name, size):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_size[geom_id] = size
        # adjust rbound (radius of bounding sphere for which collisions are not checked)
        self.model.geom_rbound[geom_id] = np.sqrt(np.sum(np.square(self.model.geom_size[geom_id])))

    def get_site_size(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_size[site_id]

    def set_site_size(self, name, size):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_size[site_id] = size

    def get_site_pos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.model.site_pos[site_id]

    def set_site_pos(self, name, pos):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_pos[site_id] = pos

    def set_site_quat(self, name, quat):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.model.site_quat[site_id] = quat

    # Small bug in gymnasium robotics returns wrong values, use local implementation for now.
    @staticmethod
    def custom_get_joint_qvel(model, data, name):
        """Return the joints linear and angular velocities (qvel) of the model."""
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_type = model.jnt_type[joint_id]
        joint_addr = model.jnt_dofadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim

        return data.qvel[start_idx:end_idx]

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps
