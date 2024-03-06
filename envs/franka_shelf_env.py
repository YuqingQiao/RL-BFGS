import os
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils import mujoco_utils

from envs.ik_controller_bfgs_shelf import IKController
from envs.utils import eul2quat
from envs.custom_scenarios import scenarios as scene

DEFAULT_CAMERA_CONFIG = {
    "distance": 1,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}

DEFAULT_SIZE = 480


class FrankaShelfEnv(gym.Env, EzPickle):

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
            control_mode='position_rotation',
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
            self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_shelf.xml")
        else:
            self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_shelf_ik.xml")

        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        # total_obst = number of obstacles declared in the xml file. (num_obst defines the number of obstacles to be randomized on the table)
        self.total_obst = 8

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
            self.n_actions = 7
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
            # euler angle
            target_rot = [action[3], action[4], action[5]]

            # to avoid hitting the table, we clip the target in z-direciton
            table_edge = (self.get_body_pos('table0') + self.get_geom_size('table0'))[2] + 0.025
            if target_pos[2] < table_edge:
                target_pos[2] = table_edge

            self.set_site_pos("target_pos", target_pos)

            qpos = self.data.qpos[:7].copy()

            # calculate forward kinematics and capsule positions for visualization
            q_res, robot_capsules, obst_capsules = self.IKC.solve(qpos, obstacles, target_pos, target_rot)

            for i, caps in enumerate(robot_capsules + obst_capsules):
                pos = (caps['u'] + caps['p']) / 2
                size = np.array([caps['roh'], np.linalg.norm((caps['u'] - caps['p']) / 2), 0.005])
                quat = np.empty((4,))
                mujoco.mju_quatZ2Vec(quat, caps['u'] - caps['p'])

                self.set_site_pos("capsule" + str(i + 1), pos)
                self.set_site_size("capsule" + str(i + 1), size)
                self.set_site_quat("capsule" + str(i + 1), quat)

            ######################
            self._set_action(np.append(q_res, action[6]))
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
                # obst0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle0')
                # obst1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle1')
                # obst2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle2')
                # obst3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle3')
                # obst4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle4')
                obst5 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle5')
                # obst6 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle6')
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
# target will spawn in either block of the shelf
#         if int(object_qpos[0] * 100)%2==0:
        target = self.np_random.uniform(target_init_space['min'], target_init_space['max'], size=3)
#         else:
#         target = self.np_random.uniform(np.array(target_init_space['min'])+np.array([0, 0, 0.22]), np.array(target_init_space['max'])+np.array([0, 0, 0.22]), size=3)
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
        self.set_site_pos('obst2_visual_1', self.obstacles[2]['pos'][:3])

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
