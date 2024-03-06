import gymnasium as gym

Robotics_envs_id = [
    'RandDynObstEnv-v1',
    'Sim2RealEnv-v1'
    'FrankaDrawerEnv'
    'FrankaDrawerEnv_2'
    'FrankaShelfEnv'
    'FrankaMazeEnv'
    'FrankaObstx2Env'
    # 'Sim2RealEnv-franky'
]


def register_custom_envs():
    gym.envs.register(
        id='RandDynObstEnv-v1',
        entry_point='envs:RandDynObstEnv',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    gym.envs.register(
        id='Sim2RealEnv-v1',
        entry_point='envs:Sim2RealEnv',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    # gym.envs.register(
    #     id='Sim2RealEnv-franky',
    #     entry_point='envs:Sim2RealEnvFranky',
    #     max_episode_steps=500,
    #     kwargs={'render_mode': 'human'}
    # )

    gym.envs.register(
        id='FrankaDrawerEnv-v1',
        entry_point='envs:FrankaDrawerEnv',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    gym.envs.register(
        id='FrankaDrawerEnv_2-v1',
        entry_point='envs:FrankaDrawerEnv_2',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    gym.envs.register(
        id='FrankaObstx2Env-v1',
        entry_point='envs:FrankaObstx2Env',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    gym.envs.register(
        id='FrankaShelfEnv-v1',
        entry_point='envs:FrankaShelfEnv',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )
    gym.envs.register(
        id='FrankaMazeEnv-v1',
        entry_point='envs:FrankaMazeEnv',
        max_episode_steps=500,
        kwargs={'render_mode': 'human'}
    )


def make_env(args):
    return gym.make(
        args.env,
        control_mode=args.control_mode,
        n_substeps=args.env_n_substeps,
        obj_lost_reward=args.obj_lost_reward,
        collision_reward=args.collision_reward,
        num_obst=args.num_obst)

def make_vector_env(args):
    return gym.vector.make(
        args.env,
        control_mode=args.control_mode,
        n_substeps=args.env_n_substeps,
        num_envs=args.num_envs,
        obj_lost_reward=args.obj_lost_reward,
        collision_reward=args.collision_reward,
        num_obst=args.num_obst)

