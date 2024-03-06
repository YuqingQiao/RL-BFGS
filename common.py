import numpy as np
from envs.config import make_env, make_vector_env, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from algorithm import create_agent
from learner import create_learner, learner_collection
from tester import Tester
from algorithm.replay_buffer import ReplayBuffer_Episodic, goal_based_process
from envs.custom_scenarios import scenarios
import os


def get_args():
    parser = get_arg_parser()

    # General
    parser.add_argument('--tag', help='optional prefix for logging', type=str, default='')
    parser.add_argument('--learn', help='type of training method', type=str, default='normal',
                        choices=learner_collection.keys())
    parser.add_argument('--env', help='gym env id', type=str, default='FrankaDrawerEnv',
                        choices=Robotics_envs_id)
    parser.add_argument('--render', help='whether to render a single rollout after each epoch', type=bool, default=True)
    parser.add_argument('--scenario', help='use a custom scenario for training', type=str, default='', choices=list(scenarios.keys()))
    parser.add_argument('--control_mode', help='Control mode for the robot', type=str, default='ik_controller', choices=['position', 'position_rotation', 'torque', 'ik_controller'])

    args, _ = parser.parse_known_args()

    # Play
    parser.add_argument('--model_path', help='path to model directory of trained agent', type=str, default='/home/qiao/RL-Dyn-Env-main/log/sim-4')
    parser.add_argument('--play_epoch', help='epoch number to play. Defaults to best epoch', type=str, default='best')

    # Train
    # Note: For each epoch the agent collects [cycles * num_envs * timesteps] of data and trains [cycles] times
    #       on [train_batches * batch_size] of collected data from the replay buffer (buffer_size).
    parser.add_argument('--epochs', help='number of epochs', type=np.int32, default=200)
    parser.add_argument('--cycles', help='number of training cycles per epoch.', type=np.int32, default=50)
    parser.add_argument('--num_envs', help='number of async environments. Should be less than cpu kernels to avoid overhead', type=np.int32, default=10)
    parser.add_argument('--train_batches', help='number of batches to train per cycle', type=np.int32, default=50)
    parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)

    # Environment
    parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32, default=80)
    parser.add_argument('--env_n_substeps', help='Steps to simulate', type=np.int32, default=80)
    parser.add_argument('--num_obst', help='starting number of obstacles to be randomized. Max number is fixed in xml file.', type=np.int32, default=1)

    # RL Hyperparameters
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
    parser.add_argument('--pi_lr', help='learning rate of actor network', type=np.float32, default=1e-3)
    parser.add_argument('--q_lr', help='learning rate of critic network', type=np.float32, default=1e-3)
    parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
    parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)

    # RL Architecture
    parser.add_argument('--actor_layer_sizes', help='number of hidden layers and neurons for actor', type=np.float32, default=[512, 256, 128])
    parser.add_argument('--actor_batch_norm', help='apply batch normalization after each hidden layer for actor', type=np.float32, default=[False]*3)
    parser.add_argument('--critic_layer_sizes', help='number of hidden layers and neurons for critic', type=np.float32, default=[512, 256, 128])
    parser.add_argument('--critic_batch_norm', help='apply batch normalization after each hidden layer for critic', type=np.float32, default=[False]*3)

    # Rewards
    parser.add_argument('--obj_lost_reward', help='additional reward for loosing object from gripper', type=np.float32, default=-0.3)
    parser.add_argument('--collision_reward', help='additional reward for collisions', type=np.float32, default=-0.5)

    parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
    parser.add_argument('--reward_min', help='min factor for clip_return', type=np.float32, default=-1.)
    parser.add_argument('--reward_max', help='max factor for clip_return', type=np.float32, default=0.)

    # Exploration
    parser.add_argument('--eps_act', help='starting percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
    parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32,
                        default=0.2)

    # Replay Buffer
    parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=20000)
    parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization',
                        type=str, default='energy', choices=['normal', 'energy'])

    # HER
    parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future',
                        choices=['none', 'final', 'future'])
    parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
    parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full',
                        choices=['full', 'final'])

    args = parser.parse_args()

    args.goal_based = True
    args.cur_acc = 0.

    # temporary env to retrieve experiment params
    env = make_env(args)
    args.obs_dims = list(goal_based_process(env.reset()[0]).shape)
    args.acts_dims = [env.action_space.shape[0]]
    args.compute_reward = env.compute_reward

    if args.env == "Sim2RealEnv-v1":
        env.unwrapped.cam.stop()
    return args


def experiment_setup(args):
    # Create logger
    if args.tag:
        logger_name = args.tag + '-' + args.env
    else:
        logger_name = args.env

    if args.scenario:
        logger_name = logger_name + '-' + args.scenario

    args.logger = get_logger(logger_name)

    # print params
    for key, value in args.__dict__.items():
        if key != 'logger':
            args.logger.info('{}: {}'.format(key, value))

    envs = make_vector_env(args)
    # set custom scenario if given
    if args.scenario:
        envs.set_attr('scenario', [scenarios[args.scenario] for _ in range(args.num_envs)])

    learner = create_learner(args, envs)

    agent = create_agent(args)
    buffer = ReplayBuffer_Episodic(args)

    # load model weights if path is given
    if args.model_path != '':
        agent.load(os.path.join(args.model_path, "model/saved_policy-{}".format(args.play_epoch)))

    args.logger.info('*** agent initialization complete ***')

    tester = Tester(args, envs)
    args.logger.info('*** tester initialization complete ***')

    return agent, buffer, learner, tester
