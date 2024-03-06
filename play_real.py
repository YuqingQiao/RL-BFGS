import os
import time
import numpy as np

from algorithm import create_agent
from common import get_args
from envs.config import register_custom_envs, make_env


class Player:
    def __init__(self, args):
        # initialize environment
        self.args = args

        self.info = []
        self.test_rollouts = 1

        self.agent = create_agent(args)
        self.agent.load(os.path.join(args.model_path, "model/saved_policy-{}".format(args.play_epoch)))

    def play(self):
        # play policy on env
        env = make_env(self.args)

        acc_sum = 0
        col_sum = 0
        agent_time = []
        ik_time = []
        comm_time = []
        obs_time = []
        move_time = []
        res = {}

        for i in range(self.test_rollouts):

            obs, _ = env.reset()

            for timestep in range(1500):
                if timestep > 0:
                    st = time.perf_counter()
                    action = self.agent.step(obs)
                    t = time.perf_counter() - st
                    agent_time.append(t)

                    obs, _, _, _, info = env.step(action)
                    t = env.comm_time
                    comm_time.append(t)
                    t = env.ik_time
                    ik_time.append(t)
                    t = env.obs_time
                    obs_time.append(t)
                    # Move robot if everything looks fine:
                    env.unwrapped.robot.robot.recover_from_errors()
                    st = time.perf_counter()
                    # time.sleep(0.1)
                    env.unwrapped.robot.move_q_async(info['action'])
                    t = time.perf_counter() - st
                    move_time.append(t)

                    res['mean_agent'] = round(np.mean(agent_time), 4)
                    res['max_agent'] = round(np.max(agent_time), 4)
                    res['mean_ik'] = round(np.mean(ik_time), 4)
                    res['max_ik'] = round(np.max(ik_time), 4)
                    res['mean_comm'] = round(np.mean(comm_time), 4)
                    res['max_comm'] = round(np.max(comm_time), 4)
                    res['mean_obs'] = round(np.mean(obs_time), 4)
                    res['max_obs'] = round(np.max(obs_time), 4)
                    res['mean_move'] = round(np.mean(move_time), 4)
                    res['max_move'] = round(np.max(move_time), 4)
                    print(res)

                else:

                    action = self.agent.step(obs)

                    obs, _, _, _, info = env.step(action)

                    # Move robot if everything looks fine:
                    env.unwrapped.robot.robot.recover_from_errors()

                    env.unwrapped.robot.move_q_async(info['action'])



                # time.sleep(0.05)
                if info['Success']:
                    acc_sum += 1
                    col_sum += info['Collisions']
                    env.unwrapped.robot.move_q_async(np.append(info['action'][:7], 2))
                    res['mean_agent'] = np.mean(agent_time)
                    res['max_agent'] = np.max(agent_time)
                    res['mean_obs'] = np.mean(obs_time)
                    res['max_obs'] = np.max(obs_time)
                    res['mean_move'] = np.mean(move_time)
                    res['max_move'] = np.max(move_time)
                    print(res)
                    input('goal reached!! press enter for next rollout\n')
                    break

                # obs, _, _, _, info = env.step(action)

        print('AccSum: ', acc_sum)
        print('Collisions: ', col_sum)


if __name__ == "__main__":
    register_custom_envs()
    args = get_args()

    player = Player(args)
    player.play()
