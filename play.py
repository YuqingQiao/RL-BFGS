from common import get_args
from envs.config import register_custom_envs, make_env
from envs.custom_scenarios import scenarios
from algorithm import create_agent
import os
import numpy as np
import time


class Player:
    def __init__(self, args, rollouts):
        # initialize environment
        self.args = args

        self.info = []
        self.rollouts = rollouts

        self.agent = create_agent(args)
        self.agent.load(os.path.join(args.model_path, "model/saved_policy-{}".format(args.play_epoch)))

    def play(self, scene=''):
        if scene != '':
            self.args.scenario = scene
        # play policy on env
        env = make_env(self.args)
        if self.args.scenario:
            env.unwrapped.scenario = scenarios[self.args.scenario]
        else:
            env.unwrapped.obj_range = 1
            env.unwrapped.target_range = 1
            env.unwrapped.num_obst = 1

        acc_sum = 0
        col_sum = 0
        seed = 1
        control_step =[]
        control_times = []
        res = {}
        for i in range(self.rollouts):
            print(seed + i)
            obs, _ = env.reset(seed=seed + i)

            for timestep in range(self.args.timesteps):
                st = time.perf_counter()
                # get rl action
                action = self.agent.step(obs)
                # time.sleep(0.05)
                t = time.perf_counter() - st
                # do step in environment
                obs, _, _, _, info = env.step(action)

                if self.args.control_mode == 'ik_controller':
                    t += env.unwrapped.IKC.control_time
                # the startup of the agent always takes a bit longer, so we don't measure the first step
                if timestep > 0:
                    control_times.append(t)

                if self.args.render:
                    env.render()

                if info['Success'] or timestep == self.args.timesteps-1:
                    acc_sum += info['Success']
                    col_sum += info['Collisions']
                    control_step.append(timestep)
                    break

        res['success_rate'] = acc_sum / self.rollouts
        res['avg_collisions'] = col_sum / self.rollouts
        res['mean_time'] = np.mean(control_times)
        res['max_time'] = np.max(control_times)
        res['avg_step'] = np.mean(control_step)

        return res


if __name__ == "__main__":
    register_custom_envs()
    args = get_args()

    player = Player(args, 30)
    print(player.play())
