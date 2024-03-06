from common import get_args, experiment_setup
import time
from envs.config import register_custom_envs
import os


def train():
    register_custom_envs()
    args = get_args()
    agent, buffer, learner, tester = experiment_setup(args)

    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Episodes')
    args.logger.add_item('TimeCost(sec)')

    # Algorithm info
    args.logger.add_item('Actor_Loss')
    args.logger.add_item('Critic_Loss')
    args.logger.add_item('Success')
    args.logger.add_item('CollisionsAvg')
    args.logger.add_item('ExRewardAvg')
    args.logger.add_item('Range')

    best_acc = -1
    best_col = 1000
    for epoch in range(args.epochs):
        start_time = time.time()

        learner.learn(args, agent, buffer)

        acc, col_avg = tester.test_acc(agent, buffer, render=args.render)

        args.cur_acc = acc

        args.logger.add_record('Epoch', str(epoch+1) + '/' + str(args.epochs))
        args.logger.add_record('Episodes', buffer.counter)
        args.logger.add_record('TimeCost(sec)', round(time.time() - start_time, 2))

        # Save learning progress to progress.csv file
        args.logger.save_csv()
        args.logger.tabular_show(epoch)

        # Save policy if new best success and lowest collisions were reached
        if acc > best_acc or (acc == best_acc and col_avg <= best_col):
            best_acc = acc
            best_col = col_avg
            policy_file = args.logger.my_log_dir + "model/saved_policy-best"
            agent.save(policy_file)

        # Save periodic policy every epoch
        policy_file = args.logger.my_log_dir + "model/saved_policy-" + str(epoch)
        agent.save(policy_file)


if __name__ == '__main__':
    train()
