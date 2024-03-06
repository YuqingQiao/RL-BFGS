import tensorflow as tf
from envs.config import make_env
from envs.custom_scenarios import scenarios
import numpy as np
import time


class Tester:
	def __init__(self, args, envs, rollouts=100):
		self.args = args
		self.rollouts = rollouts
		self.envs = envs
		self.render_env = make_env(args)
		if args.scenario:
			self.render_env.unwrapped.scenario = scenarios[args.scenario]

	def test_acc(self, agent, buffer, render=False):
		acc_sum, ex_rew_sum, col_sum = 0, 0, 0

		# make sure rollouts is multiple of vector envs
		assert self.rollouts % self.args.num_envs == 0

		for _ in range(int(self.rollouts/self.args.num_envs)):
			info = None
			if not self.args.scenario:
				# random number of obstacles for each env
				# self.envs.set_attr('num_obst', list(np.random.randint(0, self.args.num_obst + 1, size=self.args.num_envs)))
				# test with max number of obstacles
				self.envs.set_attr('num_obst', [self.args.num_obst]*self.args.num_envs)
			obs, _ = self.envs.reset()
			for timestep in range(self.args.timesteps):
				actions = agent.step_batch([dict(zip(obs, t)) for t in zip(*obs.values())]) 	# convert dict of lists into list of dicts
				obs, _, _, _, info = self.envs.step(actions)
			acc_sum += info['Success'].sum()
			ex_rew_sum += info['ExReward'].sum()
			col_sum += info['Collisions'].sum()

		# only render one example env
		if render:
			# get current settings
			self.render_env.unwrapped.obj_range = self.envs.get_attr('obj_range')[0]
			self.render_env.unwrapped.target_range = self.envs.get_attr('target_range')[0]
			self.render_env.unwrapped.num_obst = self.args.num_obst  # always render max number of obstacles
			# self.render_env.unwrapped.num_obst = np.random.randint(0, self.args.num_obst + 1)

			obs, _ = self.render_env.reset()
			for _ in range(self.args.timesteps):
				obs, _, _, _, info = self.render_env.step(agent.step(obs))
				self.render_env.render()
				time.sleep(0.01)

		acc = acc_sum / self.rollouts
		ex_rew_avg = ex_rew_sum / self.rollouts
		col_avg = col_sum / self.rollouts

		self.args.logger.add_record('Success', round(acc, 2))
		self.args.logger.add_record('CollisionsAvg', round(col_avg, 2))
		self.args.logger.add_record('ExRewardAvg', round(ex_rew_avg, 2))

		# log to tensorboard
		with self.args.logger.summary_writer.as_default():
			tf.summary.scalar('Success', acc, buffer.counter)
			tf.summary.scalar('CollisionsAvg', col_avg, buffer.counter)
			tf.summary.scalar('ExRewardAvg', ex_rew_avg, buffer.counter)

		return acc, col_avg
