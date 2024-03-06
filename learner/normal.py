import copy
import numpy as np
from envs.config import make_env
from envs.utils import goal_distance
from algorithm.replay_buffer import Trajectory
import tensorflow as tf


class NormalLearner:
	def __init__(self, args, envs):
		self.args = args
		self.envs = envs

	def learn(self, args, agent, buffer):
		actor_losses = [0]
		critic_losses = [0]

		# adjust difficulty based on current accuracy if no custom scenario is given
		if args.scenario == '' and args.cur_acc >= 0.9:
			# add obstacles
			if self.envs.get_attr('obj_range')[0] > 0.95 and args.num_obst < 3:
				args.num_obst += 1
			# adjust range
			if self.envs.get_attr('obj_range')[0] < 0.99:
				self.envs.set_attr('obj_range', [x + 0.3 for x in self.envs.get_attr('obj_range')])
			if self.envs.get_attr('target_range')[0] < 0.99:
				self.envs.set_attr('target_range', [x + 0.3 for x in self.envs.get_attr('target_range')])

		for c in range(args.cycles):
			if args.scenario == '':
				# # random number of obstacles for each env, many obstacles should come more often
				n = list(range(args.num_obst + 1))
				p = np.array([2**i for i in n])
				p = p/p.sum()
				self.envs.set_attr('num_obst', list(np.random.choice(n, p=p, size=args.num_envs)))
				# random number of obstacles for each env
				# self.envs.set_attr('num_obst', list(np.random.randint(0, args.num_obst + 1, size=args.num_envs)))

			obs, _ = self.envs.reset()
			obs = [dict(zip(obs, t)) for t in zip(*obs.values())]

			current_trajectories = [Trajectory(obs[i]) for i in range(args.num_envs)]
			achieved_trajectories = [[obs[i]['achieved_goal']] for i in range(args.num_envs)]

			for timestep in range(args.timesteps):
				actions = agent.step_batch(obs, explore=True)
				obs, rewards, done, _, _ = self.envs.step(actions)
				for i in range(self.args.num_envs):
					achieved_trajectories[i].append(obs['achieved_goal'][i])
				if timestep == args.timesteps - 1:
					done = [True] * self.args.num_envs
				# convert dict of list to list of dicts
				obs = [dict(zip(obs, t)) for t in zip(*obs.values())]
				# store steps
				for i in range(self.args.num_envs):
					current_trajectories[i].store_step(actions[i], obs[i], rewards[i], done[i])
				if any(done):
					break

			for current in current_trajectories:
				buffer.store_trajectory(current)
				agent.normalizer_update(buffer.sample_batch())

			# train
			for b in range(args.train_batches):
				# train with Hindsight Goals (HER step)
				batch = buffer.sample_batch()
				actor_loss, critic_loss = agent.train(batch)

				actor_losses.append(actor_loss)
				critic_losses.append(critic_loss)
				# update target network
				agent.target_update()

		for key, val in zip(['Actor_Loss', 'Critic_Loss'], [actor_losses, critic_losses]):
			args.logger.add_record(key, np.round(np.mean(val), 2))
			with args.logger.summary_writer.as_default():
				tf.summary.scalar(key, np.mean(val), buffer.counter)

		args.logger.add_record('Range', round(self.envs.get_attr('target_range')[0], 2))

