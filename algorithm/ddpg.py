import os

import numpy as np
import tensorflow as tf

from algorithm.replay_buffer import goal_based_process


class Normalizer(tf.Module):
	"""Normalizes inputs.

	Args:
		args: parsed arguments.
		eps_std (float, optional): A small value used for stability. Defaults to 1e-2.
		norm_clip (int, optional): The maximum value for normalization. Defaults to 5.
	"""

	def __init__(self, args, eps_std=1e-2, norm_clip=5):
		self.shape = args.obs_dims
		self.eps_std = eps_std
		self.norm_clip = norm_clip

		self.sum = tf.Variable(tf.zeros(self.shape), dtype=tf.float32, trainable=False)
		self.sum_sqr = tf.Variable(tf.zeros(self.shape), dtype=tf.float32, trainable=False)
		self.cnt = tf.Variable(tf.zeros([1]), dtype=tf.float32, trainable=False)
		self.mean = tf.Variable(tf.zeros(self.shape), dtype=tf.float32, trainable=False)
		self.sqrt = tf.Variable(tf.zeros(self.shape), dtype=tf.float32, trainable=False)
		self.std = tf.Variable(tf.zeros(self.shape), dtype=tf.float32, trainable=False)

	def update(self, batch):
		inputs = tf.cast(tf.concat([batch['obs'], batch['obs_next']], 0), tf.float32)
		self.update_graph(inputs)

	@tf.function
	def update_graph(self, inputs):
		self.sum.assign_add(tf.reduce_sum(inputs, 0))
		self.sum_sqr.assign_add(tf.reduce_sum(tf.square(inputs), 0))
		self.cnt.assign_add([inputs.shape[0]])

		self.mean.assign(self.sum / self.cnt)
		# due to floating points this can sometimes be negative
		self.sqrt.assign(tf.sqrt(tf.maximum(0.0, self.sum_sqr / self.cnt - tf.square(self.sum / self.cnt))))
		self.std.assign(tf.maximum(self.eps_std, self.sqrt))

	def normalize(self, inputs):
		inputs = tf.cast(inputs, tf.float32)
		return self.normalize_graph(inputs)

	@tf.function
	def normalize_graph(self, inputs):
		return tf.clip_by_value(tf.math.divide_no_nan((inputs - self.mean), self.std), -self.norm_clip, self.norm_clip)


def neural_net(arch):
	"""Builds and returns the actor network.

	Args:
		arch (dict): A dictionary containing the architecture specifications for the network.

	Returns:
		model (tf.keras.Model): The actor network.
	"""
	# Input layer
	if type(arch['input_shape']) == list:
		# Critic
		states = tf.keras.layers.Input(shape=(arch['input_shape'][0],), dtype=tf.float32)
		actions = tf.keras.layers.Input(shape=(arch['input_shape'][1],), dtype=tf.float32)
		out = tf.keras.layers.Concatenate(axis=1)([states, actions])
	else:
		# Actor
		states = tf.keras.layers.Input(shape=(arch['input_shape'],), dtype=tf.float32)
		actions = None
		out = states

	# Hidden layers
	for (layer_shape, batch_norm, activation) in zip(arch['layer_shapes'], arch['batch_norms'], arch['activations']):
		out = tf.keras.layers.Dense(layer_shape, activation=activation, kernel_initializer="glorot_normal")(out)
		if batch_norm:
			out = tf.keras.layers.BatchNormalization()(out)

	# Build and return model
	if type(arch['input_shape']) == list:
		# Critic
		model = tf.keras.Model([states, actions], out)
	else:
		# Actor
		model = tf.keras.Model(states, out)

	return model


class DDPG(tf.Module):
	"""
	Implementation of Deep Deterministic Policy Gradients algorithm for robotic manipulation.
	"""

	def __init__(self, args):
		"""
		Initialize the DDPG model with the given parameters.
		"""
		# gpu = tf.config.list_physical_devices('GPU')[0]
		# tf.config.experimental.set_memory_growth(gpu, True)

		# Actor Architecture
		actor_arch = {
			'input_shape': args.obs_dims[0],
			'layer_shapes': args.actor_layer_sizes + [args.acts_dims[0]],
			'batch_norms': args.actor_batch_norm + [False],
			'activations': ['relu'] * len(args.actor_batch_norm) + ['tanh']
		}

		# Critic Architecture
		critic_arch = {
			'input_shape': [args.obs_dims[0], args.acts_dims[0]],
			'layer_shapes': args.critic_layer_sizes + [1],
			'batch_norms': args.critic_batch_norm + [False],
			'activations': ['relu'] * len(args.actor_batch_norm) + [None]
		}

		#  parameters
		self.args = args
		self.trace = True # helper to only trace the graph once

		self.normalizer = Normalizer(args)

		# initialize everything
		self.actor_network = neural_net(actor_arch)
		self.critic_network = neural_net(critic_arch)
		self.actor_target = neural_net(actor_arch)
		self.critic_target = neural_net(critic_arch)

		# making the weights equal initially
		self.actor_target.set_weights(self.actor_network.get_weights())
		self.critic_target.set_weights(self.critic_network.get_weights())

		# optimizers
		self.critic_optimizer = tf.keras.optimizers.Adam(args.q_lr)
		self.actor_optimizer = tf.keras.optimizers.Adam(args.pi_lr)

	def step(self, obs, explore=False):
		"""
		Returns an action for a given observation, with epsilon-greedy exploration.

		Args:
			obs (numpy.ndarray): current observation.
			explore (bool, optional): whether to perform epsilon-greedy exploration or not. Default is False.

		Returns:
			numpy.ndarray: action to be taken for the given observation.
		"""
		if self.args.goal_based:
			obs = goal_based_process(obs)

		# eps-greedy exploration
		if explore and np.random.uniform() <= self.args.eps_act:
			return np.random.uniform(-1, 1, size=self.args.acts_dims)

		obs = self.normalizer.normalize(obs)
		action = self.actor_network(tf.expand_dims(obs, axis=0)).numpy()[0]

		# uncorrelated gaussian exploration
		if explore:
			action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
		action = np.clip(action, -1, 1)

		return action

	def step_batch(self, obs_batch, explore=False):
		"""
			Returns actions for a batch of observations

			Args:
				obs_batch: observation batch
				explore: whether to perform epsilon-greedy exploration or not. Default is False.
		"""
		if self.args.goal_based:
			obs_batch = [goal_based_process(obs) for obs in obs_batch]

		# eps-greedy exploration
		if explore and np.random.uniform() <= self.args.eps_act:
			return list(np.random.uniform(-1, 1, size=(len(obs_batch), self.args.acts_dims[0])))

		obs_batch = [self.normalizer.normalize(obs) for obs in obs_batch]
		actions = self.actor_network(tf.convert_to_tensor(obs_batch)).numpy()

		# uncorrelated gaussian exploration
		if explore:
			actions += np.random.normal(0, self.args.std_act, size=(len(obs_batch), self.args.acts_dims[0]))
		actions = np.clip(actions, -1, 1)

		return actions

	def train(self, batch):
		"""
			Trains the DDPG model on the given batch of data

			Args:
				batch (dict): Dictionary containing training batch

			Returns:
				pi_q_loss (float): actor network loss.
				pi_l2_loss (float): L2 regularization loss for the actor network.
				q_loss (float): critic network loss.
		"""
		obs, obs_next, acts, rews, done = batch.values()

		with self.args.logger.summary_writer.as_default():
			if self.trace:
				tf.summary.trace_on(graph=True)

			obs = self.normalizer.normalize(obs)
			obs_next = self.normalizer.normalize(obs_next)

			actor_loss, critic_loss = self.update_weights(
				tf.convert_to_tensor(obs, dtype=tf.float32),
				tf.convert_to_tensor(obs_next, dtype=tf.float32),
				tf.convert_to_tensor(acts, dtype=tf.float32),
				tf.convert_to_tensor(rews, dtype=tf.float32),
				tf.convert_to_tensor(done, dtype=tf.float32)
			)

			# trace model graph only once
			if self.trace:
				tf.summary.trace_export(
					name="model_graph",
					step=0,
					profiler_outdir='log/board/'
				)
				self.trace = False

		return actor_loss, critic_loss

	@tf.function
	def update_weights(self, obs, obs_next, acts, rews, done):
		"""
		Computes the loss for critic and actor network and updates their weights.

		Args:
			obs (numpy.ndarray): current observation.
			obs_next (numpy.ndarray): next observation.
			acts (numpy.ndarray): actions taken at current observation.
			rews (numpy.ndarray): rewards received for taking actions at current observation.
			done (numpy.ndarray): whether the episode ended after taking actions at current observation.

		Returns:
			pi_q_loss (float): actor network loss.
			pi_l2_loss (float): L2 regularization loss for the actor network.
			q_loss (float): critic network loss.
		"""
		with tf.GradientTape() as tape:
			# evaluate return
			ret = self.critic_target([obs_next, self.actor_target(obs_next, training=True)], training=True)
			if self.args.clip_return:
				ret = tf.clip_by_value(
					ret,
					self.args.reward_min / (1.0 - self.args.gamma),
					self.args.reward_max / (1.0 - self.args.gamma)
				)
			# define target
			y = tf.stop_gradient(rews + self.args.gamma * ret) 	# TODO: terminal state missing (1 - done)
			# define the delta Q
			critic_loss = tf.reduce_mean(tf.square(self.critic_network([obs, acts], training=True) - y))
		critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
		self.critic_optimizer.apply_gradients(
			zip(critic_grad, self.critic_network.trainable_variables))

		with tf.GradientTape() as tape:
			# define the delta mu
			acts_pi = self.actor_network(obs, training=True)
			pi_q_loss = -tf.reduce_mean(self.critic_network([obs, acts_pi]))
			pi_l2_loss = self.args.act_l2 * tf.reduce_mean(tf.square(acts_pi))
			actor_loss = tf.add(pi_q_loss, pi_l2_loss)

		actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
		self.actor_optimizer.apply_gradients(
			zip(actor_grad, self.actor_network.trainable_variables))
		return actor_loss, critic_loss

	def target_update(self):
		"""
		Updates the target networks.
		"""
		self.target_update_graph(self.actor_target.variables, self.actor_network.variables)
		self.target_update_graph(self.critic_target.variables, self.critic_network.variables)

	@tf.function
	def target_update_graph(self, target, network):
		for (target_weight, net_weight) in zip(target, network):
			target_weight.assign(self.args.polyak * target_weight + (1 - self.args.polyak) * net_weight)

	def normalizer_update(self, batch):
		"""
		Updates the normalizer used for normalizing observations.

		Args:
			obs (np.ndarray): Current observations.
		"""
		self.normalizer.update(batch)

	def get_value(self, obs):
		"""
		Gets the value (critic output) for the given observations.

		Args:
			obs (np.ndarray): Current observations.

		Returns:
			np.ndarray: Value for the given observations and actions.
		"""
		obs = self.normalizer.normalize(obs)
		obs = tf.convert_to_tensor(obs)
		value = self.critic_network([obs, self.actor_network(obs)]).numpy()

		return value

	def save(self, path):
		"""
		Saves the weights of the actor and critic networks to a file.

		Args:
			path (str): Path to the file to save the weights to.
		"""
		parent_dir = os.path.dirname(path)
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir)
		# Save Model
		ckpt = tf.train.Checkpoint(self)
		manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=None)
		manager.save()

	def load(self, path):
		"""
		Loads the weights of the actor and critic networks from a file.

		Args:
			path (str): Path to the file to load the weights from.
		"""
		ckpt = tf.train.Checkpoint(self)
		manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=None)
		ckpt.restore(manager.latest_checkpoint).expect_partial()


# def critic_network(arch):
# 	"""Builds and returns the critic network.
#
# 	If arch['layer_shape'] contains lists in the beginning, state and actions will be passed through separate layers
# 	before concatenating. E.g. [[512, 512], 256, 256, 1]
#
# 	Args:
# 		arch (dict): A dictionary containing the architecture specifications for the network.
#
# 	Returns:
# 		model (tf.keras.Model): The critic network.
# 	"""
# 	# helper
# 	concat = True
#
# 	# Input layers
# 	states = tf.keras.layers.Input(shape=(arch['input_shape'][0],), dtype=tf.float32)
# 	actions = tf.keras.layers.Input(shape=(arch['input_shape'][1],), dtype=tf.float32)
#
# 	# Normalize observations
# 	s_out = states
# 	a_out = actions
#
# 	# Hidden layers
# 	out = None
# 	for (layer_shape, batch_norm, activation) in zip(arch['layer_shapes'], arch['batch_norms'], arch['activations']):
# 		if type(layer_shape) == list:
# 			s_out = tf.keras.layers.Dense(layer_shape[0], activation=activation[0], kernel_initializer="glorot_normal")(
# 				s_out)
# 			if batch_norm[0]:
# 				s_out = tf.keras.layers.BatchNormalization()(s_out)
# 			a_out = tf.keras.layers.Dense(layer_shape[1], activation=activation[1], kernel_initializer="glorot_normal")(a_out)
# 			if batch_norm[1]:
# 				a_out = tf.keras.layers.BatchNormalization()(a_out)
# 		else:
# 			if concat:
# 				out = tf.keras.layers.Concatenate(axis=1)([s_out, a_out])
# 				concat = False
# 			out = tf.keras.layers.Dense(layer_shape, activation=activation, kernel_initializer="glorot_normal")(out)
# 			if batch_norm:
# 				out = tf.keras.layers.BatchNormalization()(out)
#
# 	# Build and return model
# 	model = tf.keras.Model([states, actions], out)
# 	return model