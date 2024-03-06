from .normal import NormalLearner

learner_collection = {
	'normal': NormalLearner,
}


def create_learner(args, envs):
	return learner_collection[args.learn](args, envs)
