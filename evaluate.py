# script for evaluating a model on all scenarios and ploting figures.
# Pass the path to the models parent directory, and everything will be saved there.
from common import get_args
from envs.config import register_custom_envs, make_env
from plot import plot_training
from play import Player


def evaluate(args):
    # plot training procedure

    plot_training(args.model_path, args.play_epoch)

    # init player
    player = Player(args, 30)
    # evaluate model
    res = {}
    for scene in ['lift_drawer']:#,'lifted_obst', 'dyn_sqr_obst', 'dyn_obst_v1', 'dyn_obst_v2', 'lift_drawer','lift_drawer_2','lift_shelf','lift_maze']:
        print('Evaluating: ', scene)
        res[scene] = player.play(scene)

    for k in res.keys():
        print(k, ': ', res[k])


if __name__ == "__main__":
    register_custom_envs()
    args = get_args()

    evaluate(args)

