# RL-Dyn-Env

train robot with random generated obstacles:

`train.py`

the model will be saved under newly created `log` directory.

Play  model in mujoco:

`play.py --model_path path/to/model/dir`

fine tune on custom scenario (e.g. lifted_obstacles):

`train.py --model_path path/to/model/dir --scenario lifted_obst`

For more details please refer to:
"https://www.notion.so/Getting-Started-751a40e40a7b4763a2655f86d9fccce9"






