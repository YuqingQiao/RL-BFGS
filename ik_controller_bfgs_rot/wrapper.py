import ctypes
import numpy as np

lib = ctypes.CDLL('./ik_controller_bfgs_rot/build_shelf/libsolve_ik.so')
# choices: build_rand_obst or build_drawer or build_drawer_2 or build_shelf
# these c++ builds are slightly different in the obstacle dimension. dim=3/7/6/8


class Obstacle(ctypes.Structure):
    _fields_ = [("pos", ctypes.c_float * 3),
                ("size", ctypes.c_float * 3)]

class Options(ctypes.Structure):
    _fields_ = [
        ("robot_base", ctypes.c_float * 3),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("gamma", ctypes.c_float),
        ("fDelta", ctypes.c_float),
        ("xDelta", ctypes.c_float),
        ("gradNorm", ctypes.c_float),
        ("maxiter", ctypes.c_int),
        ("population", ctypes.c_int),
        ("sigma", ctypes.c_float),
        ("skip", ctypes.c_int),
        ("bounds", (ctypes.c_float * 7) * 2),
    ]


# Define the argument types and return types
lib.solve_ik.argtypes = (ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(Obstacle),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(Options),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float))


def solve_ik(q, obstacles, target_pos, target_rot, options):
    # Convert data to ctypes
    q_arr = (ctypes.c_float * 7)(*q)
    obstacles_arr = (Obstacle * len(obstacles))(*[
        Obstacle((ctypes.c_float * 3)(*obs['pos']),
                 (ctypes.c_float * 3)(*obs['size']))
        for obs in obstacles
    ])
    target_pos_arr = (ctypes.c_float * 3)(*target_pos)
    target_rot_arr = (ctypes.c_float * 3)(*target_rot)
    q_res = (ctypes.c_float * 7)()
    f_res = ctypes.c_float()

    options_ctypes = Options()
    options_ctypes.robot_base = (ctypes.c_float * 3)(*options['robot_base'])
    options_ctypes.alpha = options['alpha']
    options_ctypes.beta = options['beta']
    options_ctypes.gamma = options['gamma']
    options_ctypes.sigma = options['sigma']
    options_ctypes.maxiter = options['maxiter']
    options_ctypes.xDelta = options['xDelta']
    options_ctypes.fDelta = options['fDelta']
    options_ctypes.population = options['population']
    options_ctypes.gradNorm = options['gradNorm']
    options_ctypes.skip = options['skip']
    for i, bound in enumerate(options['bounds']):
        options_ctypes.bounds[i] = (ctypes.c_float * 7)(*bound)

    # arrays are alredy pointers so we don't need byref
    lib.solve_ik(q_arr, obstacles_arr, target_pos_arr, target_rot_arr, ctypes.byref(options_ctypes), q_res, ctypes.byref(f_res))

    return np.array(q_res), f_res.value