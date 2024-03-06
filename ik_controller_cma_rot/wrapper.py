import ctypes
import numpy as np

lib = ctypes.CDLL('./ik_controller_cma_rot/build_shelf/libsolve_ik.so')
# choices: build or build2, where build is implemented by Fabio and build2 improves the real-time performance
# by doing a skip search of the solution set.

class Obstacle(ctypes.Structure):
    _fields_ = [("pos", ctypes.c_float * 3),
                ("size", ctypes.c_float * 3)]

class Options(ctypes.Structure):
    _fields_ = [
        ("robot_base", ctypes.c_float * 3),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("gamma", ctypes.c_float),
        ("sigma", ctypes.c_float),
        ("ngen", ctypes.c_int),
        ("popsize", ctypes.c_int),
        ("ftol", ctypes.c_float),
        ("bounds", (ctypes.c_float * 7) * 2)
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
    options_ctypes.ngen = options['ngen']
    options_ctypes.popsize = options['popsize']
    options_ctypes.ftol = options['ftol']
    for i, bound in enumerate(options['bounds']):
        options_ctypes.bounds[i] = (ctypes.c_float * 7)(*bound)

    # arrays are alredy pointers so we don't need byref
    lib.solve_ik(q_arr, obstacles_arr, target_pos_arr, target_rot_arr, ctypes.byref(options_ctypes), q_res, ctypes.byref(f_res))

    return np.array(q_res), f_res.value