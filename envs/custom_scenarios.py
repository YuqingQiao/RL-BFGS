# Dynamic Obstacles Scenarios

lifted_obst = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.2, 0.45, 0.42],
        'max': [1.4, 0.5, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.85, 0.49],
        'size': [0.025, 0.025, 0.05],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.85, 0.46],
        'site_size': [0.2, 0.025, 0.05]
    },
    'obstacle1': {
        'pos': [1.3, 0.65, 0.43],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.43],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.65, 0.49],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.49],
        'site_size': [0.2, 0.025, 0.025]
    }
}

dyn_sqr_obst = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.2, 0.45, 0.42],
        'max': [1.4, 0.5, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.85, 0.426],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.85, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle1': {
        'pos': [1.3, 0.65, 0.426],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [0.8, 0.6, 0.42],
        'size': [0.02, 0.02, 0.02],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [0.8, 0.6, 0.42],
        'site_size': [0.02, 0.02, 0.02]
    }
}

dyn_obst_v1 = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.2, 0.45, 0.42],
        'max': [1.4, 0.5, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.85, 0.426],
        'size': [0.06, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.85, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle1': {
        'pos': [1.3, 0.65, 0.426],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [0.8, 0.6, 0.42],
        'size': [0.02, 0.02, 0.02],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [0.8, 0.6, 0.42],
        'site_size': [0.02, 0.02, 0.02]
    }
}

dyn_obst_v2 = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.2, 0.45, 0.42],
        'max': [1.4, 0.5, 0.425]
    },
    'obstacle0': {
        'pos': [1.3, 0.85, 0.426],
        'size': [0.1, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.85, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle1': {
        'pos': [1.3, 0.65, 0.426],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.426],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [0.8, 0.6, 0.42],
        'size': [0.02, 0.02, 0.02],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [0.8, 0.6, 0.42],
        'site_size': [0.02, 0.02, 0.02]
    }
}

lift_obstx2 = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.1]
    },
    'target_init_space': {
        'min': [1.2, 0.35, 0.42],
        'max': [1.4, 0.4, 0.44]
    },
    'obstacle0': {
        'pos': [1.3, 0.82, 0.53],
        'size': [0.02, 0.02, 0.02],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.82, 0.53],
        'site_size': [0.2, 0.02, 0.02]
    },
    'obstacle1': {
        'pos': [1.3, 0.6, 0.48],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.6, 0.48],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.6, 0.59],
        'size': [0.05, 0.02, 0.02],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.6, 0.59],
        'site_size': [0.2, 0.02, 0.02]
    },
    'obstacle3': {
        'pos': [1.3, 0.82, 0.46],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.82, 0.46],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle4': {
        'pos': [0.3, 0.3, 0.3],
        'size': [0.01, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [0.3, 0.3, 0.3],
        'site_size': [0.01, 0.01, 0.01]
    },
    'obstacle5': {
        'pos': [0.3, 0.4, 0.3],
        'size': [0.01, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [0.3, 0.4, 0.3],
        'site_size': [0.01, 0.01, 0.01]
    },
    'obstacle6': {
        'pos': [0.3, 0.5, 0.3],
        'size': [0.01, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [0.3, 0.5, 0.3],
        'site_size': [0.01, 0.01, 0.01]
    },
}

lift_drawer = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.29, 0.42, 0.42],
        'max': [1.31, 0.43, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.85, 0.46],
        'size': [0.025, 0.025, 0.05],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.85, 0.46],
        'site_size': [0.2, 0.025, 0.05]
    },
    'obstacle1': {
        'pos': [1.3, 0.65, 0.43],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.43],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.65, 0.49],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.65, 0.49],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle3': {
        'pos': [1.3, 0.55, 0.445],
        'size': [0.15, 0.001, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 0.55, 0.445],
        'site_size': [0.15, 0.001, 0.03]
    },
    'obstacle4': {
        'pos': [1.3, 0.3, 0.445],
        'size': [0.15, 0.001, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 0.3, 0.445],
        'site_size': [0.15, 0.001, 0.03]
    },
    'obstacle5': {
        'pos': [1.455, 0.42, 0.445],
        'size': [0.001, 0.12, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.455, 0.42, 0.445],
        'site_size': [0.001, 0.12, 0.03]
    },
    'obstacle6': {
        'pos': [1.145, 0.42, 0.445],
        'size': [0.001, 0.12, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.145, 0.42, 0.445],
        'site_size': [0.001, 0.12, 0.03]
    },
}

lift_drawer_2 = {
    'obj_init_space': {
        'min': [1.2, 0.86],
        'max': [1.4, 0.86]
    },
    'target_init_space': {
        'min': [1.29, 0.42, 0.42],
        'max': [1.31, 0.43, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.68, 0.58],
        'size': [0.025, 0.025, 0.05],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.68, 0.58],
        'site_size': [0.2, 0.025, 0.05]
    },
    'obstacle1': {
        'pos': [1.3, 0.52, 0.55],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.52, 0.55],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.52, 0.61],
        'size': [0.025, 0.025, 0.025],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.52, 0.61],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle3': {
        'pos': [1.3, 0.3, 0.445],
        'size': [0.15, 0.001, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 0.3, 0.445],
        'site_size': [0.15, 0.001, 0.03]
    },
    'obstacle4': {
        'pos': [1.455, 0.42, 0.445],
        'size': [0.001, 0.12, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.455, 0.42, 0.445],
        'site_size': [0.001, 0.12, 0.03]
    },
    'obstacle5': {
        'pos': [1.145, 0.42, 0.445],
        'size': [0.001, 0.12, 0.03],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.145, 0.42, 0.445],
        'site_size': [0.001, 0.12, 0.03]
    },
    'obstacle6': {
        'pos': [1.3, 0.7, 0.52],
        'size': [0.2, 0.2, 0.001],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 0.7, 0.52],
        'site_size': [0.2, 0.2, 0.001]
    }

}

lift_shelf = {
    'obj_init_space': {
        'min': [1.28, 0.5],
        'max': [1.32, 0.52]
    },
    # 'target_init_space': {
    #     'min': [1.23, 1.05, 0.47],
    #     'max': [1.27, 1.1, 0.51]
    # }, halfway3
    'target_init_space': {
        'min': [1.23, 1, 0.47],
        'max': [1.35, 1.01, 0.48]
    },
    # 'target_init_space': {
    #     'min': [1.22, 1, 0.47],
    #     'max': [1.35, 1.05, 0.55]
    # },
    # 'target_init_space': {
    #     'min': [1.22, 1, 0.47],
    #     'max': [1.35, 1.05, 0.7]
    # },halfway6
    # 'obstacle0': {
    #     'pos': [1.1, 0.92, 0.85],
    #     'size': [0.025, 0.05, 0.15],
    #     'vel': {
    #         'min': 0.01,
    #         'max': 0.02
    #     },
    #     'dir': 0,
    #     'site_pos': [1.1, 0.92, 0.85],
    #     'site_size': [0.025, 0.05, 0.15]
    # },
    'obstacle0': {
        'pos': [0.8, 0.92, 0.45],
        'size': [0.025, 0.025, 0.05],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [0.8, 0.92, 0.45],
        'site_size': [0.025, 0.025, 0.05]
    },
    'obstacle1': {
        'pos': [1.3, 0.73, 0.43],
        'size': [0.2, 0.025, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.73, 0.43],
        'site_size': [0.2, 0.025, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.73, 0.49],
        'size': [0.01, 0.01, 0.02],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 0,
        'site_pos': [1.3, 0.73, 0.49],
        'site_size': [0.2, 0.01, 0.02]
    },
    'obstacle3': {
        'pos': [1.04, 1, 0.65],
        'size': [0.01, 0.01, 0.24],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.04, 1, 0.65],
        'site_size': [0.01, 0.01, 0.24]
    },
    'obstacle4': {
        'pos': [1.56, 1, 0.65],
        'size': [0.01, 0.01, 0.24],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.56, 1, 0.65],
        'site_size': [0.01, 0.01, 0.24]
    },
    'obstacle5': {
        'pos': [1.3, 1, 0.42],
        'size': [0.238, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 1, 0.42],
        'site_size': [0.238, 0.01, 0.01]
    },
    'obstacle6': {
        'pos': [1.3, 1, 0.86],
        'size': [0.238, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 1, 0.86],
        'site_size': [0.238, 0.01, 0.01]
    },
    'obstacle7': {
        'pos': [1.3, 1, 0.88],
        'size': [0.238, 0.01, 0.01],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.3, 1, 0.88],
        'site_size': [0.238, 0.01, 0.01]
    }

}

lift_maze = {
    'obj_init_space': {
        'min': [1.34, 1.15],
        'max': [1.35, 1.16]
    },
    # #bfgs target using cma policy
    # 'target_init_space': {
    #     'min': [1.44, 0.34, 0.42],
    #     'max': [1.45, 0.36, 0.42]
    # },
    # cma target
    'target_init_space': {
        'min': [1.25, 0.34, 0.42],
        'max': [1.3, 0.36, 0.42]
    },
    'obstacle0': {
        'pos': [0.7, 0.6, 0.46],
        'size': [0.02, 0.02, 0.06],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 1,
        'site_pos': [0.7, 0.6, 0.46],
        'site_size': [0.08, 0.02, 0.06]
    },
    'obstacle1': {
        'pos': [1.31, 0.7, 0.43],
        'size': [0.025, 0.2, 0.025],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.31, 0.7, 0.43],
        'site_size': [0.025, 0.2, 0.025]
    },
    'obstacle2': {
        'pos': [1.3, 0.73, 0.49],
        'size': [0.015, 0.015, 0.03],
        'vel': {
            'min': 0.01,
            'max': 0.02
        },
        'dir': 1,
        'site_pos': [1.3, 0.73, 0.49],
        'site_size': [0.015, 0.2, 0.03]
    },
    'obstacle3': {
        'pos': [1.4, 0.95, 0.65],
        'size': [0.1, 0.02, 0.02],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.4, 0.95, 0.65],
        'site_size': [0.1, 0.02, 0.02]
    },
    'obstacle4': {
        'pos': [1.2, 0.5, 0.55],
        'size': [0.1, 0.02, 0.02],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.2, 0.5, 0.55],
        'site_size': [0.1, 0.02, 0.02]
    },
    'obstacle5': {
        'pos': [1.48, 1.1, 0.65],
        'size': [0.02, 0.1, 0.02],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.48, 1.1, 0.65],
        'site_size': [0.02, 0.1, 0.02]
    },
    'obstacle6': {
        'pos': [1.1, 0.35, 0.65],
        'size': [0.02, 0.1, 0.02],
        'vel': {
            'min': 0,
            'max': 0
        },
        'dir': 0,
        'site_pos': [1.1, 0.35, 0.65],
        'site_size': [0.02, 0.1, 0.02]
    },
}

sim2real = {
    'obj_init_space': {
        'min': [1.2, 1.0],
        'max': [1.4, 1.05]
    },
    'target_init_space': {
        'min': [1.2, 0.45, 0.42],
        'max': [1.4, 0.5, 0.42]
    },
    'obstacle0': {
        'pos': [1.3, 0.75, 0.49],
        'size': [0.02, 0.042, 0.035],
        'vel': {
            'min': 0.01,
            'max': 0.1
        },
        'dir': 0,
        'site_pos': [1.3, 0.75, 0.451],
        'site_size': [0.2, 0.042, 0.035]
    },
    'obstacle1': {
        'pos': [1.3, 0.75, 0.41],
        'size': [0.2, 0.02, 0.005],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [1.3, 0.75, 0.41],
        'site_size': [0.2, 0.02, 0.005]
    },
    'obstacle2': {
        'pos': [0.8, 0.6, 0.42],
        'size': [0.02, 0.02, 0.02],
        'vel': {
            'min': 0.,
            'max': 0.
        },
        'dir': 0,
        'site_pos': [0.8, 0.6, 0.42],
        'site_size': [0.02, 0.02, 0.02]
    }
}

scenarios = {
    'lifted_obst': lifted_obst,
    'dyn_sqr_obst': dyn_sqr_obst,
    'dyn_obst_v1': dyn_obst_v1,
    'dyn_obst_v2': dyn_obst_v2,
    'sim2real': sim2real,
    'lift_drawer': lift_drawer,
    'lift_drawer_2': lift_drawer_2,
    'lift_shelf': lift_shelf,
    'lift_maze': lift_maze,
    'lift_obstx2': lift_obstx2
}
