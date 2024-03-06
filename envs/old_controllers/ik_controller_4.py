import time
import numpy as np
from scipy.optimize import minimize

# scipy
# no c++

TRANSFORM = [
    # Joint 1
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0.333],
        [0, 0, 0, 1]
    ]),
    # Joint 2
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [0, 0, 1, 0],
        [-np.sin(theta), -np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Joint 3
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [0, 0, -1, -0.316],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Joint 4
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0.0825],
        [0, 0, -1, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Joint 5
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, -0.0825],
        [0, 0, 1, 0.384],
        [-np.sin(theta), -np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Joint 6
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [0, 0, -1, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Joint 7
    lambda theta: np.array([
        [np.cos(theta), -np.sin(theta), 0, 0.088],
        [0, 0, -1, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1]
    ]),
    # Flange
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.207],
        [0, 0, 0, 1]
    ]),
]

class IKController:
    def __init__(self, T_wb, q, obstacles):
        """Inverse Kinematic Controller.

         The input of the controller are a target cartesian position and quaternion orientation for
         the frame of the end-effector.
         The controller will output angular displacements in the joint space to achieve the desired
         end-effector configuration.

         The controller solves the nonlinear constrained optimization problem of reaching the desired end-effector
         configuration under the constraints of minimal joint displacements and collision avoidance.

         Args:
             T_wb: Transformation from world frame to the robot base
             q: starting joint angles
             obstacles: starting obstacle positions
        """
        self.T_wb = T_wb
        self.current_q = q
        self.current_obstacles = obstacles
        self.calc_dist = True
        # list of transformation matrices from forward kinematics
        self.fk_list = self.forward_kinematics(self.current_q)
        # capsule positions
        self.robot_capsules, self.obst_capsules = self.get_capsule_pos(self.fk_list, self.current_obstacles)
        # init
        self.target_pos = np.empty(3)
        self.target_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # weights
        self.alpha = 0.7    # position error
        self.beta = 0.2     # orientation error
        self.gamma = 0.1    # joint movement error
        self.ftol = 0.000001
        # joint limits
        self.jnt_bounds = [
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        ]
        self.total_time_cpp = 0
        self.control_time = 0

    def solve(self, q, obstacles, target_pos):
        self.current_q = q.copy()
        self.current_obstacles = obstacles
        self.target_pos = target_pos
        cached_pop = []
        x_cached = []

        def store(xk):
            x = xk
            f = self.obj_fun(x)
            cached_pop.append((x, f))

        st = time.perf_counter()
        #guassian 500 {'success_rate': 1.0, 'avg_collisions': 0.0, 'mean_time': 0.03835700908998027, 'max_time': 0.17813651699998445}
        for i in range(100):
            q_normal = np.random.normal(self.current_q, 0.1)
            store(q_normal)

        cached_pop.sort(key=lambda x: x[1])

        # sum = 0
        # mean = 0
        # cov = 0
        #
        # for i in range(len(cached_pop)):
        #     sum = sum + cached_pop[i][0]


        #scipy package
        # SLSQP
        cons = {'type':'ineq',
                'fun':lambda x: self.dist_constraint(x)}

        # q_min = minimize(self.obj_fun, self.current_q, method='SLSQP',
        #                                    constraints=cons,
        #                                    options={'ftol': 1e-5, 'disp': True})

        #'ftol': 1e-4 {'success_rate': 0.9, 'avg_collisions': 0.0, 'mean_time': 0.19905246225121095, 'max_time': 0.554984052000691}
        #'ftol': 1e-5 {'success_rate': 1.0, 'avg_collisions': 0.0, 'mean_time': 0.31420621130984416, 'max_time': 0.6836270860003424}

        # COBYLA
        q_min = minimize(self.obj_fun, self.current_q, method='COBYLA',
                         constraints=cons,
                         options={'tol': 2e-3, 'disp': False})

        # suc: 0.74 col: 0.03 ave: 0.2263 max: 0.7419

        #trust-constr
        # q_min = minimize(self.obj_fun, self.current_q, method='trust-constr',
        #                  constraints=cons,
        #                  options={'xtol': 1e-2, 'gtol': 1e-2, 'barrier_tol': 1e-2, 'disp': True})
        # #'xtol': 1e-2, 'gtol': 1e-2, 'barrier_tol': 1e-2 {'success_rate': 1.0, 'avg_collisions': 0.0, 'mean_time': 0.4664712916376745, 'max_time': 4.553669453999646}

        # q_res = q_min.x

        # BFGS
        # with gaussian {'success_rate': 0.99, 'avg_collisions': 0.0, 'mean_time': 0.07919230591479363, 'max_time': 0.29529873299998144}
        # L-BFGS-B
        # with gaussian {'success_rate': 0.98, 'avg_collisions': 0.0, 'mean_time': 0.14302286410811882, 'max_time': 0.49707902499994816}
        #CG
        # with gaussian {'success_rate': 0.9, 'avg_collisions': 0.0, 'mean_time': 0.14461662058731647, 'max_time': 0.4972258770003464}
        # Powell
        # with gaussian {'success_rate': 1.0, 'avg_collisions': 0.0, 'mean_time': 0.09941636415205964, 'max_time': 0.3568026809998628}
        # Nelder-Mead
        # with gaussian {'success_rate': 1.0, 'avg_collisions': 0.0, 'mean_time': 0.2919324639011805, 'max_time': 2.2965252700005294}
        # q_min = minimize(self.obj_fun, cached_pop[0][0], method='COBYLA', tol = 1e-3,
        #                  options={'disp': True, 'maxiter': 10}, callback=store)


    #
        cached_pop.sort(key=lambda x: x[1])
        x_cached = [i[0] for i in cached_pop]
        # q_res, _ = cached_pop[0]

    #         # f_cached = [i[1] for i in cached_pop]
    #         #
    #         i = 0
    # quick search by skipping inds
    #     for ind in range(len(x_cached)):
    #         if ind == 0:
    #             step = 0
    #         else:
    #             step = 64 // ind
    #             if step < 1:
    #                 step = 1
    #         if self.dist_constraint(x_cached[ind + step]) > 0:
    #             q_res = x_cached[ind + step]
    #             # print(f_cached)
    #             break

        for ind in range(len(x_cached)):
            if self.dist_constraint(x_cached[ind]) > 0:
                q_res = x_cached[ind]
                # print(len(x_cached))
                break
    #         else:
    #             cached_pop = []
    #             x_cached = []
    #             for i in range(500):
    #                 q_normal = np.random.normal(self.current_q, 0.05)
    #                 store(q_normal)
    #             cached_pop.sort(key=lambda x: x[1])
    #             x_cached = [i[0] for i in cached_pop]
    #             for ind in range(len(x_cached)):
    #                 if self.dist_constraint(x_cached[ind]) > 0:
    #                     q_res = x_cached[ind]
    #                     print(len(x_cached))
    #                     break

        self.control_time = time.perf_counter() - st
        fk_list = self.forward_kinematics(q_res)
        # get capsule positions
        robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, self.current_obstacles)

        return q_res, robot_capsules, obst_capsules

    def obj_fun(self, q):
        # calculate forward kinematics
        fk_list = self.forward_kinematics(q)

        current_pos = fk_list[-1][:3, 3]
        current_rot = fk_list[-1][:3, :3]
        pos_error = np.linalg.norm(current_pos - self.target_pos)
        orientation_error = np.arccos(np.clip((np.trace(current_rot.T @ self.target_rot) - 1) / 2, -1, 1))
        angle_error = np.linalg.norm(q - self.current_q)

        error = self.alpha * pos_error + self.beta * orientation_error + self.gamma * angle_error
        return error

    def forward_kinematics(self, q):
        """
        Calculates the forward kinematics of the robot for each link, given its angle.

        Args:
            q (np.ndarray): array with joint angles

        Return:
            fk_list (dict): a list with the resulting transformation for each joint.
        """
        fk = np.eye(4)
        # Add robot base displacement
        fk[:3, 3] += self.T_wb
        fk_list = [fk]
        for i, q_i in enumerate(q):
            fk = fk @ TRANSFORM[i](q_i)
            fk_list.append(fk)
        # for flange without q
        fk = fk @ TRANSFORM[-1]
        fk_list.append(fk)

        return fk_list

    def get_capsule_pos(self, fk_list, obstacles):
        """
        Calculates the desired capsule positions of the robot and obstacles, based on the forward kinematics and
        obstacle size/positions.

        The forward kinematics gives us the coordinate frames of the franka robot
        (cf. https://frankaemika.github.io/docs/control_parameters.html)
        We will use those frames to create geometric capsules that encapsulate the robot arm. We want to use as few
        capsules as necessary to encapsulate the robot as tight as possible.

        We will use the 7 coordinate frames resulting from the forward kinematics to simplify the encapsulation process.
        Capsules are usually defined by two 3D points from the start to end position and a scalar radius.

        The capsule positions and sizes here were chosen specifically for the Franka Emika Panda Robot.

        For more information about capsules and the distance calculation refer to
        "Efficient Calculation of Minimum Distance Between Capsules and Its Use in Robotics, 2019"

        Args:
            fk_list:
            obstacles:
        Returns:

        """
        # we can skip the robot base (up to the second coordinate frame) as it can't be moved to dodge obstacles.
        robot_capsules = []
        # 1: Elbow (4. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[4], np.array([0, 0, -0.055, 1]))[:3],
            'u': np.dot(fk_list[4], np.array([0, 0, 0.055, 1]))[:3],
            'roh': 0.075
        })
        # 2: Forearm 1 (5. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[5], np.array([0, 0, -0.23, 1]))[:3],
            'u': np.dot(fk_list[5], np.array([0, 0, -0.32, 1]))[:3],
            'roh': 0.07
        })
        # 3: Forearm 2 (5. & 6. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[5], np.array([0, 0.07, -0.18, 1]))[:3],
            'u': np.dot(fk_list[6], np.array([0, 0, -0.1, 1]))[:3],
            'roh': 0.045
        })
        # 4: Wrist (6. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[6], np.array([0, 0, -0.08, 1]))[:3],
            'u': np.dot(fk_list[6], np.array([0, 0, 0.01, 1]))[:3],
            'roh': 0.067
        })
        # 5: Hand 1 (7. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[7], np.array([0, 0, -0.04, 1]))[:3],
            'u': np.dot(fk_list[7], np.array([0, 0, 0.175, 1]))[:3],
            'roh': 0.065
        })
        # 6: Hand 2 (7. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[7], np.array([0, 0.061, 0.13, 1]))[:3],
            'u': np.dot(fk_list[7], np.array([0, -0.061, 0.13, 1]))[:3],
            'roh': 0.065
        })
        # 7: Hand 3 (7. Frame)
        robot_capsules.append({
            'p': np.dot(fk_list[7], np.array([0.03, 0.06, 0.085, 1]))[:3],
            'u': np.dot(fk_list[7], np.array([0.06, 0.03, 0.085, 1]))[:3],
            'roh': 0.035
        })

        # Obstacles (we will use the cuboids largest dimension capsule height and second largest as radius)
        obst_capsules = []
        for obst in obstacles:
            # sorted dimensions
            dims = np.argsort(obst['size'])
            l = obst['size'][dims[-1]]
            p = obst['pos'].copy()
            p[dims[-1]] += l
            u = obst['pos'].copy()
            u[dims[-1]] -= l
            obst_capsules.append({
                'p': p,
                'u': u,
                'roh': np.sqrt(obst['size'][dims[0]]**2 + obst['size'][dims[1]]**2) + 0.01      # safe space
            })

        return robot_capsules, obst_capsules

    def dist_constraint(self, q):
        if not self.calc_dist:
            return 1.0

        # calculate forward kinematics
        fk_list = self.forward_kinematics(q)

        # get capsule positions
        robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, self.current_obstacles)

        res = self.calc_min_dist(robot_capsules, obst_capsules)

        return res

    def calc_min_dist(self, robot_capsules, obst_capsules):
        """
        Computes the minimum distance between all capsules of the robot arm with all obstacle capsules.

        Returns:
            min_dist: The minimum distance from all comparisons
        """
        distances = []
        for n, r_caps in enumerate(robot_capsules):
            for m, o_caps in enumerate(obst_capsules):
                #
                p1 = r_caps['p']
                p2 = o_caps['p']
                s1 = r_caps['u'] - p1
                s2 = o_caps['u'] - p2

                A = np.stack([s2, -s1], 1)
                y = p2-p1
                Q, R = np.linalg.qr(A)

                u = lambda x: np.dot(R, x) + np.dot(Q.T, y)

                # check whether the origin is inside the parallelogram given by u. If it is inside, we can abort the calculation
                # and return that a collision is imminent. repeat the first point to create a closed loop.
                vertices = [u([0,0]), u([0,1]), u([1, 1]), u([1, 0])]

                sides = []
                u_temp = []
                for i in range(4):
                    # o = np.array([0, 0])
                    v1 = vertices[i]
                    v2 = vertices[i+1] if i != 3 else vertices[0]
                    # for each edge of the parallelogram, check on which side the point lies. If all queries return the same
                    # result, the origin lies inside the parallelogram.
                    res = -v1[1] * (v2[0] - v1[0]) - (-v1[0]) * (v2[1] - v1[1])
                    if res >= 0:
                        sides.append(1)
                    else:
                        sides.append(-1)
                    # for each egde we also calculate the closest point to the origin
                    u_temp.append(self.closest_point_on_segment(v1, v2))

                if abs(sum(sides)) == 4:
                    u_min = np.array([0, 0])
                else:
                    u_min = u_temp[np.argmin(np.linalg.norm(u_temp, axis=1))]

                # now calculate the distance
                dist = np.sqrt(u_min.T @ u_min + y.T @ y - y.T @ Q @ Q.T @ y) - r_caps['roh'] - o_caps['roh']
                distances.append(dist)
        return np.min(distances)

    @staticmethod
    def closest_point_on_segment(a, b):
        # If the origin is 'behind' point A
        if np.dot(-a, b - a) < 0:
            return a
        # If the origin is 'ahead' of point B
        if np.dot(-b, a - b) < 0:
            return b

        ab = b - a  # Vector from A to B
        # Compute the projection of vector AO onto AB
        proj = np.dot(-a, ab) / np.dot(ab, ab) * ab
        # The closest point
        return a + proj







