import time

import numpy as np

from ik_controller_bfgs.wrapper import solve_ik

#bfgs_c++

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
        self.alpha = 0.52    # position error
        self.beta = 0.38     # orientation error
        self.gamma = 0.01    # joint movement error

        # joint limits
        self.jnt_bounds = [
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        ]
        self.total_time_cpp = 0
        self.control_time = 0

    def solve(self, q, obstacles, target_pos):
        options = {
            "robot_base": self.T_wb,  # displacement of the robot base in world xyz coordinates
            "alpha": 0.7,  # weight for position error
            "beta": 0.2,  # weight for posture error
            "gamma": 0.1,  # weight for joint movement error

            "fDelta": 0,  # min change in cost function
            "xDelta": 0,  # min change in parameter vector
            "gradNorm": 0, #min norm og grad
            "maxiter": 10, #max iteration of bfgs

            "population": 50, #number of gaussian generated
            "sigma": 0.1,  # standard deviation of gaussian

            "skip": 1, # search with i = i + skip
            # tolerance of f value (error). Algorithm stops early once the mean improvement per generation is below this value.
            "bounds": self.jnt_bounds  # Joint limit constraints
        }

        st = time.perf_counter()

        q_res, f_res = solve_ik(q, obstacles, target_pos, options)

        self.control_time = time.perf_counter() - st

        fk_list = self.forward_kinematics(q_res)
        # get capsule positions
        robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, obstacles)

        return q_res, robot_capsules, obst_capsules

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
        i = 0
        for obst in obstacles:
            # sorted dimensions
            # for capsule visualization
            i = i + 1
            if i == 7:
                break
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


class CMA:
    """
    A strategy that will keep track of the basic parameters of the CMA-ES
    algorithm ([Hansen2001]_).

    :param centroid: An iterable object that indicates where to start the
                     evolution.
    :param sigma: The initial standard deviation of the distribution.
    :param lambda_: The initial population size.

    .. [Hansen2001] Hansen and Ostermeier, 2001. Completely Derandomized
       Self-Adaptation in Evolution Strategies. *Evolutionary Computation*

    """
    def __init__(self, centroid, sigma, lambda_):

        # Create a centroid as a numpy array
        self.centroid = np.array(centroid)

        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1. / (4. * self.dim) +
                                      1. / (21. * self.dim ** 2))

        self.C = np.identity(self.dim)
        self.diagD, self.B = np.linalg.eigh(self.C)

        indx = np.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.lambda_ = lambda_
        self.update_count = 0
        self.computeParams()

    def generate(self):
        r"""Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """
        arz = np.random.standard_normal((self.lambda_, self.dim))
        return self.centroid + self.sigma * np.dot(arz, self.BD.T)

    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of (individual, fitness) from which to update the
                           parameters.
        """
        population.sort(key=lambda x: x[1])

        individuals = np.array([pop[0] for pop in population])

        old_centroid = self.centroid
        self.centroid = np.dot(self.weights, individuals[0:self.mu])

        c_diff = self.centroid - old_centroid

        # Cumulation : update evolution path
        self.ps = (1 - self.cs) * self.ps \
            + np.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
            * np.dot(self.B, (1. / self.diagD) *
                        np.dot(self.B.T, c_diff))

        hsig = float((np.linalg.norm(self.ps) /
                      np.sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.))) / self.chiN <
                      (1.4 + 2. / (self.dim + 1.))))

        self.update_count += 1

        self.pc = (1 - self.cc) * self.pc + hsig \
            * np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
            * c_diff

        # Update covariance matrix
        artmp = individuals[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) *
                  self.ccov1 * self.cc * (2 - self.cc)) * self.C \
            + self.ccov1 * np.outer(self.pc, self.pc) \
            + self.ccovmu * np.dot((self.weights * artmp.T), artmp) \
            / self.sigma ** 2

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1.) * self.cs / self.damps)

        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self):
        r"""Computes the parameters depending on :math:`\lambda`. It needs to
        be called again if :math:`\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
        self.mu = int(self.lambda_ / 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))

        self.weights /= sum(self.weights)
        self.mueff = 1. / sum(self.weights ** 2)

        self.cc = 4. / (self.dim + 4.)
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 3.)
        self.ccov1 = 2. / ((self.dim + 1.3) ** 2 + self.mueff)
        self.ccovmu = 2. * (self.mueff - 2. + 1. / self.mueff) / ((self.dim + 2.) ** 2 + self.mueff)
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)
        self.damps = 1. + 2. * max(0, np.sqrt((self.mueff - 1.) / (self.dim + 1.)) - 1.) + self.cs