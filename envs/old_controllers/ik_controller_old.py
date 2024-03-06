# import time
#
# import cma
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from tinyfk import KinematicModel, RotationType
#
# # from distance3d import gjk, colliders
#
# # Transformation matrices given by DH-parameters (Craig's convention)
# # dh = [
# #     (0,       0.333, 0,        q[0]),      # Joint 1
# #     (0,       0,     -np.pi/2, q[1]),      # Joint 2
# #     (0,       0.316, np.pi/2,  q[2]),      # Joint 3
# #     (0.0825,  0,     np.pi/2,  q[3]),      # Joint 4
# #     (-0.0825, 0.384, -np.pi/2, q[4]),      # Joint 5
# #     (0,       0,     np.pi/2,  q[5]),      # Joint 6
# #     (0.088,   0,     np.pi/2,  q[6]),      # Joint 7
# #     (0,       0.207, 0,        0)          # Flange (the original is 0.107, but we want the mid-point of the gripper 0.1 below)
# # We will write them explicitly to improve performance
# TRANSFORM = [
#     # Joint 1
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0],
#         [np.sin(theta), np.cos(theta), 0, 0],
#         [0, 0, 1, 0.333],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 2
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0],
#         [0, 0, 1, 0],
#         [-np.sin(theta), -np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 3
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0],
#         [0, 0, -1, -0.316],
#         [np.sin(theta), np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 4
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0.0825],
#         [0, 0, -1, 0],
#         [np.sin(theta), np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 5
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, -0.0825],
#         [0, 0, 1, 0.384],
#         [-np.sin(theta), -np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 6
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0],
#         [0, 0, -1, 0],
#         [np.sin(theta), np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Joint 7
#     lambda theta: np.array([
#         [np.cos(theta), -np.sin(theta), 0, 0.088],
#         [0, 0, -1, 0],
#         [np.sin(theta), np.cos(theta), 0, 0],
#         [0, 0, 0, 1]
#     ]),
#     # Flange
#     np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0.207],
#         [0, 0, 0, 1]
#     ]),
# ]
#
# class IKController:
#     def __init__(self, T_wb, q, obstacles):
#         """Inverse Kinematic Controller.
#
#          The input of the controller are a target cartesian position and quaternion orientation for
#          the frame of the end-effector.
#          The controller will output angular displacements in the joint space to achieve the desired
#          end-effector configuration.
#
#          The controller solves the nonlinear constrained optimization problem of reaching the desired end-effector
#          configuration under the constraints of minimal joint displacements and collision avoidance.
#
#          Args:
#              T_wb: Transformation from world frame to the robot base
#              q: starting joint angles
#              obstacles: starting obstacle positions
#         """
#         self.T_wb = T_wb
#         self.current_q = q
#         self.current_obstacles = obstacles
#         self.calc_dist = True
#         # list of transformation matrices from forward kinematics
#         self.fk_list = self.forward_kinematics(self.current_q)
#         # capsule positions
#         self.robot_capsules, self.obst_capsules = self.get_capsule_pos(self.fk_list, self.current_obstacles)
#         # init
#         self.target_pos = np.empty(3)
#         self.target_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#         # weights
#         self.alpha = 0.52    # position error
#         self.beta = 0.38     # orientation error
#         self.gamma = 0.01    # joint movement error
#         # joint movement constraint
#         self.epsilon = 0.1
#         # joint boundaries
#         self.jnt_bounds_scipy = (
#             (-2.8973, 2.8973),
#             (-1.7628, 1.7628),
#             (-2.8973, 2.8973),
#             (-3.0718, -0.0698),
#             (-2.8973, 2.8973),
#             (-0.0175, 3.7525),
#             (-2.8973, 2.8973)
#         )
#         self.jnt_bounds_cma = [
#             [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
#             [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
#         ]
#
#         self.hyper_dict = {}
#
#         self.sigs = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
#         self.ngens = (3, 5, 10, 30)
#         self.pops = (5, 15, 30, 50, 100, 200)
#
#         # self.alphas = (0.3, 0.5, 0.7)
#         # self.betas = (0.3, 0.5, 0.7)
#         # self.gammas = (0.005, 0.01, 0.05)
#
#         for sig in self.sigs:
#             for ngen in self.ngens:
#                 for pop in self.pops:
#                     self.hyper_dict[('sig: ' + str(sig), 'ngen: ' + str(ngen), 'pop: ' + str(pop))] = \
#                         {'time': 0., 'error': 0.}
#
#         # for alph in self.alphas:
#         #     for bet in self.betas:
#         #         for gam in self.gammas:
#         #             self.hyper_dict[('alpha: ' + str(alph), 'beta: ' + str(bet), 'gamma: ' + str(gam))] = \
#         #                 {'time': 0., 'error': 0.}
#
#     def solve(self, q, obstacles, target_pos, grip_pos, mocap_sol):
#         self.current_q = q.copy()
#         self.current_obstacles = obstacles
#         self.target_pos = target_pos
#
#         self.total_time_fcl = 0
#         self.total_time_std = 0
#         self.total_time_fk = 0
#         self.total_time_cpp = 0
#
#         kin = KinematicModel("/home/fabioubu/Documents/gitlocal/RL-Dyn-Env/envs/assets/Panda/panda.urdf")
#
#         joint_names = [
#             "panda_joint1",
#             "panda_joint2",
#             "panda_joint3",
#             "panda_joint4",
#             "panda_joint5",
#             "panda_joint6",
#             "panda_joint7",
#         ]
#
#         joint_ids = kin.get_joint_ids(joint_names)
#         end_link_id = kin.get_link_ids(["panda_link8"])[0]
#
#         # objective function
#         def obj_fun(q):
#             # calculate forward kinematics
#             fk_list = self.forward_kinematics(q)
#             # get capsule positions
#             # self.robot_capsules, self.obst_capsules = self.get_capsule_pos(self.fk_list, self.current_obstacles)
#             # target error - we will calculate the error directly on the rotation matrix to save some time.
#             current_pos = fk_list[-1][:3, 3]
#             current_rot = fk_list[-1][:3, :3]
#             pos_error = np.linalg.norm(current_pos - self.target_pos)
#             orientation_error = np.arccos(np.clip((np.trace(current_rot.T@self.target_rot)-1)/2, -1, 1))
#             angle_error = np.linalg.norm(q - self.current_q)
#
#             error = self.alpha * pos_error + self.beta * orientation_error + self.gamma * angle_error
#             return error
#
#         def obj_func_vectorized_tiny(q):
#
#             pos = []
#             rot = []
#             for q_ in q:
#                 res, _ = kin.solve_fk(
#                     q_,
#                     end_link_id,
#                     joint_ids,
#                     rot_type=RotationType.XYZW,
#                 )
#                 pos.append(res[0][:3])
#                 rot.append(res[0][3:])
#
#             pos_error = np.linalg.norm(np.array(pos) - self.target_pos, axis=1)
#
#             orientation_error = np.linalg.norm(np.array(rot) - np.array([1, 0, 0, 0]), axis=1)
#             angle_error = np.linalg.norm(q - self.current_q, axis=1)
#
#             error = self.alpha * pos_error + self.beta * orientation_error + self.gamma * angle_error
#
#             return error
#
#         def obj_func_vectorized(q):
#             batch_size = len(q)
#             q = np.array(q)
#             fk = np.repeat(np.eye(4)[None, :], batch_size, axis=0)
#
#             # Add robot base displacement
#             fk[:, :3, 3] += self.T_wb
#
#             for i in range(len(q[0])):  # For each joint
#                 fk = fk @ np.array([TRANSFORM[i](t) for t in q[:, i]])
#
#             # For flange without q
#             fk = fk @ TRANSFORM[-1]
#
#             pos_error = np.linalg.norm(fk[:, :3, 3] - self.target_pos, axis=1)
#             traces = np.trace(np.transpose(fk[:, :3, :3], axes=(0, 2, 1)) @ self.target_rot, axis1=1, axis2=2)
#             orientation_error = np.arccos(np.clip((traces - 1) / 2, -1, 1))
#             angle_error = np.linalg.norm(q - self.current_q, axis=1)
#
#             error = self.alpha * pos_error + self.beta * orientation_error + self.gamma * angle_error
#
#             return error
#
#         # The distance calculation is quite costly, so we will first see if the minimum distance is below a certain
#         # threshold and only then include it in the ik calculation.
#         self.calc_dist = True
#         dist = self.dist_constraint(q)
#         if dist >= 0.02:
#             self.calc_dist = False
#
#         # ######## pycma
#         # opts = cma.CMAOptions()
#         # opts['tolfun'] = 0.005
#         # # opts['maxfevals'] = 300
#         # # opts['bounds'] = self.jnt_bounds_cma
#         # opts['verbose'] = -9
#         # opts['verb_disp'] = 1000000000
#         # st = time.perf_counter()
#         # constraint_cma = lambda x: np.array([-self.dist_constraint(x)]*2)
#         # cfun = cma.ConstrainedFitnessAL(obj_fun, constraint_cma)
#         #
#         # es = cma.CMAEvolutionStrategy(self.current_q, 0.1, inopts=opts)
#         #
#         # while not es.stop():
#         #     X = es.ask()
#         #     es.tell(X, self.obj_func_vectorized(X))
#         #     # es.tell(X, [cfun(x) for x in X])
#         #     cfun.update(es)
#         # q_res_cma = cfun.find_feasible(es)
#         # time_old = time.perf_counter()-st
#         # f_old = obj_fun(q_res_cma)
#         # iter_old = es.countiter
#         # eval_old = es.countevals
#         # print('CMA-ES Elapsed Time: ', time_old)
#         # print('f-Value: ', obj_fun(q_res_cma))
#         # print('Total Iterations: ', es.countiter)
#         # print('Total function evaluations: ', es.countevals)
#         # # print('Calculated own distance: ', self.dist_constraint(q_res_cma))
#         # print('--------------------------')
#         # ######## pycma      ASK & TELL
#         opts = cma.CMAOptions()
#         opts['tolfun'] = 0.02
#         opts['verbose'] = -9
#         opts['verb_disp'] = 1000000000
#         opts['popsize'] = 18
#
#         constraint_cma = lambda x: np.array([-self.dist_constraint(x)]*2)
#         cfun = cma.ConstrainedFitnessAL(obj_fun, constraint_cma)
#
#         es = cma.CMAEvolutionStrategy(self.current_q, 0.08, inopts=opts, )
#
#         st = time.perf_counter()
#         while not es.stop():
#             X = es.ask()
#             es.tell(X, obj_func_vectorized_tiny(X))
#         # print('AskTell Time: ', time.perf_counter()-st)
#
#         if self.calc_dist:
#             st = time.perf_counter()
#             q_res_cma = cfun.find_feasible(es)
#             # print('Cfun Time: ', time.perf_counter()-st)
#         else:
#             q_res_cma = es.best.x
#
#         print('----------PYCMA----------------')
#         print('Elapsed Time: ', time.perf_counter()-st)
#         print('f-Value: ', obj_fun(q_res_cma))
#         print('Total Iterations: ', es.countiter)
#         print('Total function evaluations: ', es.countevals)
#         print('Calculated distance: ', self.dist_constraint(q_res_cma))
#         print('--------------------------')
#
#
#         ############### DEAP #################
#
#         from deap import base, creator, tools
#         from deap import cma as dcma
#
#         # hypertune
#         # for sig in self.sigs:
#         #     for ngen in self.ngens:
#         #         for pop in self.pops:
#
#         # for alph in self.alphas:
#         #     for bet in self.betas:
#         #         for gam in self.gammas:
#
#         sig = 0.08
#         ngen = 30
#         pop = 18
#         #
#         # self.alpha = alph
#         # self.beta = bet
#         # self.gamma = gam
#
#         creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#         creator.create("Individual", list, fitness=creator.FitnessMin)
#
#         toolbox = base.Toolbox()
#
#         cma_es = dcma.Strategy(centroid=self.current_q, sigma=sig, lambda_=pop)
#         toolbox.register("generate", cma_es.generate, creator.Individual)
#         toolbox.register("update", cma_es.update)
#
#         hof = tools.HallOfFame(1)
#
#         stats = tools.Statistics(lambda ind: ind.fitness.values)
#         stats.register("avg", np.mean)
#         stats.register("std", np.std)
#         stats.register("min", np.min)
#         stats.register("max", np.max)
#
#         logbook = tools.Logbook()
#         logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
#
#         cached_pop = []
#         st = time.perf_counter()
#         for gen in range(ngen):
#             # Generate a new population
#             population = toolbox.generate()
#             # Evaluate the individuals
#             fitnesses = obj_func_vectorized_tiny([np.array(p) for p in population])
#
#             for ind, fit in zip(population, fitnesses):
#                 ind.fitness.values = (fit, )
#
#             cached_pop += list(zip(population, fitnesses))
#
#             hof.update(population)
#
#             # Update the strategy with the evaluated individuals
#             toolbox.update(population)
#
#             record = stats.compile(population) if stats is not None else {}
#             logbook.record(gen=gen, nevals=len(population), **record)
#
#         # self.hyper_dict[('sig: ' + str(sig), 'ngen: ' + str(ngen), 'pop: ' + str(pop))]['time'] += time.perf_counter() - st
#         # self.hyper_dict[('sig: ' + str(sig), 'ngen: ' + str(ngen), 'pop: ' + str(pop))]['error'] += hof[0].fitness.values[0]
#         # self.hyper_dict[('alpha: ' + str(alph), 'beta: ' + str(bet), 'gamma: ' + str(gam))]['time'] += time.perf_counter() - st
#         # self.hyper_dict[('alpha: ' + str(alph), 'beta: ' + str(bet), 'gamma: ' + str(gam))]['error'] += hof[0].fitness.values[0]
#
#             #
#             # pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=iter_t, stats=stats,
#             #                                            halloffame=hof, verbose=False)
#
#             cached_pop.sort(key=lambda x: x[1])
#
#         q_res_cma = hof[0]
#         i = 0
#         for ind, fit in cached_pop:
#             i += 1
#             if self.dist_constraint(ind) > 0:
#                 q_res_feasible = ind
#                 break
#
#         q_res_cma = q_res_feasible
#
#         print('------------DEAP--------------')
#         print('Elapsed Time: ', time.perf_counter()-st)
#         # print('Resamples: ', i)
#         print('f-Value: ', hof[0].fitness.values[0])
#         print('Total Iterations: ', ngen)
#         print('Total function evaluations: ', ngen * 18)
#         print('Calculated distance: ', self.dist_constraint(hof[0]))
#         print('--------------------------')
#
#         fk_list = self.forward_kinematics(q_res_cma)
#         # get capsule positions
#         robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, self.current_obstacles)
#
#         return q_res_cma, robot_capsules, obst_capsules
#
#     def joint_angle_constraint(self, q):
#         return self.epsilon - np.linalg.norm(q - self.current_q)
#
#     def forward_kinematics(self, q):
#         """
#         Calculates the forward kinematics of the robot for each link, given its angle.
#
#         Args:
#             q (np.ndarray): array with joint angles
#
#         Return:
#             fk_list (dict): a list with the resulting transformation for each joint.
#         """
#         fk = np.eye(4)
#         # Add robot base displacement
#         fk[:3, 3] += self.T_wb
#         fk_list = [fk]
#         for i, q_i in enumerate(q):
#             fk = fk @ TRANSFORM[i](q_i)
#             fk_list.append(fk)
#         # for flange without q
#         fk = fk @ TRANSFORM[-1]
#         fk_list.append(fk)
#
#         return fk_list
#
#     @staticmethod
#     def capsule_params(start, end, radius):
#         # Compute normalized rotation vector
#         vec = end - start
#         height = np.linalg.norm(vec)
#         direction = vec / height
#
#         # Compute the rotation matrix to align the z-axis with the direction
#         z_axis = np.array([0, 0, 1])
#         axis = np.cross(z_axis, direction)
#         axis_norm = np.linalg.norm(axis)
#
#         # If axis and direction are nearly aligned, axis_norm will be close to zero
#         # In that case, we can skip the rotation
#         if axis_norm > 1e-8:
#             axis /= axis_norm
#             angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
#             rotation = R.from_rotvec(axis * angle)
#             rotation_matrix = rotation.as_matrix()
#         else:
#             rotation_matrix = np.eye(3)
#
#         # Combine the position and rotation into a 4x4 transformation matrix
#         pose = np.eye(4)
#         pose[:3, :3] = rotation_matrix
#         pose[:3, 3] = (np.array(start) + np.array(end)) / 2
#
#         return pose, radius, height
#
#     def get_capsule_pos(self, fk_list, obstacles):
#         """
#         Calculates the desired capsule positions of the robot and obstacles, based on the forward kinematics and
#         obstacle size/positions.
#
#         The forward kinematics gives us the coordinate frames of the franka robot
#         (cf. https://frankaemika.github.io/docs/control_parameters.html)
#         We will use those frames to create geometric capsules that encapsulate the robot arm. We want to use as few
#         capsules as necessary to encapsulate the robot as tight as possible.
#
#         We will use the 7 coordinate frames resulting from the forward kinematics to simplify the encapsulation process.
#         Capsules are usually defined by two 3D points from the start to end position and a scalar radius.
#
#         The capsule positions and sizes here were chosen specifically for the Franka Emika Panda Robot.
#
#         For more information about capsules and the distance calculation refer to
#         "Efficient Calculation of Minimum Distance Between Capsules and Its Use in Robotics, 2019"
#
#         Args:
#             fk_list:
#             obstacles
#         Returns:
#
#         """
#         # we can skip the robot base (up to the second coordinate frame) as it can't be moved to dodge obstacles.
#         robot_capsules = []
#         # 1: Shoulder (2. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[2], np.array([0, 0, -0.075, 1]))[:3],
#         #     'u': np.dot(fk_list[2], np.array([0, 0, 0.075, 1]))[:3],l
#         #     'roh': 0.065
#         # })
#         # # 2: Upper Arm 1 (2. & 3. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[2], np.array([0, 0, 0.08, 1]))[:3],
#         #     'u': np.dot(fk_list[3], np.array([0, 0, -0.18, 1]))[:3],
#         #     'roh': 0.065
#         # })
#         # 3: Upper Arm 2 (3. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[3], np.array([0, 0, -0.07, 1]))[:3],
#         #     'u': np.dot(fk_list[3], np.array([0, 0, -0.18, 1]))[:3],
#         #     'roh': 0.065
#         # })
#         # 4: Elbow (4. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[4], np.array([0, 0, -0.055, 1]))[:3],
#             'u': np.dot(fk_list[4], np.array([0, 0, 0.055, 1]))[:3],
#             'roh': 0.07
#         })
#         # 5: Forearm 1 (5. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[5], np.array([0, 0, -0.23, 1]))[:3],
#             'u': np.dot(fk_list[5], np.array([0, 0, -0.32, 1]))[:3],
#             'roh': 0.065
#         })
#         # # 6: Forearm 2 (5. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[5], np.array([0, 0, -0.215, 1]))[:3],
#         #     'u': np.dot(fk_list[5], np.array([0, 0.05, -0.18, 1]))[:3],
#         #     'roh': 0.057
#         # })
#         # 7: Forearm 3 (5. & 6. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[5], np.array([0, 0.07, -0.18, 1]))[:3],
#             'u': np.dot(fk_list[6], np.array([0, 0, -0.1, 1]))[:3],
#             'roh': 0.04
#         })
#         # 8: Wrist (6. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[6], np.array([0, 0, -0.08, 1]))[:3],
#             'u': np.dot(fk_list[6], np.array([0, 0, 0.01, 1]))[:3],
#             'roh': 0.062
#         })
#         # 9: Hand 1 (6. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[6], np.array([0, 0, 0.02, 1]))[:3],
#         #     'u': np.dot(fk_list[6], np.array([0.05, 0.02, 0, 1]))[:3],
#         #     'roh': 0.06
#         # })
#         # 10: Hand 2 (7. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[7], np.array([0, 0, -0.04, 1]))[:3],
#             'u': np.dot(fk_list[7], np.array([0, 0, 0.175, 1]))[:3],
#             'roh': 0.06
#         })
#         # 11: Hand 3 (7. Frame)
#         # robot_capsules.append({
#         #     'p': np.dot(fk_list[7], np.array([0.03, 0.06, 0.085, 1]))[:3],
#         #     'u': np.dot(fk_list[7], np.array([0.06, 0.03, 0.085, 1]))[:3],
#         #     'roh': 0.03
#         # })
#         # 12: Hand 4 (7. Frame)
#         robot_capsules.append({
#             'p': np.dot(fk_list[7], np.array([0, 0.061, 0.13, 1]))[:3],
#             'u': np.dot(fk_list[7], np.array([0, -0.061, 0.13, 1]))[:3],
#             'roh': 0.06
#         })
#
#         # Obstacles (we will use the cuboids largest dimension capsule height and second largest as radius)
#         obst_capsules = []
#         for obst in obstacles:
#             # sorted dimensions
#             dims = np.argsort(obst['size'])
#             l = obst['size'][dims[-1]]
#             p = obst['pos'].copy()
#             p[dims[-1]] += l
#             u = obst['pos'].copy()
#             u[dims[-1]] -= l
#             obst_capsules.append({
#                 'p': p,
#                 'u': u,
#                 'roh': np.sqrt(obst['size'][dims[0]]**2 + obst['size'][dims[1]]**2) + 0.01      # safe space
#             })
#
#         return robot_capsules, obst_capsules
#
#     # def gjk_inter_constraint(self, q):
#     #     # calculate forward kinematics
#     #     fk_list = self.forward_kinematics(q)
#     #     # get capsule positions
#     #     robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, self.current_obstacles)
#     #     # transform to gjk parametrization
#     #     robot_capsules = [self.capsule_params(caps['p'], caps['u'], caps['roh']) for caps in robot_capsules]
#     #     obst_capsules = [self.capsule_params(caps['p'], caps['u'], caps['roh']) for caps in obst_capsules]
#     #
#     #     return self.gjk_get_intersection(robot_capsules, obst_capsules)
#
#     # def gjk_get_intersection(self, robot_capsules, obst_capsules):
#     #     distances = []
#     #     for n, r_caps in enumerate(robot_capsules):
#     #         for m, o_caps in enumerate(obst_capsules):
#     #             c1 = colliders.Capsule(*r_caps)
#     #             c2 = colliders.Capsule(*o_caps)
#     #             res = gjk.gjk(c1, c2)
#     #             distances.append(res[0])
#     #     return np.min(distances)
#
#     def dist_constraint(self, q):
#         if not self.calc_dist:
#             return 0.1
#
#         # st = time.perf_counter()
#
#         # calculate forward kinematics
#         fk_list = self.forward_kinematics(q)
#
#         # get capsule positions
#         robot_capsules, obst_capsules = self.get_capsule_pos(fk_list, self.current_obstacles)
#
#         # self.total_time_fk += time.perf_counter() - st
#         # st = time.perf_counter()
#
#         # ############## TEST FCL ###############
#         #
#         # robot_capsules_fcl = [self.capsule_params(caps['p'], caps['u'], caps['roh']) for caps in robot_capsules]
#         # obst_capsules_fcl = [self.capsule_params(caps['p'], caps['u'], caps['roh']) for caps in obst_capsules]
#         #
#         # r_capsules_fcl = []
#         # for caps in robot_capsules_fcl:
#         #     capsule = hppfcl.Capsule(caps[1], caps[2])
#         #     pose = hppfcl.Transform3f(caps[0][:3, :3], caps[0][:3, 3])
#         #     r_capsules_fcl.append(hppfcl.CollisionObject(capsule, pose))
#         #
#         # o_capsules_fcl = []
#         # for caps in obst_capsules_fcl:
#         #     capsule = hppfcl.Capsule(caps[1], caps[2]/2)
#         #     pose = hppfcl.Transform3f(caps[0][:3, :3], caps[0][:3, 3])
#         #     o_capsules_fcl.append(hppfcl.CollisionObject(capsule, pose))
#         #
#         # min_distance = float('inf')
#         # for r_obj in r_capsules_fcl:
#         #     for o_obj in o_capsules_fcl:
#         #         request = hppfcl.DistanceRequest()  # Create a distance request object
#         #         result = hppfcl.DistanceResult()  # Create a result object to store the result
#         #
#         #         hppfcl.distance(r_obj, o_obj, request, result)
#         #
#         #         # Update the minimum distance if necessary
#         #         if result.min_distance < min_distance:
#         #             min_distance = result.min_distance
#         #
#         # self.total_time_fcl += time.perf_counter() - st
#         #
#         # st = time.perf_counter()
#         #
#         # res1 = self.calc_min_dist(robot_capsules, obst_capsules)
#
#         # self.total_time_std += time.perf_counter() - st
#
#         ################### TEST CPP ######################
#
#         st = time.perf_counter()
#
#         res = self.calc_min_dist(robot_capsules, obst_capsules)
#
#         self.total_time_cpp += time.perf_counter() - st
#         self.total_time_fk += 1
#
#         ###################################################
#
#         return res
#
#     def calc_min_dist(self, robot_capsules, obst_capsules):
#         """
#         Computes the minimum distance between all capsules of the robot arm with all obstacle capsules.
#
#         Returns:
#             min_dist: The minimum distance from all comparisons
#         """
#         distances = []
#         for n, r_caps in enumerate(robot_capsules):
#             for m, o_caps in enumerate(obst_capsules):
#                 #
#                 p1 = r_caps['p']
#                 p2 = o_caps['p']
#                 s1 = r_caps['u'] - p1
#                 s2 = o_caps['u'] - p2
#
#                 A = np.stack([s2, -s1], 1)
#                 y = p2-p1
#                 Q, R = np.linalg.qr(A)
#
#                 u = lambda x: np.dot(R, x) + np.dot(Q.T, y)
#
#                 # check whether the origin is inside the parallelogram given by u. If it is inside, we can abort the calculation
#                 # and return that a collision is imminent. repeat the first point to create a closed loop.
#                 vertices = [u([0,0]), u([0,1]), u([1, 1]), u([1, 0])]
#
#                 sides = []
#                 u_temp = []
#                 for i in range(4):
#                     # o = np.array([0, 0])
#                     v1 = vertices[i]
#                     v2 = vertices[i+1] if i != 3 else vertices[0]
#                     # for each edge of the parallelogram, check on which side the point lies. If all queries return the same
#                     # result, the origin lies inside the parallelogram.
#                     res = -v1[1] * (v2[0] - v1[0]) - (-v1[0]) * (v2[1] - v1[1])
#                     if res >= 0:
#                         sides.append(1)
#                     else:
#                         sides.append(-1)
#                     # for each egde we also calculate the closest point to the origin
#                     u_temp.append(self.closest_point_on_segment(v1, v2))
#
#                 if abs(sum(sides)) == 4:
#                     u_min = np.array([0, 0])
#                 else:
#                     u_min = u_temp[np.argmin(np.linalg.norm(u_temp, axis=1))]
#
#                 # now calculate the distance
#                 dist = np.sqrt(u_min.T @ u_min + y.T @ y - y.T @ Q @ Q.T @ y) - r_caps['roh'] - o_caps['roh']
#                 distances.append(dist)
#         return np.min(distances)
#
#     @staticmethod
#     def closest_point_on_segment(a, b):
#         # If the origin is 'behind' point A
#         if np.dot(-a, b - a) < 0:
#             return a
#         # If the origin is 'ahead' of point B
#         if np.dot(-b, a - b) < 0:
#             return b
#
#         ab = b - a  # Vector from A to B
#         # Compute the projection of vector AO onto AB
#         proj = np.dot(-a, ab) / np.dot(ab, ab) * ab
#         # The closest point
#         return a + proj
#
#     @staticmethod
#     def trans_mat(a, d, alpha, theta):
#         # transformation matrix by Craig's convention
#         return np.array([
#             [np.cos(theta), -np.sin(theta), 0, a],
#             [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
#             [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
#             [0, 0, 0, 1]
#         ])
#
#
#
#
# ########### ARCHIVE
#         # self.calc_dist = True
#         # dist = self.calc_min_distance(q)
#         # if dist >= 0.01:
#         #     self.calc_dist = False
#         # print('--------------------------')
#         # print('Starting f-Value: ', obj_fun(self.current_q))
#         # constraint_dict = {'type': 'ineq', 'fun': self.calc_min_distance}
#         # constraint = spo.NonlinearConstraint(fun=self.calc_min_distance, lb=0, ub=np.inf)
#         # #
#         # ###### scipy SLSQP
#         # print('--------------------------')
#         # st = time.perf_counter()
#         # # noinspection PyTypeChecker
#         # res = spo.minimize(
#         #     fun=obj_fun,
#         #     x0=self.current_q,
#         #     method='SLSQP',
#         #     bounds=self.jnt_bounds_scipy,
#         #     constraints=constraint,
#         #     tol=0.2
#         # )
#         # q_res_scipy = res.x
#         # print('SLSQP Elapsed Time: ', time.perf_counter()-st)
#         # print('f-Value: ', obj_fun(q_res_scipy))
#         # print('Total function evaluations: ', res.nfev)
#         # print('--------------------------')
#         #
#         # ###### scipy DE
#         # st = time.perf_counter()
#         # # noinspection PyTypeChecker
#         # res = spo.differential_evolution(
#         #     func=obj_fun,
#         #     x0=self.current_q,
#         #     bounds=self.jnt_bounds_scipy,
#         #     constraints=constraint,
#         #     tol=0.3
#         #
#         # )
#         # q_res_de = res.x
#         # print('DE Elapsed Time: ', time.perf_counter()-st)
#         # print('f-Value: ', obj_fun(q_res_de))
#         # print('Total function evaluations: ', res.nfev)
#         # print('--------------------------')
#         #
#         ########### SHGO
#         # st = time.perf_counter()
#         # # noinspection PyTypeChecker
#         # res = spo.shgo(
#         #     func=obj_fun,
#         #     sampling_method='simplicial',
#         #     bounds=self.jnt_bounds_scipy,
#         #     constraints=constraint_dict,
#         #     options={'maxtime': 0.01},
#         # )
#         # q_res_shgo = res.x
#         # print('SHGO Elapsed Time: ', time.perf_counter()-st)
#         # print('f-Value: ', obj_fun(q_res_shgo))
#         # print('Total function evaluations: ', res.nfev)
#         # print('Calculated distance: ', self.calc_min_distance(q_res_shgo))
#         # print('--------------------------')
#         #
#         #
#         # ############## PSO
#         # constraint_pso = lambda x: np.array([self.calc_min_distance(x)])
#         # st = time.perf_counter()
#         # res_pso = pso(
#         #     func= obj_fun,
#         #     lb= self.jnt_bounds_cma[0],
#         #     ub= self.jnt_bounds_cma[1],
#         #     f_ieqcons=constraint_pso
#         # )
#         # q_res_pso = res_pso[0]
#         # print('PSO Elapsed Time: ', time.perf_counter() - st)
#         # print('f-Value: ', res_pso[1])
#         # print('Total function evaluations: ',)
#         # print('--------------------------')
#
#         # constraint_pso_sko = (lambda x: -self.calc_min_distance(x), )
#         # st = time.perf_counter()
#         # pop_size = 40
#         # sigma = 0.5
#         # sko_pso = PSO(
#         #     func= obj_fun,
#         #     n_dim=7,
#         #     pop=pop_size,
#         #     max_iter=30,
#         #     lb= self.jnt_bounds_cma[0],
#         #     ub= self.jnt_bounds_cma[1],
#         #     constraint_ueq=constraint_pso_sko
#         # )
#         # sko_pso.X = np.random.uniform(
#         #     low=np.max([self.jnt_bounds_cma[0] , self.current_q - sigma], axis=0),
#         #     high=np.min([self.jnt_bounds_cma[1] , self.current_q + sigma], axis=0),
#         #     size=(pop_size, 7)
#         # )
#         # sko_pso.V = np.random.uniform(low=-sigma, high=sigma, size=(pop_size, 7))  # speed of particles
#         # sko_pso.Y = sko_pso.cal_y()  # y = f(x) for all particles
#         # sko_pso.run()
#         # q_res = sko_pso.gbest_x
#         # print('SKO- PSO Elapsed Time: ', time.perf_counter() - st)
#         # print('f-Value: ', sko_pso.gbest_y[0])
#         # print('Total function evaluations: ',)
#         # print('--------------------------')
#
#         ########
#
#         # q_prev = self.current_q.copy()
#         # mocap = mocap_sol.copy()
#         # # test
#         # grip = grip_pos.copy()
#         #
#         # target = self.target_pos.copy()
#         # current_pos = self.forward_kinematics(q_prev)[-1][:3,3]
#         #
#         # calc_pos = self.forward_kinematics(q_res)[-1][:3, 3]
#         #
#         # current_rot = self.forward_kinematics(q_res)[-1][:3, :3]
#         # pos_e = np.linalg.norm(calc_pos - self.target_pos)
#         # m_pos_e = np.linalg.norm(self.forward_kinematics(mocap)[-1][:3,3] - self.target_pos)
#         #
#         # orient_e = np.arccos((np.trace(current_rot.T@self.target_rot)-1)/2)
#         # # m_orient_e = np.arccos((np.trace(self.forward_kinematics(mocap)[-1][:3,:3].T@self.target_rot)-1)/2)
#         #
#         # angle_e = np.linalg.norm(q_res - self.current_q)
#         # m_angle_e = np.linalg.norm(mocap - self.current_q)
#
#         # self.nit += res.nit
#         # self.nfev += res.nfev
#         # print(f'Solved in {res.nit} iterations with {res.nfev} function evaluations')
#         # print('Position Error: ', self.alpha * pos_e)
#         # print('Orientation Error: ', self.beta * orient_e)
#         # print('Angle Error: ',self.gamma * angle_e)
#         # print('Distance: ', dist)
#         # print('--------------------------')
#         # constraint_cma = lambda x: np.array([-self.calc_min_distance(x)]*2)
#         # cfun = cma.ConstrainedFitnessAL(obj_fun, constraint_cma)
#         #
#         # st = time.perf_counter()
#         # q_res_cma, es = cma.fmin2(cfun, self.current_q, 0.1, options=opts, callback=cfun.update)
#         # q_res_cma = cfun.find_feasible(es)
#         # print('CMA-ES Elapsed Time: ', time.perf_counter()-st)
#         # print('f-Value: ', obj_fun(q_res_cma))
#         # print('Total function evaluations: ', es.countevals)
#         # print('Calculated distance: ', self.calc_min_distance(q_res_cma))
#         # print('--------------------------')