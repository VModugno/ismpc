import numpy as np
import casadi as cs
import copy

class Ismpc:
  def __init__(self, initial, footstep_planner, N=100, delta=0.01, g=9.81, h=0.75):
    # parameters
    self.N = N
    self.delta = delta
    self.eta = np.sqrt(g/h)
    self.step_height = 0.02
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.footstep_plan = self.footstep_planner.footstep_plan

    # lip model matrices
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # dynamics
    self.f = lambda x, u: cs.vertcat(
      self.A_lip @ x[:3] + self.B_lip @ u[0],
      self.A_lip @ x[3:] + self.B_lip @ u[1]
    )

    # optimization problem
    self.opt = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}
    self.opt.solver("osqp", p_opts, s_opts)

    self.U = self.opt.variable(2, N)
    self.X = self.opt.variable(6, N + 1)

    self.x0_param = self.opt.parameter(6)
    self.zmp_x_mid_param = self.opt.parameter(N)
    self.zmp_y_mid_param = self.opt.parameter(N)

    for i in range(N):
      self.opt.subject_to(self.X[:, i + 1] == self.X[:, i] + delta * self.f(self.X[:, i], self.U[:, i]))

    cost = cs.sumsqr(self.U[0, :]) + cs.sumsqr(self.U[1, :]) + \
           100 * cs.sumsqr(self.X[2, 1:].T - self.zmp_x_mid_param) + \
           100 * cs.sumsqr(self.X[5, 1:].T - self.zmp_y_mid_param)

    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + 0.1)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - 0.1)
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + 0.1)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - 0.1)

    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # stability constraint with periodic tail
    self.opt.subject_to(self.X[1, 0] + self.eta**3 * (self.X[0, 0] - self.X[2, 0]) == \
                        self.X[1, N] + self.eta**3 * (self.X[0, N] - self.X[2, N]))
    self.opt.subject_to(self.X[4, 0] + self.eta**3 * (self.X[3, 0] - self.X[5, 0]) == \
                        self.X[4, N] + self.eta**3 * (self.X[3, N] - self.X[5, N]))

    self.opt.minimize(cost)

    self.x = np.zeros(6)
    self.desired = copy.deepcopy(initial)

  def solve(self, current, t):
    self.x = np.array([current.com_position[0], current.com_velocity[0], self.desired.zmp_position[0],
                       current.com_position[1], current.com_velocity[1], self.desired.zmp_position[1]])

    # solve optimization problem
    self.opt.set_value(self.x0_param, self.x)
    for i in range(self.N):
      mc = self.generate_moving_constraint_at_time(t + i)
      self.opt.set_value(self.zmp_x_mid_param[i], mc[0])
      self.opt.set_value(self.zmp_y_mid_param[i], mc[1])

    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    feet_trajectories = self.generate_feet_trajectories_at_time(t)

    # create desired state
    self.desired_old = copy.deepcopy(self.desired)
    self.desired.com_position            = np.array([self.x[0], self.x[3], 0.75])
    self.desired.com_velocity            = np.array([self.x[1], self.x[4], 0.])
    self.desired.zmp_position            = np.array([self.x[2], self.x[5], 0.])
    self.desired.zmp_velocity            = np.hstack((self.u, 0.))
    self.desired.com_acceleration        = np.hstack((self.eta**2 * (self.desired.com_position[:2] - self.desired.zmp_position[:2]), 0.))
    self.desired.left_foot_pose          = feet_trajectories['left']['pos']
    self.desired.left_foot_velocity      = feet_trajectories['left']['vel']
    self.desired.left_foot_acceleration  = feet_trajectories['left']['acc']
    self.desired.right_foot_pose         = feet_trajectories['right']['pos']
    self.desired.right_foot_velocity     = feet_trajectories['right']['vel']
    self.desired.right_foot_acceleration = feet_trajectories['right']['acc']
    self.desired.torso_orientation          = (self.desired.left_foot_pose[:3]         + self.desired.right_foot_pose[:3])         / 2.
    self.desired.torso_angular_velocity     = (self.desired.left_foot_velocity[:3]     + self.desired.right_foot_velocity[:3])     / 2.
    self.desired.torso_angular_acceleration = (self.desired.left_foot_acceleration[:3] + self.desired.right_foot_acceleration[:3]) / 2.
    self.desired.base_orientation           = (self.desired.left_foot_pose[:3]         + self.desired.right_foot_pose[:3])         / 2.
    self.desired.base_angular_velocity      = (self.desired.left_foot_velocity[:3]     + self.desired.right_foot_velocity[:3])     / 2.
    self.desired.base_angular_acceleration  = (self.desired.left_foot_acceleration[:3] + self.desired.right_foot_acceleration[:3]) / 2.

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact += self.footstep_planner.footstep_plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.desired, contact
  
  def generate_feet_trajectories_at_time(self, time):
    step_index = self.footstep_planner.get_step_index_at_time(time)
    time_in_step = time - self.footstep_planner.get_start_time(step_index)
    phase = self.footstep_planner.get_phase_at_time(time)
    support_foot = self.footstep_planner.footstep_plan[step_index]['foot_id']
    swing_foot = 'left' if support_foot == 'right' else 'right'
    single_support_duration = self.footstep_planner.footstep_plan[step_index]['ss_duration']

    # if first step, return initial foot poses with zero velocities and accelerations
    if step_index == 0:
        zero_vel = np.zeros(6)
        zero_acc = np.zeros(6)
        return {
            'left': {
                'pos': self.initial.left_foot_pose,
                'vel': zero_vel,
                'acc': zero_acc
            },
            'right': {
                'pos': self.initial.right_foot_pose,
                'vel': zero_vel,
                'acc': zero_acc
            }
        }

    # if double support, return planned foot poses with zero velocities and accelerations
    if phase == 'ds':
        support_pose = np.hstack((
            self.footstep_plan[step_index]['ang'],
            self.footstep_plan[step_index]['pos']
        ))
        swing_pose = np.hstack((
            self.footstep_plan[step_index + 1]['ang'],
            self.footstep_plan[step_index + 1]['pos']
        ))
        zero_vel = np.zeros(6)
        zero_acc = np.zeros(6)
        return {
            support_foot: {
                'pos': support_pose,
                'vel': zero_vel,
                'acc': zero_acc
            },
            swing_foot: {
                'pos': swing_pose,
                'vel': zero_vel,
                'acc': zero_acc
            }
        }
    
    # get positions and angles for cubic interpolation
    start_pos  = self.footstep_plan[step_index - 1]['pos']
    target_pos = self.footstep_plan[step_index + 1]['pos']
    start_ang  = self.footstep_plan[step_index - 1]['ang']
    target_ang = self.footstep_plan[step_index + 1]['ang']

    # time variables
    t = time_in_step
    T = single_support_duration

    # cubic polynomial for position and angle
    A = - 2 / T**3
    B =   3 / T**2
    swing_pos     = start_pos + (target_pos - start_pos) * (    A * t**3 +     B * t**2)
    swing_vel     =             (target_pos - start_pos) * (3 * A * t**2 + 2 * B * t   ) / self.delta
    swing_acc     =             (target_pos - start_pos) * (6 * A * t    + 2 * B       ) / self.delta**2
    swing_ang_pos = start_ang + (target_ang - start_ang) * (    A * t**3 +     B * t**2)
    swing_ang_vel =             (target_ang - start_ang) * (3 * A * t**2 + 2 * B * t   ) / self.delta
    swing_ang_acc =             (target_ang - start_ang) * (6 * A * t    + 2 * B       ) / self.delta**2

    # quartic polynomial for vertical position
    A =   16 * self.step_height / T**4
    B = - 32 * self.step_height / T**3
    C =   16 * self.step_height / T**2
    swing_pos[2] =       A * t**4 +     B * t**3 +     C * t**2
    swing_vel[2] = ( 4 * A * t**3 + 3 * B * t**2 + 2 * C * t   ) / self.delta
    swing_acc[2] = (12 * A * t**2 + 6 * B * t    + 2 * C       ) / self.delta**2

    # support foot remains stationary
    support_pos = self.footstep_plan[step_index]['pos']
    support_ang = self.footstep_plan[step_index]['ang']
    zero_vel = np.zeros(3)
    zero_acc = np.zeros(3)

    # assemble pose, velocity, and acceleration for each foot
    support_data = {
        'pos': np.hstack((support_ang, support_pos)),
        'vel': np.hstack((np.zeros(3), zero_vel)),
        'acc': np.hstack((np.zeros(3), zero_acc))
    }

    swing_data = {
        'pos': np.hstack((swing_ang_pos, swing_pos)),
        'vel': np.hstack((swing_ang_vel, swing_vel)),
        'acc': np.hstack((swing_ang_acc, swing_acc))
    }

    return {
        support_foot: support_data,
        swing_foot: swing_data
    }

  def generate_moving_constraint_at_time(self, time):
    step_index = self.footstep_planner.get_step_index_at_time(time)
    time_in_step = time - self.footstep_planner.get_start_time(step_index)
    phase = self.footstep_planner.get_phase_at_time(time)
    single_support_duration = self.footstep_plan[step_index]['ss_duration']
    double_support_duration = self.footstep_plan[step_index]['ds_duration']

    if phase == 'ss':
      return self.footstep_plan[step_index]['pos']

    # linear interpolation for x and y coordinates of the foot positions during double support
    if step_index == 0: start_pos = (self.initial.left_foot_pose[3:] + self.initial.right_foot_pose[3:]) / 2.
    else:               start_pos = np.array(self.footstep_plan[step_index]['pos'])
    target_pos = np.array(self.footstep_plan[step_index + 1]['pos'])
    
    moving_constraint = start_pos + (target_pos - start_pos) * ((time_in_step - single_support_duration) / double_support_duration)
    return moving_constraint