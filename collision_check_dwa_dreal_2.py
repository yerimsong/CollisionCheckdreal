"""
Collision Checking with dreal and Dynamic Window Approach
"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import dreal

show_animation = True

def dwa_control(x, config, goal, ob, xs, x_i, reach_goal):
    """
    Dynamic Window Approach control
    """

    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob, xs, x_i, reach_goal)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.stop = 0.0 # [m/s]
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3  # [s]
        self.to_goal_cost_gain = 0.20
        self.speed_cost_gain = 3.0
        self.obstacle_cost_gain = 1.0
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yaw_rate min, yaw_rate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config, predict_time):
    """
    predict trajectory with an input
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj

def make_oth_trj_win(x, config, predict_time):
    cur_x = np.array(x)
    traj = np.array(cur_x)
    time = 0
    while time <= predict_time:
        cur_x = motion(cur_x, [cur_x[3], cur_x[2]], 1.0)
        traj = np.vstack((traj, cur_x))
        time += 1.0

    xmin = float(np.min(traj[:, 0]))
    xmax = float(np.max(traj[:, 0]))
    ymin = float(np.min(traj[:, 1]))
    ymax = float(np.max(traj[:, 1]))
    return [xmin, xmax, ymin, ymax]

def calc_control_and_trajectory(x, dw, config, goal, ob, xs, x_i, reach_goal):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    predict_time = config.predict_time
    while predict_time >= 0:
        min_cost = float("inf")
        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], config.v_reso):
            for y in np.arange(dw[2], dw[3], config.yawrate_reso):

                trajectory = predict_trajectory(x_init, v, y, config, predict_time)

                # calc cost
                to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config, xs, x_i, reach_goal, predict_time)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory

        # if current predict time results in unavoidable collision (infinite cost),
        # decrease predict time and check for collision again
        if min_cost == float("inf"):
            predict_time = predict_time - 1
        else:
            break

    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config, xs, x_i, reach_goal, predict_time):
    """
        calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    # calculate the distance between robots in order to determine cost
    ox2 = np.array([])
    oy2 = np.array([])
    for i in range(len(xs)):
        if i != x_i and reach_goal[i] != 1:
            ox2 = np.append(ox2, xs[i][0])
            oy2 = np.append(oy2, xs[i][1])
    dx2 = trajectory[:, 0] - ox2[:, None]
    dy2 = trajectory[:, 1] - oy2[:, None]
    r2 = np.hypot(dx2, dy2)

    # trajectory window for the current robot
    min_dx = float(np.min(trajectory[:, 0]))
    max_dx = float(np.max(trajectory[:, 0]))
    min_dy = float(np.min(trajectory[:, 1]))
    max_dy = float(np.max(trajectory[:, 1]))

    # did not consider collision among rectangle robots yet
    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")

    elif config.robot_type == RobotType.circle:
        if (r <= config.robot_radius).any() or (r2 <= config.robot_radius*2).any():
            return float("Inf")

        # collision checking with dreal
        for i in range(len(xs)):
            if i != x_i and reach_goal[i] != 1:

                oth_trj_win = make_oth_trj_win(xs[i], config, predict_time)

                my_x = dreal.Variable("my_x")
                my_y = dreal.Variable("my_y")
                oth_x = dreal.Variable("oth_x")
                oth_y = dreal.Variable("oth_y")
                f_sat = dreal.And(min_dx <= my_x, max_dx >= my_x,
                                  min_dy <= my_y, max_dy >= my_y,
                                  oth_trj_win[0] <= oth_x, oth_trj_win[1] >= oth_x,
                                  oth_trj_win[2] <= oth_y, oth_trj_win[3] >= oth_y,
                                  (my_x - oth_x) ** 2 + (my_y - oth_y) ** 2 <= 4)
                result = dreal.CheckSatisfiability(f_sat, 0.001)
                if result is not None:
                    return float("Inf")


    if r2.size == 0: # this means only one robot remains and collision checking among robots is unneccessary
        min_r = np.min([np.min(r) - config.robot_radius])
    else:
        min_r = np.min([np.min(r) - config.robot_radius, np.min(r2) - config.robot_radius * 2])
    return 1.0 / min_r


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = [np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0]),
         np.array([-3.0, 2.0, math.pi / 8.0, 0.0, 0.0]),
         np.array([-3.0, 5.0, math.pi / 8.0, 0.0, 0.0])
         ]
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])
    # obstacles [x(m) y(m), ....]
    ob = np.array([[-1, -1],
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [5.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 9.0],
                   [8.0, 10.0],
                   [9.0, 11.0],
                   [12.0, 13.0],
                   [12.0, 12.0],
                   [15.0, 15.0],
                   [13.0, 13.0]
                   ])

    config = Config()
    config.robot_type = robot_type
    trajectory = [0] * len(x)
    for i in range(len(x)):
        trajectory[i] = np.array(x[i])

    reach_goal = [0] * len(x)

    while True:
        u = [0] * len(x)
        predicted_trajectory = [0] * len(x)
        for i in range(len(x)):
            if reach_goal[i] == 1:
                continue
            u[i], predicted_trajectory[i] = dwa_control(x[i], config, goal, ob, x, i, reach_goal)
            x[i] = motion(x[i], u[i], config.dt)
            trajectory[i] = np.vstack((trajectory[i], x[i]))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(len(x)):
                if reach_goal[i] == 1:
                    continue
                plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-g")
            for i in range(len(x)):
                plt.plot(x[i][0], x[i][1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            for i in range(len(x)):
                plot_robot(x[i][0], x[i][1], x[i][2], config)
            for i in range(len(x)):
                plot_arrow(x[i][0], x[i][1], x[i][2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        for i in range(len(x)):
            dist_to_goal = math.hypot(x[i][0] - goal[0], x[i][1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print("Goal!!")
                reach_goal[i] = 1

        if sum(reach_goal) == len(x):
            break

        # check reaching goal

    print("Done")

    if show_animation:
        for i in range(len(x)):
            plt.plot(trajectory[i][:, 0], trajectory[i][:, 1], "-r")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.circle)
