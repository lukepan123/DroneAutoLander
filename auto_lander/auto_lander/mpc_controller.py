#!/usr/bin/env python3
# Dynamics model based on https://arxiv.org/pdf/2506.15447

import numpy as np
import casadi as ca

from tf_transformations import quaternion_from_euler
from mavros_msgs.msg import AttitudeTarget

class MPCController:
    def __init__(self):
        # ---- STATE VARIABLES ----
        # MPC Params
        self.dt = 0.5
        self.N = 40

        # Quad state variables:
        self.pos = ca.MX.sym('p', 3) # type: ignore
        self.vel = ca.MX.sym('v', 3) # type: ignore
        self.rot = ca.MX.sym('q', 3) # type: ignore

        self.x = ca.vertcat(self.pos, self.vel, self.rot)
        self.nx = self.x.size1()

        # Quad inputs
        self.u = ca.MX.sym('u', 4)   # type: ignore [Thrust, Roll, Pitch, Yaw]
        self.nu = self.u.size1()

        # Rover state variables:
        self.nr = 6         # number of rover states (x,y,z,yaw,v,yaw_rate(w))

        # ---- QUAD DYNAMICS MODEL ----
        g = 9.81
        m = 1.82
        tau_roll = 0.08
        tau_pitch = 0.08
        tau_yaw = 0.08
        throttle_thrust_scaler = 14.0 #16N - adjust accordingly

        px, py, pz       = self.x[0], self.x[1], self.x[2]
        vx, vy, vz       = self.x[3], self.x[4], self.x[5]
        roll, pitch, yaw = self.x[6], self.x[7], self.x[8]

        throttle_cmd     = self.u[0]
        roll_cmd         = self.u[1]
        pitch_cmd        = self.u[2]
        yaw_cmd         = self.u[3]

        thrust_cmd = throttle_cmd * throttle_thrust_scaler

        # Translational dynamics
        s_rol = ca.sin(roll)
        c_rol = ca.cos(roll)
        s_pit = ca.sin(pitch)
        c_pit = ca.cos(pitch)
        s_yaw = ca.sin(yaw)
        c_yaw = ca.cos(yaw)
        
        # Rotation (base_link -> map/world)
        R = ca.vertcat(
            ca.horzcat(c_yaw*c_pit, c_yaw*s_rol*s_pit - c_rol*s_yaw, s_rol*s_yaw + c_rol*c_yaw*s_pit),
            ca.horzcat(c_pit*s_yaw, c_rol*c_yaw + s_rol*s_yaw*s_pit, c_rol*s_yaw*s_pit - c_yaw*s_rol),
            ca.horzcat(-s_pit , c_pit*s_rol, c_rol*c_pit)
        )

        # Thrust in body frame (z-axis)
        T_b = ca.vertcat(0, 0, thrust_cmd + m*g)

        # Acceleration in world frame
        acc = -ca.vertcat(0, 0, g) + (1/m) * ca.mtimes(R, T_b)

        ax = acc[0]
        ay = acc[1]
        az = acc[2]

        # Rotational dynamics
        vroll  = 1/tau_roll * (roll_cmd - roll)
        vpitch = 1/tau_pitch * (pitch_cmd - pitch)
        vyaw   = 1/tau_yaw * (yaw_cmd - yaw)

        # Next state
        f_quad = ca.vertcat(
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            vroll,
            vpitch,
            vyaw
        )

        self.f = ca.Function(
            "f",
            [self.x, self.u],
            [self.x + self.dt * f_quad],
            {
                "jit": True,
                "jit_options": {
                    "compiler": "gcc",
                    "flags": ["-O3"]
                }
            }
        )

        # ---------- Decision variables ----------
        X = ca.MX.sym('X', self.nx, self.N+1) # type: ignore
        U = ca.MX.sym('U', self.nu, self.N)   # type: ignore

        # ---------- Parameters ----------
        # P = [quad_state(7), rover_state(6)]
        P = ca.MX.sym('P', self.nx + self.nr) # type: ignore

        x0 = P[0:self.nx]
        xr0 = P[self.nx:self.nx+self.nr]

        # ---------- Weights ----------
        Q_xy     = 20.0
        Q_z      = 1.0
        Q_rp     = 0#50.0
        Q_yaw    = 0#50.0
        Q_vxy    = 0#200.0
        Q_chase  = 0#10.0

        R_att    = 0.0
        R_thrust = 0.0
        Q_term   = 0.5

        # ---------- Cost & constraints ----------
        cost = 0
        g = [X[:, 0] - x0]

        xr = xr0

        for k in range(self.N):
            # Positional/distance error
            dx = X[0, k] - xr[0]
            dy = X[1, k] - xr[1]
            dz = X[2, k] - xr[2]
            dist_vec = X[0:2, k] - xr[0:2]      # (drone - landing_pad) or vice versa, see below
            dist_unit = dist_vec / (ca.norm_2(dist_vec) + 1e-6) # If we want drone to move INFRONT, drone should aim opposite direction of dist

            # Rotational error (cosine difference)
            roll_err = X[6, k]
            pitch_err = X[7, k]
            yaw_err = 1 - ca.cos(X[8, k] - xr[3])

            # Velocity errors
            vel_vec = ca.vertcat(X[3, k], X[4, k])
            vx_r = xr[4] * ca.cos(xr[3])
            vy_r = xr[4] * ca.sin(xr[3])
            landing_pad_vel_vec = ca.vertcat(vx_r, vy_r)

            # Lookahead weighting (emphasise future errors more)
            lookahead_weight = (k+1)/self.N

            # Distance to landing_pad aggressiveness
            sensitivity = 5/(ca.fabs(X[2, k]) + 0.8)

            # ---- COST FUNCTION ----
            # Position error
            cost += sensitivity * lookahead_weight * Q_xy * (dx*dx + dy*dy)
            cost += sensitivity * lookahead_weight * Q_z * dz*dz

            # Velocity catch-up term: penalize lag or wrong velocity direction
            cost += sensitivity * Q_vxy * ca.mtimes((vel_vec - landing_pad_vel_vec).T, (vel_vec - landing_pad_vel_vec))
            cost += sensitivity * Q_chase * ca.mtimes(vel_vec.T, dist_unit) # incentivizes moving toward landing_pad

            # Rotational error
            cost += Q_rp * (roll_err*roll_err + pitch_err*pitch_err) + Q_yaw * yaw_err*yaw_err

            # Control effort
            cost += R_att * (U[1,k]**2 + U[2,k]**2 + U[3,k]**2)
            cost += R_thrust * (U[0,k]**2)

            # Quad dynamics constraint
            g.append(X[:, k+1] - self.f(X[:, k], U[:, k]))

            # Rover propagated explicitly outside the symbolic graph
            xr = xr + self.dt * ca.vertcat(
                xr[4] * ca.cos(xr[3]),
                xr[4] * ca.sin(xr[3]),
                0,
                xr[5],
                0,
                0
            )

        # Terminal cost
        dxT = X[0, self.N] - xr[0]
        dyT = X[1, self.N] - xr[1]
        dzT = X[2, self.N] - xr[2]
        yawT = 1 - ca.cos(X[8, self.N] - xr[3])

        cost += Q_term * (dxT*dxT + dyT*dyT + dzT*dzT + yawT*yawT)

        # ---------- NLP Construction ----------
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g_concat = ca.vertcat(*g)

        # Bounds
        self.lbx = -1e12 * np.ones(opt_vars.numel())
        self.ubx = +1e12 * np.ones(opt_vars.numel())

        # Constrain z ≥ 0
        for i in range(self.N+1):
            self.lbx[i*self.nx + 2] = 0.0

        # Control bounds
        u_start = self.nx * (self.N+1)
        for k in range(self.N):
            i = u_start + k*self.nu
            self.lbx[i:i+4] = np.array([0.0, -0.5, -0.5, -2.0944])
            self.ubx[i:i+4] = np.array([1.0,  0.5,  0.5,  2.0944])

        # Constraint equality bounds
        ng = g_concat.numel()
        self.lbg = np.zeros(ng)
        self.ubg = np.zeros(ng)

        # ---------- Solver ----------
        nlp = {
            "f": cost,
            "x": opt_vars,
            "g": g_concat,
            "p": P
        }

        opts = {
            "jit": True,
            "jit_options": {
                "compiler": "gcc",
                "flags": ["-O3"]
            },
            "ipopt": {
                "max_iter": 5,
                "tol": 1e-2,
                "print_level": 0,
                "warm_start_init_point": "yes",
            },
            "print_time": 0,
            "expand": True
        }

        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        self.prev_sol = None
        self.n_dec = opt_vars.numel()

    # -------------- Compute control at runtime --------------
    def compute_control(self, node):

        msg = AttitudeTarget()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # We send only rates + thrust
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        # Select landing_pad altitude
        if node.controller_state == 4000:
            landing_pad_z = 1.0
        else:
            landing_pad_z = 15.0

        # Build the parameter vector P
        P_val = ca.DM([
            node.odometry.pose.pose.position.x,
            node.odometry.pose.pose.position.y,
            node.odometry.pose.pose.position.z,
            node.odometry.twist.twist.linear.x,
            node.odometry.twist.twist.linear.y,
            node.odometry.twist.twist.linear.z,
            node.roll,
            node.pitch,
            node.yaw,

            node.landing_pad_position[0],
            node.landing_pad_position[1],
            landing_pad_z,
            node.landing_pad_yaw,
            node.landing_pad_velocity_mag,
            node.landing_pad_yaw_rate
        ])

        # Warm start
        if self.prev_sol is None:
            x0 = np.zeros(self.n_dec)
        else:
            x0 = self.prev_sol["x"].full().flatten()

        sol = self.solver(
            x0=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=P_val
        )

        self.prev_sol = sol
        sol_opt = sol["x"].full().flatten()

        # Extract first control input U0
        u_start = self.nx * (self.N+1)
        u = sol_opt[u_start:u_start+self.nu]

        # Thrust + Attitude
        msg.thrust = float(u[0])
        
        q = quaternion_from_euler(float(u[1]), float(u[2]), float(u[3]))  # roll, pitch, yaw
        # msg.orientation.x = q[0]
        # msg.orientation.y = q[1]
        # msg.orientation.z = q[2]
        # msg.orientation.w = q[3]

        node.get_logger().info(f'throt: {u[0]}')
        # node.get_logger().info(f'euler: {u[1]}, {u[2]}, {u[3]}')

        #node.att_pub.publish(msg)
        return msg