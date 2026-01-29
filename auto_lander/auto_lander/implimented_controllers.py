#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import TwistStamped
import casadi as ca

class MPCController:
    def __init__(self, dt):
        # State variables: 
        x  = ca.SX.sym('x')
        y  = ca.SX.sym('y')
        z  = ca.SX.sym('z')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        vz = ca.SX.sym('vz')
        yaw = ca.SX.sym('yaw')
        vyaw   = ca.SX.sym('vyaw')   # yaw rate

        states = ca.vertcat(x, y, z, vx, vy, vz, yaw, vyaw)
        self.n_states = states.size1()  # should be 8
        self.rover_n_states = 6         # number of rover states (x,y,z,yaw,v,yaw_rate(w))

        # Control variable: u (velocity commands)
        x_vel     = ca.SX.sym('x_vel')
        y_vel     = ca.SX.sym('y_vel')
        z_vel     = ca.SX.sym('z_vel')
        yaw_vel     = ca.SX.sym('yaw_vel')
        u = ca.vertcat(x_vel, y_vel, z_vel, yaw_vel)
        self.n_controls = u.size1()     # should be 4

        # P = [quad_state(8), rover_state(5)]
        P = ca.SX.sym('P', self.n_states + self.rover_n_states)

        # Define the number of discretization points
        self.dt = dt
        self.N = 15      # Prediction horizon (number of control intervals)

        # Decision variables for the entire horizon
        # X will be of size (n_states x (N+1)) and U of size (n_controls x N)
        X = ca.SX.sym('X', self.n_states, self.N+1)
        U = ca.SX.sym('U', self.n_controls, self.N)

    
        # Define the discrete dynamics function using simple Euler integration:
        self.f = ca.Function(
            'f',
            [states, u],
            [states + self.dt * self.drone_dynamics(states, u)]
        )

        # Initial rover state
        xr = P[self.n_states:self.n_states + self.rover_n_states]

        # We choose weights for the cost function:
        Q_pos = 50.0
        Q_z   = 0.5
        Q_yaw = 10.0
        Q_vel = 50.0
        R_v   = 100.0
        R_r   = 40.0

        # Initialize objective (cost function) and constraints
        obj = 0         # Accumulated cost
        g = []          # Equality constraints list

        # Enforce initial condition constraint: the first state in X equals the current state P[0:self.n_states].
        g.append(X[:,0] - P[0:self.n_states])

        # Loop over each control interval to build the cost and dynamics constraints
        for k in range(self.N):
            # Cost function: Penalize errors in drone position, yaw, and control effort.
            err_x = X[0,k] - xr[0]
            err_y = X[1,k] - xr[1]
            err_z = X[2,k] - xr[2]

            err_yaw = ca.atan2(
                ca.sin(X[6,k] - xr[3]),
                ca.cos(X[6,k] - xr[3])
            )

            vx_ref = xr[4] * ca.cos(xr[3])
            vy_ref = xr[4] * ca.sin(xr[3])

            obj += (
                Q_pos * err_x**2 +
                Q_pos * err_y**2 +
                Q_z   * err_z**2 +
                Q_yaw * err_yaw**2 +
                Q_vel * (X[3,k] - vx_ref)**2 + (X[4,k] - vy_ref)**2 +
                R_v   * (U[0,k]**2 + U[1,k]**2 + U[2,k]**2) +
                R_r   * U[3,k]**2
            )
            
            # Dynamics constraint: next state equals the discrete dynamics from current state and control.
            x_next = self.f(X[:, k], U[:, k])
            xr = xr + self.dt * self.rover_dynamics(xr)
            xr[3] = ca.atan2(ca.sin(xr[3]), ca.cos(xr[3])) # check angle hasnt overspilled
            g.append(X[:, k+1] - x_next)

        # Optionally, add a terminal cost on the final state (i.e. dont ignore the long term effects)
        obj += 100.0 * (
            Q_pos * (X[0, self.N] - xr[0])**2 +
            Q_pos * (X[1, self.N] - xr[1])**2 +
            Q_z * (X[2, self.N] - xr[2])**2 +
            Q_yaw * ca.atan2(
                ca.sin(X[6, self.N] - xr[3]),
                ca.cos(X[6, self.N] - xr[3])
            )**2
        )

        # Define limits for control input and output
        z_min = 0.0

        # Control constraint: u_min <= u <= u_max
        x_vel_min = -3.0
        x_vel_max =  3.0
        y_vel_min = -3.0
        y_vel_max =  3.0
        z_vel_min = -3.0
        z_vel_max =  3.0
        yaw_vel_min = -5.0
        yaw_vel_max =  5.0

        # The decision vector consists of:
        #  - First: states, shaped as (n_states*(N+1),)
        #  - Second: control inputs, shaped as (n_controls*N,)
        # # Reshape decision variables into a single column vector.
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        self.n_decision = int(opt_vars.numel())
        self.lbx = -np.inf * np.ones(self.n_decision)
        self.ubx =  np.inf * np.ones(self.n_decision)

        # Set box constraints for the horizontal position (state variable X)
        # Each time step: the z-component is the third element in the state block.
        for i in range(self.N+1):
            idx = i * self.n_states  # index for x at time step i
            self.lbx[idx + 2] = z_min

        # Set box constraints for the control variable (u)
        # Controls come after the state variables in the decision vector.
        start_u = self.n_states * (self.N+1)
        for i in range(self.N):
            idx = start_u + i * self.n_controls  # index for control at time step i
            self.lbx[idx + 0] = x_vel_min
            self.lbx[idx + 1] = y_vel_min
            self.lbx[idx + 2] = z_vel_min
            self.lbx[idx + 3] = yaw_vel_min
            self.ubx[idx + 0] = x_vel_max
            self.ubx[idx + 1] = y_vel_max
            self.ubx[idx + 2] = z_vel_max
            self.ubx[idx + 3] = yaw_vel_max

        # Concatenate all constraints into one vector.
        g_concat = ca.vertcat(*g)

        # Define bounds for the constraints (all equality constraints are set to 0)
        g_bound = np.zeros(int(g_concat.numel()))
        self.lbg = g_bound
        self.ubg = g_bound

        # Define the NLP problem
        nlp_problem = {
            'f': obj,       # Objective function
            'x': opt_vars,  # Decision variables
            'g': g_concat,  # Constraints
            'p': P          # Parameter: initial state and target position
        }

        # Create an NLP solver instance using IPOPT.
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 40,
            'ipopt.tol': 1e-3
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_problem, opts)

        # Initial guess for the decision variables
        self.init_guess = np.zeros(int(opt_vars.numel()))
        self.prev_sol = None

    # Drone Dynamical Model  
    def drone_dynamics(self, s, u):
        # Since ardupilot FCU uses a internal rate PID, we only have access to the velocity setpoints, so we can use basic kinematic equations to model the system
        # Computes the time derivative of the state vector
        tau_v = 0.5
        tau_r = 0.3

        return ca.vertcat(
            s[3],                         # x_dot
            s[4],                         # y_dot
            s[5],                         # z_dot
            (u[0] - s[3]) / tau_v,        # vx_dot
            (u[1] - s[4]) / tau_v,        # vy_dot
            (u[2] - s[5]) / tau_v,        # vz_dot
            s[7],                         # psi_dot
            (u[3] - s[7]) / tau_r         # r_dot
        )
    
    def rover_dynamics(self, xr):
        x   = xr[0]
        y   = xr[1]
        z   = xr[2]
        yaw = xr[3]
        v   = xr[4]
        w   = xr[5]

        return ca.vertcat(
            v * ca.cos(yaw),   # x_dot
            v * ca.sin(yaw),   # y_dot
            0.0,               # z_dot (ground rover)
            w,                 # yaw_dot
            0.0,               # v_dot
            0.0                # w_dot
        )
 
    def compute_control(self, node):
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if node._tracking_enabled:
            if node.target_found == False:
                target_z = min(node.odometry.pose.pose.position.z + 1.0, 15.0)
            else:
                target_z = node.target_position[2] + 1.0

            # Solve the NMPC for the current state x_current
            x_current = ca.DM([
                node.odometry.pose.pose.position.x,
                node.odometry.pose.pose.position.y,
                node.odometry.pose.pose.position.z,
                node.odometry.twist.twist.linear.x,
                node.odometry.twist.twist.linear.y,
                node.odometry.twist.twist.linear.z,
                node.yaw,
                node.odometry.twist.twist.angular.z,

                node.target_position[0],
                node.target_position[1],
                target_z,
                node.target_yaw,
                node.target_velocity_mag,
                node.target_yaw_rate
            ])

            if self.prev_sol is None:
                x0 = np.zeros(self.n_decision)
            else:
                x0 = self.prev_sol['x'].full().flatten()

            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x_current)
            sol_opt = sol['x'].full().flatten()
            
            # Extract the control sequence from the solution. The state trajectory is first,
            # so the first control is located at index = n_states*(N+1), with length self.n_controls
            u_opt = sol_opt[self.n_states*(self.N+1): self.n_states*(self.N+1)+self.n_controls]
            u_current = u_opt[0:self.n_controls]
            
            # Update initial guess to warm-start the next iteration (shift the sequence)
            self.prev_sol = sol

            # Publish as TwistStamped
            msg.twist.linear.x = float(u_current[0])
            msg.twist.linear.y = float(u_current[1])
            msg.twist.linear.z = float(u_current[2])
            # msg.twist.angular.z = float(u_current[3])

            node.vel_pub.publish(msg)

        return msg
