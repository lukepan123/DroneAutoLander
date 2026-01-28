#!/usr/bin/env python3
import math
import numpy as np
from geometry_msgs.msg import TwistStamped
import casadi as ca
import rclpy
from rclpy.time import Time

def Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
           
    if node._tracking_enabled:
        # Time step
        now = node.get_clock().now()
        dt = (now - node.prev_time).nanoseconds * 1e-9
        dt = max(dt, 1e-3)
        node.prev_time = now

        # Determine target coordinates based on target position and velcoity
        lookout_factor = 1.0

        lookout_x = node.target_odometry.twist.twist.linear.x
        lookout_y = node.target_odometry.twist.twist.linear.y

        # node.get_logger().info(f'rot {euler_map_to_target[2]}')

        x_target = lookout_factor * lookout_x + node.target_pose[0]
        y_target = lookout_factor * lookout_y + node.target_pose[1]
        z_target = node.target_odometry.pose.pose.position.z + 0.5

        # node.get_logger().info(f'x {x_target}, y {y_target}, x_add {lookout_x}, y_add {lookout_y}')

        # Position error
        x_error = x_target - node.pose.pose.position.x
        y_error = y_target - node.pose.pose.position.y
        z_error = z_target - node.pose.pose.position.z

        yaw_error = 0

        # Integrate error
        node.x_error_int += x_error * dt
        node.y_error_int += y_error * dt

        # Anti-windup clamp
        max_int = 2.0
        node.x_error_int = max(min(node.x_error_int, max_int), -max_int)
        node.y_error_int = max(min(node.y_error_int, max_int), -max_int)

        # Derivative
        dx_error = (node.target_vel[0] - node.twist.twist.linear.x)
        dy_error = (node.target_vel[1] - node.twist.twist.linear.y)
        dz_error = (node.target_vel[2] - node.twist.twist.linear.z)

        node.x_p = node.kp * x_error
        node.x_i = node.ki * node.x_error_int
        node.x_d = node.kd * dx_error
        node.y_p = node.kp * y_error
        node.y_i = node.ki * node.y_error_int
        node.y_d = node.kd * dy_error

        # Make gains more sensitive as drone gets closer to rover
        node.kp = max(min((node.target_altitude + z_error)/7 * 2.5,1.7),0.2)

        # PID output
        vx = node.kp * x_error + node.ki * node.x_error_int + node.kd * dx_error
        vy = node.kp * y_error + node.ki * node.y_error_int + node.kd * dy_error
        if node.pose.pose.position.z < 0.7:
            vz = -5.0
        else:
            vz = -0.2 # 0.04 * z_error

        omg_z = yaw_error

        # Velocity saturation
        max_vel = 8.0
        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)
        vz = max(min(vz, max_vel), -max_vel)

        omg_z = max(min(omg_z, max_vel), -max_vel)

        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.z = float(omg_z)
            
        node.vel_pub.publish(msg)

        # print("X_drone", node.pose.pose.position.x, "Y_drone", node.pose.pose.position.y, "Z_drone", node.pose.pose.position.z)
        # print("X_targ", node.target_odometry.pose.pose.position.x, "Y_targ", node.target_odometry.pose.pose.position.y, "Z_targ", node.target_odometry.pose.pose.position.z)
        # print("X_err", x_error, "Y_err", y_error, "Z_targ")
        # print("X_com", vx, "Y_com", vy)
        #error_mag = np.sqrt(np.power(x_error, 2) + np.power(y_error, 2))
        #print("Distance Error: ", error_mag)

    return msg

class MPCController:
    def __init__(self, node):
        # State variables: 
        x  = ca.SX.sym('x')
        y  = ca.SX.sym('y')
        z  = ca.SX.sym('z')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        vz = ca.SX.sym('vz')

        states = ca.vertcat(x, y, z, vx, vy, vz)
        self.n_states = states.size1()  # should be 6

        # Control variable: u (velocity commands)
        x_vel     = ca.SX.sym('x_vel')
        y_vel     = ca.SX.sym('y_vel')
        z_vel     = ca.SX.sym('z_vel')
        u = ca.vertcat(x_vel, y_vel, z_vel)
        self.n_controls = u.size1()     # should be 3

        # 6 parameters: initial drone state (x0,y0,z0,vx0,vy0,vz0) + target position (x_target,y_target,z_target)
        P = ca.SX.sym('P', self.n_states + 3)

        # Define the number of discretization points
        now = node.get_clock().now()
        #self.dt = (now - node.prev_time).nanoseconds * 1e-9
        self.dt = 0.1#max(self.dt, 1e-3)    # Get dt between each cycle
        node.prev_time = now
        self.N = 20                     # Prediction horizon (number of control intervals)

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

        # We choose weights for the cost function:
        Q_x = 50.0        # Weight for drone x error
        Q_y = 50.0        # Weight for drone y error
        Q_z = 0.005       # Weight for drone z error
        R_x = 50.0       # Weight on x_dot control effort
        R_y = 50.0       # Weight on y_dot control effort
        R_z = 0.5       # Weight on z_dot control effort

        # Initialize objective (cost function) and constraints
        obj = 0         # Accumulated cost
        g = []          # Equality constraints list

        # Enforce initial condition constraint: the first state in X equals the current state P[0:self.n_states].
        g.append(X[:,0] - P[0:self.n_states])

        # Loop over each control interval to build the cost and dynamics constraints
        for k in range(self.N):
            # Cost function: Penalize errors in drone position, yaw, and control effort.
            obj += (Q_x * (X[0, k] - P[self.n_states + 0])**2 + 
                    Q_y * (X[1, k] - P[self.n_states + 1])**2 + 
                    Q_z * (X[2, k] - P[self.n_states + 2])**2 + 
                    R_x * U[0, k]**2 + 
                    R_y * U[1, k]**2 + 
                    R_z * U[2, k]**2
            )
            
            # Dynamics constraint: next state equals the discrete dynamics from current state and control.
            x_next = self.f(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)

        # Optionally, add a terminal cost on the final state (i.e. dont ignore the long term effects)
        obj += 100.0 * (Q_x * (X[0, self.N] - P[self.n_states + 0])**2 +
                Q_y * (X[1, self.N] - P[self.n_states + 1])**2 +
                Q_z * (X[2, self.N] - P[self.n_states + 2])**2
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
            self.ubx[idx + 0] = x_vel_max
            self.ubx[idx + 1] = y_vel_max
            self.ubx[idx + 2] = z_vel_max

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
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_problem, opts)

        # Initial guess for the decision variables
        self.init_guess = np.zeros(int(opt_vars.numel()))
        self.prev_sol = None

    # Drone Dynamical Model
    def drone_dynamics(self, states, u):
        # Since ardupilot FCU uses a internal rate PID, we only have access to the velocity setpoints, so we can use basic kinematic equations to model the system
        # Computes the time derivative of the state vector
        tau = 0.5
        x_dot = states[3]
        vx_dot = (u[0] - states[3])/tau
        y_dot = states[4]
        vy_dot = (u[1] - states[4])/tau
        z_dot = states[5]
        vz_dot = (u[2] - states[5])/tau

        return ca.vertcat(x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot)
 
    def compute_control(self, node):
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if node._tracking_enabled:
            if node.target_found == False:
                target_z = min(node.pose.pose.position.z + 1.0, 15.0)
            else:
                target_z = node.target_pose[2] + 1.0

            # Solve the NMPC for the current state x_current
            look_ahead = 1.0
            x_current = np.array([
                node.pose.pose.position.x,
                node.pose.pose.position.y,
                node.pose.pose.position.z,
                node.twist.twist.linear.x,
                node.twist.twist.linear.y,
                node.twist.twist.linear.z,
                node.target_pose[0] + look_ahead * node.target_vel[0],
                node.target_pose[1] + look_ahead * node.target_vel[1],
                target_z,
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
            msg.twist.angular.z = 0.0

            node.vel_pub.publish(msg)

        return msg
