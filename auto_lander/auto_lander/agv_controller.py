import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class AGV_Controller(Node):
    def __init__(self):
        super().__init__('agv_controller')

        self.pub = self.create_publisher(Twist, '/cmd_rover_vel', 10)

        self.current_speed = 0.0          # m/s
        self.current_angular_speed = 0.0

        self.target_speed = 10.0         # 100 km/h
        self.acceleration = 1.0           # m/s²

        self.dt = 0.05                    # 20 Hz
        self.timer = self.create_timer(self.dt, self.update)

    def update(self):
        if self.current_speed < self.target_speed:
            self.current_speed += self.acceleration * self.dt
            self.current_speed = min(self.current_speed, self.target_speed)

        msg = Twist()
        msg.linear.x = self.current_speed
        msg.angular.z = self.current_angular_speed

        self.pub.publish(msg)

        self.get_logger().info(
            f"Speed: {self.current_speed:.2f} m/s", throttle_duration_sec=1.0
        )


def main():
    rclpy.init()

    node = AGV_Controller()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()