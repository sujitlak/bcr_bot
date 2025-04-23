import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import math
import os

class OccupancyGridMapper(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapper')
        self.subscription = self.create_subscription(
            LaserScan,
            '/bcr_bot/scan',
            self.lidar_callback,
            10
        )
        self.grid_size = 200  # 200x200 grid
        self.resolution = 0.05  # 5 cm per cell
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.origin = (self.grid_size // 2, self.grid_size // 2)
        self.fig, self.ax = plt.subplots()

    def lidar_callback(self, msg):
        self.grid.fill(0)
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                grid_x = int(x / self.resolution) + self.origin[0]
                grid_y = int(y / self.resolution) + self.origin[1]
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.grid[grid_y, grid_x] = 100
            angle += msg.angle_increment

        self.update_map()

    def update_map(self):
        self.ax.clear()
        self.ax.imshow(self.grid, cmap='gray', origin='lower')
        self.ax.set_title("Occupancy Grid Map")
        plt.pause(0.01)

    def save_map(self, filename='occupancy_grid_map.png'):
        plt.ioff()
        plt.imshow(self.grid, cmap='gray', origin='lower')
        plt.title("Final Occupancy Grid Map")
        plt.savefig(filename)
        self.get_logger().info(f"Map saved as {os.path.abspath(filename)}")

def main(args=None):
    rclpy.init(args=args)
    mapper = OccupancyGridMapper()
    plt.ion()
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        mapper.get_logger().info("Shutting down. Saving map...")
        mapper.save_map()
    finally:
        mapper.destroy_node()
        rclpy.shutdown()
        plt.close()

if __name__ == '__main__':
    main()
