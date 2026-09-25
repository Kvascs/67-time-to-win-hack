"""Launch the tram backup odometry node with the packaged parameters, map and traction table."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tram_backup_odometry')
    default_params = os.path.join(share, 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML with estimator parameters'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Only affects /diagnostics stamps; outputs use input stamps'),
        DeclareLaunchArgument('output_frame', default_value='mgrs',
                              description='mgrs (jury: Autoware MGRS map frame 37U DB) | enu | utm | map'),
        Node(
            package='tram_backup_odometry',
            executable='tbo_node',
            name='tram_backup_odometry',
            output='screen',
            parameters=[LaunchConfiguration('params_file'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'output_frame': LaunchConfiguration('output_frame')}],
        ),
    ])
