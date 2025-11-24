import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

def generate_launch_description():
    pkg_name = 'ip_camera' 
    pkg_share = get_package_share_directory(pkg_name)
    
    offset = 1
    batch_size = 14

    component_container = ComposableNodeContainer(
        name='ip_camera_container', 
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        output='screen',
         
        composable_node_descriptions=[
            ComposableNode(
                package='ip_camera',
                plugin='IPCameraStreamer',
                name=f'ipcam_{i}',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('ip_camera'), 'config', 'ipcam.yaml']),
                ]
            ) for i in range(offset, offset + batch_size)
        ],
    )

    return LaunchDescription([
        component_container
    ])