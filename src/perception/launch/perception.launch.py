import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    ip_camera_pkg_name = 'ip_camera'

    offset = 8
    batch_size = 7

    __composable_node_descriptions = []
    # ip_camera_nodes = [
    #     ComposableNode(
    #         package='ip_camera',
    #         plugin='IPCameraStreamer',
    #         name=f'ipcam_{i}',
    #         parameters=[
    #             PathJoinSubstitution([
    #                 FindPackageShare(ip_camera_pkg_name), 'config', 'ipcam.yaml']),
    #         ],
    #         remappings=[
    #             (f'/ipcam_{i}/image_raw', f'/ipcam_{i - offset + 1}/image_raw'),
    #             (f'/ipcam_{i}/image_raw/compressed', f'/ipcam_{i - offset + 1}/image_raw/compressed'),
    #         ],
    #         extra_arguments=[{'use_intra_process_comms': True}]
    #     ) for i in range(offset, offset + batch_size)
    # ]

    perception_nodes = [
        ComposableNode(
            package='perception',
            plugin='PerceptionNode',
            name='perception_node'
            ,
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    ]
    
    # __composable_node_descriptions.extend(ip_camera_nodes)
    # __composable_node_descriptions.extend(perception_nodes)

    component_container = ComposableNodeContainer(
        name='total_container', 
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        output='screen',
        
        composable_node_descriptions=__composable_node_descriptions,
    )

    return LaunchDescription([
        component_container
    ])