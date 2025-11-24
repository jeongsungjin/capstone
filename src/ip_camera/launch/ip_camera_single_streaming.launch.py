import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

def generate_launch_description():
    # ipcamera_node 실행 (컴포넌트 컨테이너 대신 직접 executable)
    params_file = PathJoinSubstitution([
        FindPackageShare('ip_camera'), 'config', 'ipcam.yaml'
    ])

    print(params_file)

    ipcamera_node = Node(
        package='ip_camera',
        executable='ipcamera_node',
        name='ipcam_6',  # 요청: executable 직접 실행
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([ipcamera_node])