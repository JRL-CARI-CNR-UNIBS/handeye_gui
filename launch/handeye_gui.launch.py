# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Start the hand-eye calibration GUI.

    ros2 launch handeye_gui handeye_gui.launch.py config:=/path/config.yaml
    ros2 launch handeye_gui handeye_gui.launch.py session:=/path/session_folder
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _gui(context):
    args = []
    for name in ("config", "session"):
        value = LaunchConfiguration(name).perform(context)
        if value:
            args += [f"--{name}", value]
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    return [Node(package="handeye_gui", executable="handeye_gui", output="screen",
                 arguments=args, parameters=[{"use_sim_time": use_sim_time}])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value="", description="configuration YAML"),
        DeclareLaunchArgument("session", default_value="", description="session folder to open"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=_gui),
    ])
