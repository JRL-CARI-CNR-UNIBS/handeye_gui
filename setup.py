# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
from glob import glob

from setuptools import find_packages, setup

package_name = "handeye_gui"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "CHANGELOG.rst", "LICENSE"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/docs", glob("docs/*.md")),
        ("share/" + package_name + "/docs/images", glob("docs/images/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Manuel Beschi",
    maintainer_email="manuel.beschi@unibs.it",
    description="Stand-alone PySide6 GUI for eye-in-hand and eye-to-hand calibration.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "handeye_gui = handeye_gui.gui.main_window:main",
            "handeye_cli = handeye_gui.cli:main",
        ],
    },
)
