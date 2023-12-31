import os
from typing import List
from collections import defaultdict
from pathlib import Path
from glob import glob
from setuptools import setup

package_name = 'feature_tracking'

def get_data_files(data_paths: List[str]):
    """
    Give list of data_path strings relative to package root, return data_files list such that once installed, these paths can be accessed via `package://{package_name}/{path}`.
    Reference: https://answers.ros.org/question/397319/how-to-copy-folders-with-subfolders-to-package-installation-path/?answer=397410#post-id-397410

    :param data_paths: List of data path strings relative to package root, whose contents should be installed as data files
    """
    data_files = []
    path_dict = defaultdict(list)
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.is_file():
            file_paths = [data_path]
            install_path = f"share/{package_name}/{file_path.parent}"
            path_dict[install_path].append(str(file_path))
        else:
            file_paths = [fp for fp in data_path.rglob("*") if fp.is_file()]
        for file_path in file_paths:
            install_path = f"share/{package_name}/{file_path.parent}"
            path_dict[install_path].append(str(file_path))

    for key in path_dict.keys():
        data_files.append((key, path_dict[key]))
    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jiacheng',
    maintainer_email='jiacheng@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_subscriber = feature_tracking.rs_image_sub:main',
            'optical_flow_test = feature_tracking.optical_flow_test:main',
            'cv2_temporal_tracking = feature_tracking.temporal_optical_flow_tracking:main',
            'record_3d = feature_tracking.record_3d_tracking_trajectory:main',
            'visualize_3d = feature_tracking.visualize_3d_tracking:main',
            'replay_3d = feature_tracking.replay_3d_tracking:main',
            'rs_pcd_snapshot = feature_tracking.rs_pcd_snapshot:main',
            'wrist_pcd_sub = feature_tracking.wrist_pcd_subscriber:main',
        ],
    },
)
