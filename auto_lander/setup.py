from setuptools import find_packages, setup

package_name = 'auto_lander'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/sim_launch.py',
            'launch/webcam_launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luke',
    maintainer_email='luke.pan88@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tagposedetector = auto_lander.tagposedetector:main',
            'controller = auto_lander.controller:main',
            'camera_calibrate = auto_lander.camera_calibrate:main'
        ],
    },
)
