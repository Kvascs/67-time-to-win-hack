from setuptools import setup

package_name = 'tram_backup_odometry_tools'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MosTransHack team',
    maintainer_email='team@example.com',
    description='Jury tools for tram_backup_odometry: live latency probe and run evaluation.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'latency_probe = tram_backup_odometry_tools.latency_probe:main',
            'evaluate_run = tram_backup_odometry_tools.evaluate_run:main',
        ],
    },
)
