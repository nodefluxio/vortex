from setuptools import setup, Extension, distutils, find_packages

# Package Information
version = open('version.txt', 'r').read().strip()
package_name = 'visual-cortex'

# Requirements
# TODO spawn subprocess for apt installation
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# Extra requirements for optuna visualization
with open('optuna.vis.requirements.txt') as f:
    optuna_vis_requires = f.read().splitlines()

# Python scripts console entrypoints
entry_points = {
        'console_scripts': [
            'vortex=vortex.vortex_cli:main',
            ],
    }

tests_require = ['pytest','pytest-cov']

# Setup
setup(name=package_name,
      version=version,
      description='Vortex - A Deep Learning based Computer Vision development framework',
      url='https://github.com/nodefluxio/vortex',
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={
        "optuna_vis":  optuna_vis_requires
      },
      author='Nodeflux - AI Platform',
      entry_points = entry_points,
      license='MIT',
      packages=find_packages(exclude=['external','external.datasets','experiments','tests']),
      zip_safe=False)