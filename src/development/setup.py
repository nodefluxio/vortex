from setuptools import setup, Extension, distutils, find_namespace_packages

# Package Information
import importlib.util
spec = importlib.util.spec_from_file_location("version", "vortex/development/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
version = version_module.__version__
package_name = 'visual-cortex'

# Requirements
# TODO spawn subprocess for apt installation
with open('requirements.txt') as f:
    install_requires = [line for line in f.read().splitlines() if not line.startswith('#') ]

# Extra requirements for optuna visualization
with open('optuna.vis.requirements.txt') as f:
    optuna_vis_requires = f.read().splitlines()

# Python scripts console entrypoints
entry_points = {
    'console_scripts': [
        'vortex=vortex.development.vortex_cli:main',
    ]
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
      packages=find_namespace_packages(),
      zip_safe=False)

## Additional setup for NVIDIA-DALI, dependency links is no longer supported by pip

import subprocess

p = subprocess.Popen(["pip3",
                      "install",
                      "--extra-index-url","https://developer.download.nvidia.com/compute/redist",
                      "nvidia-dali-cuda100"],
                     stdout=subprocess.PIPE)
p.wait()
