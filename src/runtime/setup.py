from setuptools import setup, find_namespace_packages

# Package Information
import importlib.util
spec = importlib.util.spec_from_file_location("version", "vortex/runtime/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
version = version_module.__version__
package_name = 'visual-cortex-runtime'

# Requirements
# TODO spawn subprocess for apt installation
with open('requirements.txt') as f:
    install_requires = [line for line in f.read().splitlines() if not line.startswith('#') ]

# Extra requirements for onnruntime
with open('onnxruntime_requirements.txt') as f:
    onnxruntime_requires = f.read().splitlines()
# Extra requirements for torchscript
with open('torchscript_requirements.txt') as f:
    torchscript_requires = f.read().splitlines()

# Setup
setup(name=package_name,
      version=version,
      description='Vortex Runtime - Runtime library for Vortex IR graph',
      url='https://github.com/nodefluxio/vortex',
      install_requires=install_requires,
      extras_require={
        "onnxruntime":  onnxruntime_requires,
        "torchscript":  torchscript_requires,
        "all": torchscript_requires+onnxruntime_requires
      },
      author='Nodeflux - AI Platform',
      license='MIT',
      packages=find_namespace_packages(),
      zip_safe=False)