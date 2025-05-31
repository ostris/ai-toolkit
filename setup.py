from setuptools import find_packages, setup

with open('requirements.txt') as file:
    requirements = file.read().splitlines()

setup(
  name='ai_toolkit',
  packages=find_packages(),
  version='0.0.0',
  description='AI Toolkit by Ostris',
  author='Jaret Burkett <jaretburkett@gmail.com>',
  install_requires=requirements
)
