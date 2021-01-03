from setuptools import setup

setup(name='iclamp-glm',
    version='0.1',
    description='iclamp-glm is a Python module for the analysis of electrophysiological patch clamp data by using encoding and decoding neural models.',
    url='https://github.com/diegoarri91/iclamp-glm',
    author='Diego M. Arribas',
    author_email='diegoarri91@gmail.com',
    license='BSD 3-Clause License',
    install_requires=['numpy', 'scipy'],
    packages=['icglm']
      )
