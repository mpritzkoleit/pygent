from setuptools import setup

setup(
    name='PyGent',
    version='0.15',
    packages=['pygent', 'pygent.algorithms','pygent.modeling_scripts','pygent.modeling_scripts.c_files'],#packages=['pygent', 'pygent/algorithms', 'pygent/modeling_scripts','pygent/modeling_scripts/c_files'],
    install_requires=['torch', 'gym'],
    requires=['sympy_to_c (>=0.1.2)', 'ffmpeg'],
    package_data={'pygent.modeling_scripts.c_files': ['*.so']},
    url='https//github.com/mpritzkoleit/pygent',
    author='Max Pritzkoleit',
    author_email='Max.Pritzkoleit@tu-dresden.de',
    description = 'Python library for control and reinforcement learning',
    long_description='PyGent is a reinforcement learning library, '
                'that delivers a general framework, similar to OpenAIs "gym", for control and simulation of dynamic systems.'
                'It implements some state-of-the-art deep reinforcement learning and '
                'trajectory optimization algorithms.'
)
