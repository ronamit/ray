# Install with: $ python3 setup.py install

# based on: https://stackoverflow.com/questions/49031491/import-from-my-package-recognized-by-pycharm-but-not-by-command-line


from setuptools import setup, find_packages

setup(
    name='ray_project',
    packages=find_packages(),
    version='0.0.1',
)

# ['RLlib_runs', 'RLlib_runs.ddpg_custom', 'ray'],
#



