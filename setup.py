from setuptools import setup

setup(name='jointcd',
      version='0.1',
      description='Python package implementing change detection and change point estimation using a joint distribution estimated from a training set of similar signals',
      url='https://github.com/willemolding/JointGaussianChangeDetector',
      author='Willem Olding',
      author_email='willemolding@gmail.com',
      license='MIT',
      packages=['jointcd'],
      test_suite='nose.collector',
      tests_require=['nose']
      )