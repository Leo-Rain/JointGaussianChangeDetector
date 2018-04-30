from setuptools import setup

install_requires = [
	'sklearn',
	'numpy',
	'scipy'
	]


setup(name='jointcd',
      version='1.0',
      description='Python package implementing change detection and change point estimation using a joint distribution estimated from a training set of similar signals',
      url='https://github.com/willemolding/JointGaussianChangeDetector',
      install_requires=install_requires,
      author='Willem Olding',
      author_email='willemolding@gmail.com',
      license='MIT',
      packages=['jointcd'],
      test_suite='nose.collector',
      tests_require=['nose']
      )