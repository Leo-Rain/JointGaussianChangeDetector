from nose import *

from jointcd import ChangeDetector, ChangePointEstimator
import numpy as np

def setup():
	np.random.seed(420)

@with_setup(setup)
def test_create_cd():
	X = np.random.rand(100,200)
	cd = ChangeDetector()
	cd = cd.fit(X)
	change, distance = cd.predict(X, 0.2)

	assert change.shape[0] == X.shape[0]
	assert distance.shape[0] == X.shape[0]

@with_setup(setup)
def test_create_cpe():

	X = np.random.rand(100,200)
	cpe = ChangePointEstimator()
	cpe = cpe.fit(X)
	change_points, distance_signals = cpe.predict(X)

	assert change_points.shape[0] == X.shape[0]
	assert distance_signals.shape == X.shape