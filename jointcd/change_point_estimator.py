from change_detector import ChangeDetector
import numpy as np
from functools import partial
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinvh


def partition(sigma, k):
    """
    Partitions a covariance matrix by zeroing out off diagonal blocks to create independence
    """
    S = np.copy(sigma)
    S[k:,:k] = 0
    S[:k,k:] = 0
    return S

def distance(x, mu, precisions):
    """
    Given an array of precision matrices (inverse covariance) calculate the mhalanobis distances
    """
    distances = np.array([mahalanobis(x, mu, precisions[k]) for k in range(x.shape[0])])
    return distances


class ChangePointEstimator(ChangeDetector):

    def predict(self, X):
        """
        Returns the most probably change point in each time series.
        Also returns the time series of mahalanobis distances

        Parameters:
        -----------
        X - array of time series, shape (n_series, len_series)
        """
        D,N = X.shape

        sigma = self.covariance_estimator_.covariance_
        mu = self.covariance_estimator_.location_

        # calculate the precision matrices for all possible partitions
        precisions = [pinvh(partition(sigma, k)) for k in range(N)]

        # calculate the mahalanobis distance for each candidate change point in each time series
        distance_time_series = np.apply_along_axis(partial(distance, mu=mu, precisions=precisions), 1, X)

        # return the min distance (max likelihood change point) and array of distances
        return np.argmin(distance_time_series, axis=1), distance_time_series
