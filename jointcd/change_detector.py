from sklearn.covariance import EmpiricalCovariance, MinCovDet

class ChangeDetector(object):
    """
    Joint Gaussian Change detector using a scikit learn style interface
    
    This class is really a wrapper around the methods in scikit learn for estimating covariance using
    robust or empirical methods and calculating the mahalanobis distances.

    """

    def __init__(self, method='robust', estimator_kw_args={}):
        if method is 'robust':
            self.covariance_estimator_ = MinCovDet(**estimator_kw_args)
        elif method is 'empirical': 
            self.covariance_estimator_ = EmpiricalCovariance(**estimator_kw_args)
        else:
            raise ValueError("{} is not a valid method. Must be one of 'robust' or 'empirical'".format(method))
        


    def fit(self, X):
        """
        Fits the estimator.

        Parameters:
        -----------
        X - array of time series, shape (n_series, len_series)
        """
        self.covariance_estimator_ = self.covariance_estimator_.fit(X)
        return self



    def predict(self, X, threshold):
        """
        Returns true for each time series predicted as change. Also returns the mahalanobis distances

        parameters:
        -----------
        X - array of time series, shape (n_series, len_series)
        threshold - float

        returns:
        y_pred - shape (n_time_series), true of change detected
        distances - shape (n_time_series). The mahanobis distances of each time series under the fitted distribution
        """
        distances = self.covariance_estimator_.mahalanobis(X)
        return distances > threshold, distances
