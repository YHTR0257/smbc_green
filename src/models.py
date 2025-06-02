import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

class Model1():
    """A simple model class for demonstration purposes."""

    def __init__(self,config:dict):
        self.param1 = config.get('param1', 1)
        self.random_state = config['random_state'][0] if 'random_state' in config else 42
        self.learning_rate = config.get('learning_rate', 0.01)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 3)
        self.min_samples_split = config.get('min_samples_split', 2)

    def fit(self, X, y):
        """Fit the model to the training data."""
        # Placeholder for fitting logic
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        # Placeholder for prediction logic
        return [0] * len(X)  # Dummy prediction

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        # Placeholder for scoring logic
        return 0.5