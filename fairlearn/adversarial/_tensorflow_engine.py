
from numpy import finfo, float32

from ._backend_engine import BackendEngine

tensorflow = None
keras = None


class TensorflowEngine(BackendEngine):

    def __init__(self, base, X, Y, A):
        """
        Initialize the (Tensorflow specific parts) of the backend engine.

        There are not really tensorflow specifics besides the import, but don't
        forget to set up the optimizers after the init!
        """
        global tensorflow
        import tensorflow

        global keras
        import keras

        tensorflow.random.set_seed(base.random_state_.random())

        self.model_class = keras.Model
        self.optim_class = keras.optimizers.Optimizer
        super(TensorflowEngine, self).__init__(base, X, Y, A)

    def evaluate(self, X):
        pass

    def train_step(self, X, Y, A):
        pass

    def get_optimizer(self, optim_param, model):
        pass

    def get_loss(self, dist_type):
        pass

    def get_model(self, list_nodes):
        pass
