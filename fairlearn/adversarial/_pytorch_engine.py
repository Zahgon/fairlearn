
from ._backend_engine import BackendEngine
from ._constants import _MODEL_UNRECOGNIZED_ITEM, _MODEL_UNRECOGNIZED_STR

torch = None


class PytorchEngine(BackendEngine):

    def __init__(self, base, X, Y, A):
        """
        Initialize the (Pytorch specific parts) of the backend engine.

        The Pytorch-specifics include setting module class and handling Cuda.
        Also set up the optimizers after the init!
        """
        global torch
        import torch

        torch.manual_seed(base.random_state_.random())

        self.model_class = torch.nn.Module
        self.optim_class = torch.optim.Optimizer
        super(PytorchEngine, self).__init__(base, X, Y, A)

    def __move_model__(self):
        """Move model to CUDA."""
        if not self.base.cuda:
            self.cuda = False
        elif self.base.cuda:
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available")
            self.cuda = True
            self.device = torch.device(self.base.cuda)

        if self.cuda:
            self.adversary_model = self.adversary_model.to(self.device)
            self.predictor_model = self.predictor_model.to(self.device)

    def shuffle(self, X, Y, A):
        """Override base's shuffle to work with `torch.FloatTensor`."""
        idx = torch.randperm(X.shape[0])
        X = X[idx].view(X.size())
        Y = Y[idx].view(Y.size())
        A = A[idx].view(A.size())
        return X, Y, A

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

    def validate_input(self, X, Y, A):
        pass
