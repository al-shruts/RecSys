import torch

from models.Base import Base


class EASE(Base):
    """
    Class for the Explicit Alternating Least Squares (EASE) recommendation model.

    This class extends the Base class and implements methods for fitting the model
    and predicting probabilities.

    Args:
        lambda_weight (float): Lambda weight for regularization. Defaults to 100.0.
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        lambda_weight (float): Lambda weight for regularization.
        weight (torch.Tensor): The learned weight matrix.

    Methods:
        fit(data: torch.Tensor) -> None:
            Fits the EASE model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Predicts probabilities using the EASE model.
    """

    def __init__(self, lambda_weight: float = 100.0, remove_used: bool = True) -> None:        
        """
        Initializes the EASE class.

        Args:
            lambda_weight (float): Lambda weight for regularization. Defaults to 100.0.
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """

        super().__init__(remove_used=remove_used)
        self.lambda_weight = lambda_weight
        self.weight = None

    def fit(self, data: torch.Tensor) -> None:        
        """
        Fits the EASE model.

        Args:
            data (torch.Tensor): The input data.
        """
        
        G = torch.sparse.mm(data.transpose(0, 1), data)
        G += self.lambda_weight * torch.eye(G.shape[0]).to_sparse()
        G = G.to_dense().to(self.device)

        P = torch.linalg.inv(G)

        self.weight = P / (-torch.diag(P))
        self.weight.fill_diagonal_(0.0)
        self.weight = self.weight.float()

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:        
        """
        Predicts probabilities using the EASE model.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        
        return torch.einsum('bi,ij->bj', data, self.weight)
    