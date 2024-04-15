import torch

import scipy.sparse as sps
from models.Base import Base


class SVD(Base):
    """
    Class for the Singular Value Decomposition (SVD) recommendation model.

    This class extends the Base class and implements methods for fitting the model
    and predicting probabilities.

    Args:
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        factors (dict): Dictionary containing the learned user and item factors.
        sigma (torch.Tensor): The diagonal matrix containing singular values.

    Methods:
        fit(data: torch.Tensor) -> None:
            Fits the SVD model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Predicts probabilities using the SVD model.
    """

    def __init__(self, remove_used: bool = True) -> None:
        """
        Initializes the SVD class.

        Args:
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """

        super().__init__(remove_used=remove_used)

        self.factors = {}
        self.sigma = None

    def fit(self, data: torch.Tensor) -> None:
        """
        Fits the SVD model.

        Args:
            data (torch.Tensor): The input data.
        """

        try:
            u, sigma, vt = torch.linalg.svd(data.to_dense())
            self.factors = {
                'user': u.to(self.device),
                'item': torch.einsum('fi->if', vt).to(self.device),
            }
            self.sigma = sigma.diag().to(self.device)

        except RuntimeError as error:
            data = data.coalesce()
            data = sps.csr_matrix((data.values(), (data.indices()[0], data.indices()[1])), shape=data.shape)

            u, sigma, vt = sps.linalg.svds(data, k=min(data.shape) - 1)
            self.factors = {
                "user": torch.tensor(u.copy(), dtype=torch.float, device=self.device),
                "item": torch.einsum("fi->if", torch.tensor(vt, dtype=torch.float, device=self.device)),
            }
            self.sigma = torch.tensor(sigma.copy(), dtype=torch.float, device=self.device).diag()

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor: 
        """
        Predicts probabilities using the SVD model.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        
        return torch.einsum('ui,if,jf->uj', data, self.factors['item'], self.factors['item'])
    