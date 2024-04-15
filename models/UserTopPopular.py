import torch

from models.Base import Base


class UserTopPopular(Base):
    """
    Class for the User Top Popular recommendation model.

    This class extends the Base class and implements methods for fitting the model
    and predicting probabilities based on user-specific item popularity.

    Args:
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        logits (torch.Tensor): The popularity scores of items for each user.

    Methods:
        fit(data: torch.Tensor) -> None:
            Fits the User Top Popular model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Predicts probabilities using the User Top Popular model.
    """

    def __init__(self, remove_used: bool = True) -> None:
        """
        Initializes the UserTopPopular class.

        Args:
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """

        super().__init__(remove_used=remove_used)
        self.logits = None

    def fit(self, data: torch.Tensor) -> None:
        """
        Fits the User Top Popular model.

        Args:
            data (torch.Tensor): The input data.
        """

        item_freq = data.to_dense().to(self.device)
        self.logits = torch.zeros_like(item_freq, device=self.device).scatter_(
            dim=1,
            index=item_freq.argsort(descending=True, dim=1),
            src=torch.arange(item_freq.size(1), 0, -1, dtype=torch.float, device=self.device).unsqueeze(0).repeat(item_freq.size(0), 1)
        )

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predicts probabilities using the User Top Popular model.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
             
        return self.logits
