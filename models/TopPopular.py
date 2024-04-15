import torch

from models.Base import Base


class TopPopular(Base):
    """
    Class for the Top Popular recommendation model.

    This class extends the Base class and implements methods for fitting the model
    and predicting probabilities based on item popularity.

    Args:
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        logits (torch.Tensor): The popularity scores of items.

    Methods:
        fit(data: torch.Tensor) -> None:
            Fits the Top Popular model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Predicts probabilities using the Top Popular model.
    """

    def __init__(self, remove_used: bool = True) -> None:
        """
        Initializes the TopPopular class.

        Args:
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """

        super().__init__(remove_used=remove_used)
        self.logits = None

    def fit(self, data: torch.Tensor) -> None:
        """
        Fits the Top Popular model.

        Args:
            data (torch.Tensor): The input data.
        """

        item_freq = data.sum(axis=0).to_dense().view(-1).to(self.device)
        self.logits = torch.zeros_like(item_freq, device=self.device).scatter_(
            dim=-1,
            index=item_freq.argsort(descending=True),
            src=torch.arange(item_freq.size(-1), 0, -1, dtype=torch.float, device=self.device)
        )

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor: 
        """
        Predicts probabilities using the Top Popular model.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        
        return self.logits.repeat(data.size(0), 1)
        