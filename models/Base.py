import torch

from abc import ABC, abstractmethod


class Base(ABC):
    """
    An abstract base class for recommendation models.

    This class provides methods to fit a model, predict probabilities, and make predictions.

    Args:
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        remove_used (bool): Whether to remove already used items during prediction.
        device (torch.device): The device to use for computation.
        matrix: Placeholder for matrix data.

    Methods:
        fit(data: torch.Tensor) -> None:
            Abstract method to fit the model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Abstract method to predict probabilities.
        predict(data: torch.Tensor) -> torch.Tensor:
            Predicts the top items for the input data.
        to(device: torch.device) -> 'Base':
            Move the model to the specified device.
    """

    def __init__(self, remove_used: bool = True) -> None:
        """
        Initializes the Base class.

        Args:
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """
                
        self.remove_used = remove_used
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.matrix = None

    @abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """
        Abstract method to fit the model.

        Args:
            data (torch.Tensor): The input data.
        """
                
        pass

    @abstractmethod
    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:                
        """
        Abstract method to predict probabilities.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        
        pass

    def predict(self, data: torch.Tensor) -> torch.Tensor:                
        """
        Predicts the top items for the input data.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted top items.
        """
        
        scores = self.predict_proba(data)

        if self.remove_used:            
            scores[data.gt(0)] = float('-inf')

        return scores.argsort(descending=True)

    def to(self, device: torch.device) -> 'Base':                
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.

        Returns:
            Base: The model object.
        """
        
        self.device = device
        return self
