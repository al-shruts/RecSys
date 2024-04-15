import torch

from typing import Optional, Callable


class Metrics:
    """
    A class for evaluating recommendation metrics.

    This class computes various evaluation metrics such as Recall, Precision, Average Precision, DCG, nDCG, etc.

    Args:
        actual (torch.Tensor): A tensor containing the actual items.
        predicted (torch.Tensor): A tensor containing the predicted items.
        K (Optional[int]): The number of recommendations to consider. Defaults to None.

    Attributes:
        K (int): The number of recommendations to consider.
        actual (torch.Tensor): A tensor containing the actual items.
        predicted (torch.Tensor): A tensor containing the predicted items.
        relevant_items (torch.Tensor): A tensor indicating whether predicted items are relevant or not.

    Methods:
        rr() -> torch.Tensor:
            Computes the Reciprocal Rank (RR) metric.
        hr() -> torch.Tensor:
            Computes the Hit Rate (HR) metric.
        precision() -> torch.Tensor:
            Computes Precision at K (P@K).
        apK() -> torch.Tensor:
            Computes the Average Precision at K (AP@K).
        dcg() -> torch.Tensor:
            Computes the Discounted Cumulative Gain (DCG).
        idcg() -> torch.Tensor:
            Computes the Ideal Discounted Cumulative Gain (IDCG).
        ndcg() -> torch.Tensor:
            Computes the normalized Discounted Cumulative Gain (nDCG).
        mean(metric: Callable) -> float:
            Computes the mean of a given metric.
    """
        
    def __init__(self, actual: torch.Tensor, predicted: torch.Tensor, K: Optional[int] = None) -> None:
        """
        Initializes the Metrics class.

        Args:
            actual (torch.Tensor): A tensor containing the actual items.
            predicted (torch.Tensor): A tensor containing the predicted items.
            K (Optional[int]): The number of recommendations to consider. Defaults to None.
        """
                
        self.K = K if K is not None else predicted.size(-1)
        self.actual = actual
        self.predicted = predicted[:, :K]

        self.relevant_items = torch.eq(self.predicted.unsqueeze(2), self.actual.unsqueeze(1)).any(dim=2)

    def rr(self) -> torch.Tensor:
        """
        Computes the Reciprocal Rank (RR) metric.

        Returns:
            torch.Tensor: The RR scores.
        """
                
        result = torch.argmax(self.relevant_items.int(), dim=1) + 1
        result = 1 / result
        result[self.relevant_items.sum(dim=1) == 0] = 0
        
        return result

    def hr(self) -> torch.Tensor: 
        """
        Computes the Hit Rate (HR) metric.

        Returns:
            torch.Tensor: The HR scores.
        """

        return torch.any(self.relevant_items, dim=1).float()
    
    def precision(self) -> torch.Tensor:
        """
        Computes Precision at K (P@K).

        Returns:
            torch.Tensor: The Precision scores.
        """

        return torch.cumsum(self.relevant_items, dim=1) / torch.arange(1, self.K+1)

    def apK(self) -> torch.Tensor:
        """
        Computes the Average Precision at K (AP@K).

        Returns:
            torch.Tensor: The AP@K scores.
        """

        return torch.sum(self.precision() * self.relevant_items, dim=1) / self.K
    
    def dcg(self) -> torch.Tensor:
        """
        Computes the Discounted Cumulative Gain (DCG).

        Returns:
            torch.Tensor: The DCG scores.
        """

        return torch.sum(self.relevant_items / torch.log2(torch.arange(2, self.K+2)), dim=1)

    def idcg(self) -> torch.Tensor:
        """
        Computes the Ideal Discounted Cumulative Gain (IDCG).

        Returns:
            torch.Tensor: The IDCG scores.
        """

        return torch.sum(torch.ones(self.actual.size(-1)) / torch.log2(torch.arange(2, self.actual.size(-1)+2)))
    
    def ndcg(self) -> torch.Tensor:
        """
        Computes the normalized Discounted Cumulative Gain (nDCG).

        Returns:
            torch.Tensor: The nDCG scores.
        """
                
        return self.dcg() / self.idcg()

    def mean(self, metric: Callable) -> float:
        """
        Computes the mean of a given metric.

        Args:
            metric (Callable): The metric function.

        Returns:
            float: The mean of the metric.
        """

        return metric().mean().item()
