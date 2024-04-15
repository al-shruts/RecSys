import torch

from typing import Optional
from models.Base import Base


class KNN(Base):
    """
    Class for the k-Nearest Neighbors (KNN) recommendation model.

    This class extends the Base class and implements methods for fitting the model
    and predicting probabilities based on different similarity measures.

    Args:
        similarity (str): The type of similarity measure to use ('cooccurrence', 'cosine', or 'jaccard').
        K (Optional[int]): The number of nearest neighbors to consider. Defaults to None.
        n_threshold (Optional[int]): Threshold for the count of co-occurrences. Defaults to None.
        method (str): The method to use for computing similarity ('user' or 'item'). Defaults to 'user'.
        remove_used (bool): Whether to remove already used items during prediction. Defaults to True.

    Attributes:
        similarity (Callable): The similarity function based on the chosen similarity measure.
        K (int): The number of nearest neighbors to consider.
        n_threshold (int): Threshold for the count of co-occurrences.
        method (str): The method used for computing similarity.
        similarity_matrix (torch.Tensor): The similarity matrix computed during fitting.

    Methods:
        cooccurrence(data: torch.Tensor) -> torch.Tensor:
            Computes the co-occurrence similarity matrix.
        cosine(data: torch.Tensor) -> torch.Tensor:
            Computes the cosine similarity matrix.
        jaccard(data: torch.Tensor) -> torch.Tensor:
            Computes the Jaccard similarity matrix.
        fit(data: torch.Tensor) -> None:
            Fits the KNN model.
        predict_proba(data: torch.Tensor) -> torch.Tensor:
            Predicts probabilities using the KNN model.
    """

    def __init__(self, similarity: str, K: Optional[int] = None, 
                 n_threshold: Optional[int] = None, method: str = 'user', 
                 remove_used: bool = True) -> None:
        """
        Initializes the KNN class.

        Args:
            similarity (str): The type of similarity measure to use ('cooccurrence', 'cosine', or 'jaccard').
            K (Optional[int]): The number of nearest neighbors to consider. Defaults to None.
            n_threshold (Optional[int]): Threshold for the count of co-occurrences. Defaults to None.
            method (str): The method to use for computing similarity ('user' or 'item'). Defaults to 'user'.
            remove_used (bool): Whether to remove already used items during prediction. Defaults to True.
        """

        assert similarity in ('cooccurrence', 'cosine', 'jaccard')
        assert method in ('user', 'item')
        
        super().__init__(remove_used=remove_used)        
        self.similarity = getattr(self, similarity)
        self.K = K
        self.n_threshold = n_threshold
        self.method = method

        self.__class__.__name__ = f"KNN_{similarity}"
        
        self.similarity_matrix = None

    def cooccurrence(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the co-occurrence similarity matrix.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The co-occurrence similarity matrix.
        """

        if self.method == 'user':
            # similarity_matrix ~ (num users, num users)
            similarity_matrix = torch.sparse.mm(data, data.transpose(0, 1)).to_dense().to(self.device)
        else: 
            # similarity_matrix ~ (num items, num items)
            similarity_matrix = torch.sparse.mm(data.transpose(0, 1), data).to_dense().to(self.device)  
            
        if self.n_threshold is not None:
            similarity_matrix *= similarity_matrix > self._count_threshold
            
        return similarity_matrix
        
    def cosine(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity matrix.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The cosine similarity matrix.
        """

        if self.method == 'user':
            # similarity_matrix ~ (num users, num users)
            similarity_matrix = torch.sparse.mm(data, data.transpose(0, 1)).to_dense().to(self.device)
            # sum_of_squares ~ (num users)
            sum_of_squares = data.pow(2).sum(dim=1).view(-1).sqrt() 
        else:
            # sim_matrix ~ (num items, num items)
            similarity_matrix = torch.sparse.mm(data.transpose(0, 1), data).to_dense().to(self.device)
            # sum_of_squares ~ (num items)
            sum_of_squares = data.pow(2).sum(dim=0).to_dense().view(-1).sqrt()            
            
        if self.n_threshold is not None:
            similarity_matrix *= similarity_matrix > self._count_threshold
            
        denominator = torch.einsum("i,j->ij", sum_of_squares, sum_of_squares) + 1e-13
        similarity_matrix = similarity_matrix / denominator
        similarity_matrix.fill_diagonal_(0.0)
        
        return similarity_matrix
    
    def jaccard(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jaccard similarity matrix.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The Jaccard similarity matrix.
        """

        if self.method == 'user':
            # similarity_matrix ~ (num users, num users)
            similarity_matrix = torch.sparse.mm(data, data.transpose(0, 1)).to_dense().to(self.device)
        else:
            # similarity_matrix ~ (num items, num items)
            similarity_matrix = torch.sparse.mm(data.transpose(0, 1), data).to_dense().to(self.device)        
            
        if self.n_threshold is not None:
            similarity_matrix *= similarity_matrix > self._count_threshold
            
        diagonal = similarity_matrix.diagonal()
        return similarity_matrix / (diagonal.unsqueeze(0) + diagonal.unsqueeze(-1) - similarity_matrix)

    def fit(self, data: torch.Tensor) -> None:
        """
        Fits the KNN model.

        Args:
            data (torch.Tensor): The input data.
        """

        if self.K is None:            
            self.K = data.shape[0] if self.method == 'user' else data.shape[1]
            
        similarity_matrix = self.similarity(data)
        relevant = torch.topk(similarity_matrix, k=self.K, dim=-1)
        self.similarity_matrix = torch.zeros_like(similarity_matrix).scatter_(dim=-1, index=relevant.indices, src=relevant.values)

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predicts probabilities using the KNN model.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        
        if self.method == 'user':
            data = data.T
        
        return torch.einsum('bi,ij->bj', data, self.similarity_matrix)
    