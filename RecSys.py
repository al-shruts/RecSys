import csv
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from typing import List, Iterable, Optional


class RecSys:
    """
    A recommendation system class.

    This class provides methods to build a recommendation system,
    set sparse matrices, get recommendations, and export them to a CSV file.

    Args:
        column_names (List[str]): A list containing column names for 'user', 'item', and optionally 'values'.

    Attributes:
        column_names (dict): A dictionary containing column names for 'user', 'item', and optionally 'values'.
        model: The recommendation model.
        matrix: The sparse matrix representing user-item interactions.
        encoder (dict): A dictionary containing LabelEncoders for 'user' and 'item'.

    Methods:
        set_sparse_matrix(df: pd.DataFrame) -> None:
            Set the sparse matrix using the input DataFrame.
        get_recommendations(user_idx: Optional[List[int]] = None, K: Optional[int] = None) -> List[List[int]]:
            Get recommendations for users.
        recommendations_to_csv(user_idx: Iterable, item_idx: Iterable, ouput_file: str = 'recommendations.csv',
                               titles: List[str] = ['user_id', 'item_id']) -> None:
            Export recommendations to a CSV file.
    """

    def __init__(self, column_names: List[str]) -> None:   
        """
        Initializes the RecSys class.

        Args:
            column_names (List[str]): A list containing column names for 'user', 'item', and optionally 'values'.
                The length of the list must be 2 or 3.
        """
                             
        assert len(column_names) == 3 or len(column_names) == 2

        self.column_names = {
            'user': column_names[0],
            'item': column_names[1],
            'values': column_names[2] if len(column_names) == 3 else None
        }

        self.model = None
        self.matrix = None
        self.encoder = {
            'user': LabelEncoder(),
            'item': LabelEncoder()
        }        

    def set_sparse_matrix(self, df: pd.DataFrame) -> None:
        """
        Set the sparse matrix using the input DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing user-item interactions.
        """
                                
        user_idx = torch.tensor(self.encoder['user'].fit_transform(df[self.column_names['user']].values))
        item_idx = torch.tensor(self.encoder['item'].fit_transform(df[self.column_names['item']].values))
        values = torch.ones_like(user_idx) if self.column_names['values'] is None else torch.tensor(df[self.column_names['values']].values)

        self.matrix = torch.sparse_coo_tensor(
            torch.vstack((user_idx, item_idx)),
            values,
            (len(self.encoder['user'].classes_), len(self.encoder['item'].classes_)),
            dtype=torch.float
        )

    def get_recommendations(self, user_idx: Optional[List[int]] = None, K: Optional[int] = None) -> List[List[int]]:        
        """
        Get recommendations for users.

        Args:
            user_idx (Optional[List[int]]): Indices of users. If None, recommendations for all users are returned.
            K (Optional[int]): Number of recommendations to return for each user. If None, all available items are returned.

        Returns:
            List[List[int]]: A list of recommendation lists for each user.
        """
        
        if user_idx is None:
            user_idx = torch.arange(len(self.encoder['user'].classes_))
        else:
            user_idx = self.encoder['user'].transform(user_idx)

        if K is None:
            K = len(self.encoder['item'].classes_)

        predicted = self.model.predict(self.matrix.to_dense())[user_idx, :K]
        predicted =  self.encoder['item'].inverse_transform(predicted.flatten())

        return predicted.reshape(len(user_idx), -1).tolist()

    @staticmethod
    def recommendations_to_csv(user_idx: Iterable, item_idx: Iterable, ouput_file: str = 'recommendations.csv', titles: List[str] = ['user_id', 'item_id']) -> None:       
        """
        Export recommendations to a CSV file.

        Args:
            user_idx (Iterable): Iterable of user IDs.
            item_idx (Iterable): Iterable of recommendation lists for each user.
            ouput_file (str, optional): Output CSV file name. Defaults to 'recommendations.csv'.
            titles (List[str], optional): Column titles for user and item IDs. Defaults to ['user_id', 'item_id'].
        """
        
        with open(ouput_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(titles)
            for user_id, item_list in zip(user_idx, item_idx):
                writer.writerow([user_id, ' '.join(map(str, item_list))])
