# RecSys


The following algorithms are currently implemented:
1. TopPopular
2. UserTopPopular
3. KNN, distance:
    * cooccurrence
    * cosine
    * jaccard
4. EASE
5. SVD


### Pipline example 
```
import torch
import pandas as pd
from RecSys import RecSys
from metrics.Metrics import Metrics
from models.TopPopular import TopPopular
from models.UserTopPopular import UserTopPopular
from models.KNN import KNN
from models.EASE import EASE
from models.SVD import SVD

# Load data
data = pd.read_csv(<Your_Path_Data>)

# Initialize Recommender System
rec_sys = RecSys(['user_id', 'item_id'])
rec_sys.set_sparse_matrix(data)

# Define the number of recommendations to generate
K = 10

# Preprocess test data
user_idx = sorted(data.user_id.unique())
test = data.groupby('user_id').apply(lambda group: group.sort_values(by='order_id', ascending=False).head(K)).reset_index(drop=True)
test = test.groupby('user_id')['item_id'].apply(list).reset_index()
test = torch.tensor(test['item_id'].apply(lambda x: x + x[:K - len(x)] if len(x) < K else x))

# Evaluate different models
result = {}
for model in [TopPopular(remove_used=False), UserTopPopular(remove_used=False), 
              KNN('cooccurrence', method='item', remove_used=False),
              KNN('cosine', method='item', remove_used=False), 
              KNN('jaccard', method='item', remove_used=False), 
              EASE(remove_used=False), SVD(remove_used=False)]:
    
    # Set current model for recommendation
    rec_sys.model = model
    rec_sys.model.fit(rec_sys.matrix)
    
    # Get recommendations
    recommendations = rec_sys.get_recommendations(K=K)
    
    # Compute evaluation metrics
    metrics = Metrics(test, torch.tensor(recommendations), K)
    result[rec_sys.model.__class__.__name__] = metrics.mean(metrics.apK)
    
    # Save recommendations to CSV
    rec_sys.recommendations_to_csv(user_idx, recommendations, f"recommendations-{rec_sys.model.__class__.__name__}.csv", ['user_id', 'item_id'])

# Print results
import pprint
pprint.pprint(result)

```