# Exploratory Data Analysis

- `output/eda` is a folder with resulting exploratory data analysis `.csv` files
- `output/ast_predictions.csv` is ALS model predictions without movie features
- `output/ast_features_predictions.csv` is ALS model predictions with movie features

We are used **mAP** (mean average precision) and **NDCG** (normalized discounted cumulative gain) metrics that evaluate
recommendation systems, and higher values for both indicate better performance.

Here's a brief explanation of what these two metrics mean:

1. **mAP**: Mean average precision calculates the average precision values for each user and then averages them over all
   users. Precision is the proportion of recommended items that are relevant. The mAP metric is useful when we want to
   understand the quality of the entire ranked list of recommendations (precision at every possible cut-off point in the
   list).

2. **NDCG**: Normalized Discounted Cumulative Gain is a position-aware metric that gives more importance to the
   top-ranked items. If a relevant item is placed at the top of the rank list, the recommendation model gets a higher
   score. This metric is useful when we are interested in the quality of the top of the recommendation list.

Looking at our results:

1. **ALS with features of movies**:
    - mAP: 0.861
    - NDCG: 0.988
2. **ALS without features**:
    - mAP: 0.834
    - NDCG: 0.973

The ALS without movie features appears to have a higher mAP, meaning it might be better at providing a set of relevant
recommendations across the entire ranked list. On the other hand, the ALS model with movie features has a slightly
higher NDCG score, indicating it might be better at ranking the most relevant items at the top of the list.

**Pros and Cons:**

- **ALS with movie features**:
    - Pros: Including movie features can make the model more robust and capable of handling items with little to no
      interaction data (cold start problem). This model seems to be better at ranking the most relevant items at the
      top.
    - Cons: It can be more complex and require more computational resources. Despite having more information, it doesn't
      necessarily outperform the simpler model in terms of mAP.

- **ALS without movie features**:
    - Pros: Simpler, potentially faster and requires less computational resources. It appears to provide slightly better
      performance in terms of mAP, indicating better overall precision across the ranked list.
    - Cons: Without item features, the model might struggle with new items (cold start problem). It also seems to be
      slightly worse at ranking the most relevant items at the top compared to the model with movie features.

So, the best model depends on the specific context and prioritization. If overall precision is most important, ALS
without features performed slightly better. If ranking the most relevant items at the top is more important, the ALS
with features model has a slight edge.

