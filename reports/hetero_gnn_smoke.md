# Hetero-GNN Bot/No-Bot Results

## Dataset / Graph Summary
- User nodes: 1000
- Tweet nodes: 1000
- Hashtag nodes: 233
- URL nodes: 1

## Edge Counts
- `('user', 'co_hashtag', 'user')`: 2
- `('user', 'rev_co_hashtag', 'user')`: 2
- `('user', 'co_url', 'user')`: 8742
- `('user', 'rev_co_url', 'user')`: 8742
- `('user', 'similar_profile', 'user')`: 19871
- `('user', 'rev_similar_profile', 'user')`: 19871
- `('user', 'posts', 'tweet')`: 1000
- `('tweet', 'rev_posts', 'user')`: 1000
- `('tweet', 'contains', 'hashtag')`: 246
- `('hashtag', 'rev_contains', 'tweet')`: 246
- `('tweet', 'links', 'url')`: 114
- `('url', 'rev_links', 'tweet')`: 114

## Split
- Train: 700
- Validation: 150
- Test: 150
- Best epoch: 20

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.7000 | 0.6451 |
| Test | 0.6067 | 0.5506 |

Bot precision: 0.4130  
Bot recall: 0.3725

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 72 | 27 |
| True bot | 32 | 19 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.69      0.73      0.71        99
         bot       0.41      0.37      0.39        51

    accuracy                           0.61       150
   macro avg       0.55      0.55      0.55       150
weighted avg       0.60      0.61      0.60       150

```
