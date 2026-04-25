# Fast Bot/No-Bot KG Baseline

## Dataset / Graph Summary
- User nodes: 1000
- Tweet nodes: 1000
- Hashtag nodes: 233
- URL nodes: 1

## Edge Counts
- `('user', 'follows', 'user')`: 0
- `('user', 'posts', 'tweet')`: 1000
- `('tweet', 'contains', 'hashtag')`: 246
- `('tweet', 'links', 'url')`: 114

## Feature Set
log1p_followers_count, log1p_friends_count, log1p_statuses_count, log1p_favourites_count, average_tweets_per_day, account_age_days, verified, geo_enabled, default_profile, default_profile_image, hashtag_count, url_count, description_length

## Split
- Train: 700
- Validation: 150
- Test: 150

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.8067 | 0.7791 |
| Test | 0.8200 | 0.7871 |

Bot precision: 0.8000  
Bot recall: 0.6275

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 91 | 8 |
| True bot | 19 | 32 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.83      0.92      0.87        99
         bot       0.80      0.63      0.70        51

    accuracy                           0.82       150
   macro avg       0.81      0.77      0.79       150
weighted avg       0.82      0.82      0.81       150

```

Note: This is a fast KG-compatible baseline, not final deep GNN training.
