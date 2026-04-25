# Fast Bot/No-Bot KG Baseline

## Dataset / Graph Summary
- User nodes: 37438
- Tweet nodes: 37438
- Hashtag nodes: 7211
- URL nodes: 10

## Edge Counts
- `('user', 'follows', 'user')`: 0
- `('user', 'posts', 'tweet')`: 37438
- `('tweet', 'contains', 'hashtag')`: 10862
- `('tweet', 'links', 'url')`: 4774

## Feature Set
log1p_followers_count, log1p_friends_count, log1p_statuses_count, log1p_favourites_count, average_tweets_per_day, account_age_days, verified, geo_enabled, default_profile, default_profile_image, hashtag_count, url_count, description_length

## Split
- Train: 26206
- Validation: 5616
- Test: 5616

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.8748 | 0.8531 |
| Test | 0.8803 | 0.8605 |

Bot precision: 0.8643  
Bot recall: 0.7586

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 3530 | 222 |
| True bot | 450 | 1414 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.89      0.94      0.91      3752
         bot       0.86      0.76      0.81      1864

    accuracy                           0.88      5616
   macro avg       0.88      0.85      0.86      5616
weighted avg       0.88      0.88      0.88      5616

```

Note: This is a fast KG-compatible baseline, not final deep GNN training.
