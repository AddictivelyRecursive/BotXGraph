# Hetero-GNN Bot/No-Bot Results

## Dataset / Graph Summary
- User nodes: 37438
- Tweet nodes: 37438
- Hashtag nodes: 7211
- URL nodes: 10

## Edge Counts
- `('user', 'co_hashtag', 'user')`: 4618
- `('user', 'rev_co_hashtag', 'user')`: 4618
- `('user', 'co_url', 'user')`: 8
- `('user', 'rev_co_url', 'user')`: 8
- `('user', 'similar_profile', 'user')`: 748760
- `('user', 'rev_similar_profile', 'user')`: 748760
- `('user', 'posts', 'tweet')`: 37438
- `('tweet', 'rev_posts', 'user')`: 37438
- `('tweet', 'contains', 'hashtag')`: 10862
- `('hashtag', 'rev_contains', 'tweet')`: 10862
- `('tweet', 'links', 'url')`: 4774
- `('url', 'rev_links', 'tweet')`: 4774

## Split
- Train: 26206
- Validation: 5616
- Test: 5616
- Best epoch: 1

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.6455 | 0.4462 |
| Test | 0.6394 | 0.4404 |

Bot precision: 0.3002  
Bot recall: 0.0649

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 3470 | 282 |
| True bot | 1743 | 121 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.67      0.92      0.77      3752
         bot       0.30      0.06      0.11      1864

    accuracy                           0.64      5616
   macro avg       0.48      0.49      0.44      5616
weighted avg       0.54      0.64      0.55      5616

```
