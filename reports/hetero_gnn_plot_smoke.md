# RelGT-Inspired Hetero-GNN Bot/No-Bot Results

Model: RelGT-inspired relational attention network

## Dataset / Graph Summary
- User nodes: 400
- Tweet nodes: 400
- Hashtag nodes: 72
- URL nodes: 1
- Tweet text encoder: `tfidf_svd_fallback`
- User profile features: normalized z-score features
- URL node features: domain-level lexical features

## Edge Counts
- `('user', 'co_url', 'user')`: 1482
- `('user', 'rev_co_url', 'user')`: 1482
- `('user', 'similar_profile', 'user')`: 7836
- `('user', 'rev_similar_profile', 'user')`: 7836
- `('user', 'posts', 'tweet')`: 400
- `('tweet', 'rev_posts', 'user')`: 400
- `('tweet', 'contains', 'hashtag')`: 73
- `('hashtag', 'rev_contains', 'tweet')`: 73
- `('tweet', 'links', 'url')`: 54
- `('url', 'rev_links', 'tweet')`: 54

## Split
- Train: 280
- Validation: 60
- Test: 60
- Best epoch: 2

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.6333 | 0.5111 |
| Test | 0.7500 | 0.6865 |

Bot precision: 0.6923  
Bot recall: 0.4500

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 36 | 4 |
| True bot | 11 | 9 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.77      0.90      0.83        40
         bot       0.69      0.45      0.55        20

    accuracy                           0.75        60
   macro avg       0.73      0.68      0.69        60
weighted avg       0.74      0.75      0.73        60

```

## Explainability Snapshot
Top user feature saliency:
- `log1p_friends_count`: 0.1359
- `description_length`: 0.1299
- `log1p_favourites_count`: 0.1206
- `default_profile_image`: 0.1179
- `hashtag_count`: 0.1174
- `average_tweets_per_day`: 0.1054
- `log1p_followers_count`: 0.1028
- `account_age_days`: 0.0953

Relation importance into user prediction:
- `user__rev_similar_profile__user`: 0.2124
- `user__similar_profile__user`: 0.1998
- `user__co_url__user`: 0.1989
- `user__rev_co_url__user`: 0.1987
- `tweet__rev_posts__user`: 0.1902

## Visual Outputs
- `training_curves.png`
- `edge_type_counts.png`
- `confusion_matrix.png`
- `feature_saliency.png`
- `relation_importance.png`
- `graph_schema.png`
- `user_projection.png`
- `similarity_subgraph.png`

## Wrap-up Recommendation
Current `twitter-human-bots` data is sufficient for a wrap-up demo because it trains fast, produces clean bot/human metrics, and already supports user-text-URL heterogeneous modeling. If you want one extra easy next step later, reuse the already-wired `TwiBot-22` pipeline as the follow-up dataset rather than adding a brand-new source right now.
