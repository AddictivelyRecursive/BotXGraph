# RelGT-Inspired Hetero-GNN Bot/No-Bot Results

Model: RelGT-inspired relational attention network

## Dataset / Graph Summary
- User nodes: 2000
- Tweet nodes: 2000
- Hashtag nodes: 455
- URL nodes: 1
- Tweet text encoder: `tfidf_svd_fallback`
- User profile features: normalized z-score features
- URL node features: domain-level lexical features

## Edge Counts
- `('user', 'co_hashtag', 'user')`: 14
- `('user', 'rev_co_hashtag', 'user')`: 14
- `('user', 'similar_profile', 'user')`: 40000
- `('user', 'rev_similar_profile', 'user')`: 40000
- `('user', 'posts', 'tweet')`: 2000
- `('tweet', 'rev_posts', 'user')`: 2000
- `('tweet', 'contains', 'hashtag')`: 498
- `('hashtag', 'rev_contains', 'tweet')`: 498
- `('tweet', 'links', 'url')`: 245
- `('url', 'rev_links', 'tweet')`: 245

## Split
- Train: 1400
- Validation: 300
- Test: 300
- Best epoch: 3

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.7200 | 0.7159 |
| Test | 0.7367 | 0.7317 |

Bot precision: 0.5806  
Bot recall: 0.8654

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 131 | 65 |
| True bot | 14 | 90 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.90      0.67      0.77       196
         bot       0.58      0.87      0.69       104

    accuracy                           0.74       300
   macro avg       0.74      0.77      0.73       300
weighted avg       0.79      0.74      0.74       300

```

## Explainability Snapshot
Top user feature saliency:
- `log1p_friends_count`: 0.1608
- `description_length`: 0.1426
- `log1p_favourites_count`: 0.1387
- `account_age_days`: 0.1244
- `default_profile_image`: 0.1218
- `hashtag_count`: 0.1213
- `geo_enabled`: 0.1104
- `average_tweets_per_day`: 0.1070

Relation importance into user prediction:
- `user__rev_similar_profile__user`: 0.2165
- `user__similar_profile__user`: 0.1992
- `user__co_hashtag__user`: 0.1969
- `user__rev_co_hashtag__user`: 0.1969
- `tweet__rev_posts__user`: 0.1906

## Wrap-up Recommendation
Current `twitter-human-bots` data is sufficient for a wrap-up demo because it trains fast, produces clean bot/human metrics, and already supports user-text-URL heterogeneous modeling. If you want one extra easy next step later, reuse the already-wired `TwiBot-22` pipeline as the follow-up dataset rather than adding a brand-new source right now.
