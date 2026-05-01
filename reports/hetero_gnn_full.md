# RelGT-Inspired Hetero-GNN Bot/No-Bot Results

Model: RelGT-inspired relational attention network

## Dataset / Graph Summary
- User nodes: 37438
- Tweet nodes: 37438
- Hashtag nodes: 7211
- URL nodes: 10
- Tweet text encoder: `tfidf_svd_fallback`
- User profile features: normalized z-score features
- URL node features: domain-level lexical features

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
- Best epoch: 118

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.8634 | 0.8489 |
| Test | 0.8634 | 0.8493 |

Bot precision: 0.7698  
Bot recall: 0.8396

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 3284 | 468 |
| True bot | 299 | 1565 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.92      0.88      0.90      3752
         bot       0.77      0.84      0.80      1864

    accuracy                           0.86      5616
   macro avg       0.84      0.86      0.85      5616
weighted avg       0.87      0.86      0.86      5616

```

## Explainability Snapshot
Top user feature saliency:
- `log1p_favourites_count`: 0.6281
- `log1p_followers_count`: 0.4497
- `average_tweets_per_day`: 0.3752
- `verified`: 0.3441
- `log1p_statuses_count`: 0.3317
- `log1p_friends_count`: 0.2627
- `account_age_days`: 0.1802
- `url_count`: 0.1748

Relation importance into user prediction:
- `tweet__rev_posts__user`: 0.4021
- `user__rev_similar_profile__user`: 0.1906
- `user__similar_profile__user`: 0.1267
- `user__co_hashtag__user`: 0.0715
- `user__rev_co_hashtag__user`: 0.0713
- `user__co_url__user`: 0.0689
- `user__rev_co_url__user`: 0.0689

## Visual Outputs
- `training_curves.png`
- `edge_type_counts.png`
- `confusion_matrix.png`
- `feature_saliency.png`
- `relation_importance.png`
- `graph_schema.png`
- `user_projection.png`
- `similarity_subgraph.png`

## Stored Artifacts
Artifact directory: `reports/artifacts/hetero_gnn_full`
- `run_config.json`
- `history.json`
- `final_outputs.npz`
- `final_user_embeddings.npz`
- `best_model.pt`
- `epochs/epoch_XXX/metrics.json`
- `epochs/epoch_XXX/*_predictions.npz`
- `epochs/epoch_XXX/user_logits_probs.npz`
- `epochs/epoch_XXX/user_embeddings.npz`
- `epochs/epoch_XXX/checkpoint.pt`

## Wrap-up Recommendation
Current `twitter-human-bots` data is sufficient for a wrap-up demo because it trains fast, produces clean bot/human metrics, and already supports user-text-URL heterogeneous modeling. If you want one extra easy next step later, reuse the already-wired `TwiBot-22` pipeline as the follow-up dataset rather than adding a brand-new source right now.
