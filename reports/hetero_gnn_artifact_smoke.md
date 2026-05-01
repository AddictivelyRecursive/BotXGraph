# RelGT-Inspired Hetero-GNN Bot/No-Bot Results

Model: RelGT-inspired relational attention network

## Dataset / Graph Summary
- User nodes: 250
- Tweet nodes: 250
- Hashtag nodes: 32
- URL nodes: 1
- Tweet text encoder: `tfidf_svd_fallback`
- User profile features: normalized z-score features
- URL node features: domain-level lexical features

## Edge Counts
- `('user', 'co_url', 'user')`: 702
- `('user', 'rev_co_url', 'user')`: 702
- `('user', 'similar_profile', 'user')`: 4784
- `('user', 'rev_similar_profile', 'user')`: 4784
- `('user', 'posts', 'tweet')`: 250
- `('tweet', 'rev_posts', 'user')`: 250
- `('tweet', 'contains', 'hashtag')`: 33
- `('hashtag', 'rev_contains', 'tweet')`: 33
- `('tweet', 'links', 'url')`: 39
- `('url', 'rev_links', 'tweet')`: 39

## Split
- Train: 175
- Validation: 37
- Test: 38
- Best epoch: 2

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | 0.7027 | 0.5401 |
| Test | 0.8158 | 0.7697 |

Bot precision: 0.8750  
Bot recall: 0.5385

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | 24 | 1 |
| True bot | 6 | 7 |

## Classification Report
```text
              precision    recall  f1-score   support

       human       0.80      0.96      0.87        25
         bot       0.88      0.54      0.67        13

    accuracy                           0.82        38
   macro avg       0.84      0.75      0.77        38
weighted avg       0.83      0.82      0.80        38

```

## Explainability Snapshot
Top user feature saliency:
- `log1p_favourites_count`: 0.1462
- `log1p_friends_count`: 0.1297
- `hashtag_count`: 0.1291
- `verified`: 0.1121
- `default_profile_image`: 0.1024
- `average_tweets_per_day`: 0.1005
- `description_length`: 0.0995
- `log1p_followers_count`: 0.0927

Relation importance into user prediction:
- `user__rev_similar_profile__user`: 0.2142
- `user__rev_co_url__user`: 0.1995
- `user__co_url__user`: 0.1990
- `user__similar_profile__user`: 0.1987
- `tweet__rev_posts__user`: 0.1886

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
Artifact directory: `reports/artifacts/hetero_gnn_artifact_smoke`
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
