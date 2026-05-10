# Faithfulness Summary

Top masked features: log1p_favourites_count, log1p_followers_count, average_tweets_per_day

| perturbation | test_accuracy | test_macro_f1 | test_bot_precision | test_bot_recall | mean_abs_prob_shift |
| --- | --- | --- | --- | --- | --- |
| full_graph | 0.8636 | 0.8495 | 0.7699 | 0.8401 | 0.0000 |
| drop_similar_profile | 0.8095 | 0.7689 | 0.7836 | 0.5885 | 0.1179 |
| drop_hashtag_and_url_edges | 0.8339 | 0.8222 | 0.7014 | 0.8696 | 0.0665 |
| mask_top_user_features | 0.7425 | 0.6888 | 0.6474 | 0.4925 | 0.1993 |
