# Robustness Summary

| perturbation | test_accuracy | test_macro_f1 | test_bot_precision | test_bot_recall | mean_abs_prob_shift |
| --- | --- | --- | --- | --- | --- |
| clean | 0.8625 | 0.8485 | 0.7676 | 0.8401 | 0.0000 |
| light_rewrite | 0.8275 | 0.8095 | 0.7208 | 0.7838 | 0.0889 |
| hashtag_injection | 0.8264 | 0.8033 | 0.7433 | 0.7285 | 0.1111 |
| url_injection | 0.8316 | 0.8112 | 0.7406 | 0.7580 | 0.0889 |
| combined_attack | 0.8413 | 0.8198 | 0.7686 | 0.7468 | 0.0910 |
