# DCCM Validation Summary

Samples used: 14 | Features: 52

CV accuracy: 0.143

CV macro-F1: 0.125

Permutation acc mean ± sd: 0.552 ± 0.137

Top Spearman correlations (abs):

| feature                         | target                  |       rho |        p |
|:--------------------------------|:------------------------|----------:|---------:|
| hub_std|meanAcrossStates        | max_continuous_motion_h | -0.391521 | 0.208174 |
| hub_std|meanAcrossStates        | max_velocity            | -0.324506 | 0.257649 |
| mean_hub_score|meanAcrossStates | max_continuous_motion_h | -0.320335 | 0.310049 |
| mean_hub_score|meanAcrossStates | mean_velocity           | -0.315676 | 0.271572 |
| mean_hub_score|meanAcrossStates | max_velocity            | -0.306846 | 0.285933 |
| max_hub_score|meanAcrossStates  | max_velocity            | -0.253866 | 0.381142 |
| hub_std|meanAcrossStates        | mean_velocity           | -0.253866 | 0.381142 |
| max_hub_score|meanAcrossStates  | mean_velocity           | -0.24062  | 0.407297 |