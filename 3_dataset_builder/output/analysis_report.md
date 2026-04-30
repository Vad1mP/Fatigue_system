# Analysis dataset report

## Summary
- **n_rows**: 22
- **n_columns**: 520
- **date_min**: 2026-04-01
- **date_max**: 2026-04-22
- **n_targets**: 4
- **n_usable_targets**: 4
- **n_features_total**: 510
- **n_usable_features**: 374

## Warnings
- context_computed: suspicious column names: ['Sleep_qality', 'Sleepness_before', 'Sikness_before', 'Slakness_before', 'Speepness_after', 'Sikness_after', 'Slakness_after']
- 105 features have low uniqueness or low IQR
- 4 targets have ML/data readiness warnings

## Coverage By Day
| date       |   n_ecg_records_expected |   n_ecg_records_found |   n_ecg_records_valid_hrv |   n_ecg_records_valid_morphology | complete_core_day   |
|:-----------|-------------------------:|----------------------:|--------------------------:|---------------------------------:|:--------------------|
| 2026-04-01 |                        6 |                     6 |                         6 |                                3 | False               |
| 2026-04-02 |                        6 |                     6 |                         6 |                                4 | True                |
| 2026-04-03 |                        6 |                     6 |                         6 |                                1 | True                |
| 2026-04-04 |                        6 |                     4 |                         4 |                                0 | False               |
| 2026-04-05 |                        6 |                     6 |                         6 |                                2 | True                |
| 2026-04-06 |                        6 |                     6 |                         6 |                                2 | False               |
| 2026-04-07 |                        6 |                     6 |                         6 |                                2 | True                |
| 2026-04-08 |                        6 |                     6 |                         6 |                                2 | True                |
| 2026-04-09 |                        6 |                     6 |                         6 |                                3 | True                |
| 2026-04-10 |                        6 |                     6 |                         6 |                                1 | False               |
| 2026-04-11 |                        6 |                     6 |                         6 |                                1 | True                |
| 2026-04-12 |                        6 |                     6 |                         6 |                                2 | True                |
| 2026-04-13 |                        6 |                     6 |                         6 |                                6 | False               |
| 2026-04-14 |                        6 |                     6 |                         6 |                                4 | False               |
| 2026-04-15 |                        6 |                     6 |                         6 |                                4 | False               |
| 2026-04-16 |                        6 |                     6 |                         6 |                                5 | False               |
| 2026-04-17 |                        6 |                     6 |                         6 |                                3 | True                |
| 2026-04-18 |                        6 |                     6 |                         6 |                                1 | False               |
| 2026-04-19 |                        6 |                     6 |                         6 |                                0 | False               |
| 2026-04-20 |                        6 |                     6 |                         6 |                                0 | True                |

Showing first 20 of 22 rows.

## Ecg Quality
| phase   | record_type   | segment_label   |   n_total |   n_dates |   hrv_valid_n |   hrv_valid_ratio |   morphology_valid_n |   morphology_valid_ratio |
|:--------|:--------------|:----------------|----------:|----------:|--------------:|------------------:|---------------------:|-------------------------:|
| after   | breath_in     | end             |        22 |        22 |            22 |          1        |                    2 |                0.0909091 |
| after   | breath_in     | full            |        22 |        22 |            22 |          1        |                    3 |                0.136364  |
| after   | breath_in     | start           |        22 |        22 |            22 |          1        |                    8 |                0.363636  |
| after   | breath_out    | end             |        22 |        22 |            22 |          1        |                   11 |                0.5       |
| after   | breath_out    | full            |        22 |        22 |            22 |          1        |                   11 |                0.5       |
| after   | breath_out    | start           |        22 |        22 |            22 |          1        |                   15 |                0.681818  |
| after   | long          | full            |        21 |        21 |            21 |          1        |                    6 |                0.285714  |
| after   | sit           | full            |        22 |        22 |            22 |          1        |                   16 |                0.727273  |
| after   | squat         | window          |       262 |        22 |           262 |          1        |                    0 |                0         |
| after   | stand         | full            |        22 |        22 |            22 |          1        |                    3 |                0.136364  |
| before  | breath_in     | end             |        22 |        22 |            22 |          1        |                    4 |                0.181818  |
| before  | breath_in     | full            |        22 |        22 |            22 |          1        |                    3 |                0.136364  |
| before  | breath_in     | start           |        22 |        22 |            22 |          1        |                   10 |                0.454545  |
| before  | breath_out    | end             |        22 |        22 |            21 |          0.954545 |                    8 |                0.363636  |
| before  | breath_out    | full            |        22 |        22 |            21 |          0.954545 |                   12 |                0.545455  |
| before  | breath_out    | start           |        22 |        22 |            22 |          1        |                   11 |                0.5       |
| before  | long          | full            |        21 |        21 |            21 |          1        |                    7 |                0.333333  |
| before  | sit           | full            |        22 |        22 |            22 |          1        |                   15 |                0.681818  |
| before  | squat         | window          |       257 |        22 |           257 |          1        |                    0 |                0         |
| before  | stand         | full            |        22 |        22 |            22 |          1        |                    6 |                0.272727  |

## Context Completeness
| column                    | present   |   n_total |   n_valid |   valid_ratio |   n_missing |   missing_ratio |   n_unique |       mean |   median |        std |      iqr |      min |      max | reason   |
|:--------------------------|:----------|----------:|----------:|--------------:|------------:|----------------:|-----------:|-----------:|---------:|-----------:|---------:|---------:|---------:|:---------|
| Sleep_hours               | True      |        22 |        20 |     0.909091  |           2 |       0.0909091 |         11 |  8.145     |  8.375   |   0.987807 | 1.0625   |  6       | 10.25    | ok       |
| Day_Sleep_hours           | True      |        22 |         1 |     0.0454545 |          21 |       0.954545  |          1 |  0.25      |  0.25    | nan        | 0        |  0.25    |  0.25    | ok       |
| Sleep_qality              | True      |        22 |        20 |     0.909091  |           2 |       0.0909091 |          6 |  5.05      |  5       |   1.5035   | 2.25     |  3       |  8       | ok       |
| Difficulty_falling_asleep | True      |        22 |        15 |     0.681818  |           7 |       0.318182  |          3 |  0.8       |  1       |   0.861892 | 1.5      |  0       |  2       | ok       |
| Fragmented_sleep          | True      |        22 |         1 |     0.0454545 |          21 |       0.954545  |          1 |  1         |  1       | nan        | 0        |  1       |  1       | ok       |
| Undersleeped              | True      |        22 |        21 |     0.954545  |           1 |       0.0454545 |          2 |  0.0952381 |  0       |   0.300793 | 0        |  0       |  1       | ok       |
| Oversleeped               | True      |        22 |        21 |     0.954545  |           1 |       0.0454545 |          2 |  0.238095  |  0       |   0.436436 | 0        |  0       |  1       | ok       |
| Wakeup_class              | True      |        22 |        21 |     0.954545  |           1 |       0.0454545 |          3 |  0.380952  |  0       |   0.740013 | 0        |  0       |  2       | ok       |
| Morning_RHR               | True      |        22 |        22 |     1         |           0 |       0         |          7 | 49.6818    | 49.5     |   1.98534  | 3.5      | 46       | 52       | ok       |
| Stress                    | True      |        22 |        22 |     1         |           0 |       0         |          4 |  2.72727   |  2.5     |   0.935125 | 1        |  2       |  5       | ok       |
| Cognitive_load            | True      |        22 |        22 |     1         |           0 |       0         |          7 |  4.27273   |  4.5     |   1.95623  | 3.75     |  2       |  8       | ok       |
| DOMS                      | True      |        22 |        22 |     1         |           0 |       0         |          3 |  1.40909   |  1       |   0.73414  | 1        |  1       |  4       | ok       |
| Fatigue_before            | True      |        22 |        22 |     1         |           0 |       0         |          7 |  5.04545   |  5       |   1.58797  | 1.75     |  2       |  8       | ok       |
| Muscle_fatigue_before     | True      |        22 |        22 |     1         |           0 |       0         |          8 |  4.68182   |  5       |   1.91203  | 3        |  2       |  9       | ok       |
| Cognitive_fatigue_before  | True      |        22 |        22 |     1         |           0 |       0         |          6 |  6.81818   |  7       |   1.53177  | 2.75     |  4       |  9       | ok       |
| Wellbeing_before          | True      |        22 |        22 |     1         |           0 |       0         |          7 |  4.34091   |  4       |   1.26666  | 1        |  2       |  7       | ok       |
| Motivation_before         | True      |        22 |        22 |     1         |           0 |       0         |          5 |  4.18182   |  4       |   1.13961  | 1.75     |  2       |  6       | ok       |
| Desire_to_train           | True      |        22 |        22 |     1         |           0 |       0         |          6 |  3.18182   |  3       |   1.40192  | 2        |  1       |  6       | ok       |
| subjective_strain_before  | True      |        22 |        22 |     1         |           0 |       0         |         19 |  3.95185   |  4       |   0.768472 | 0.991477 |  2.45455 |  5.45455 | ok       |
| readiness_before          | True      |        22 |        22 |     1         |           0 |       0         |         15 |  5.42045   |  5.41667 |   0.823143 | 1.04167  |  4       |  7       | ok       |

Showing first 20 of 35 rows.

## Derived Operations
| table           | operation_id                     | operation_type     | formula         | base_formula   |   n_rows |   n_dates |   n_features |   valid_n |   valid_ratio | main_invalid_reason   |
|:----------------|:---------------------------------|:-------------------|:----------------|:---------------|---------:|----------:|-------------:|----------:|--------------:|:----------------------|
| context_derived | baseline_dev_context_before      | baseline_deviation | delta           | nan            |      462 |        22 |           21 |       372 |      0.805195 | insufficient_baseline |
| context_derived | baseline_dev_context_before      | baseline_deviation | robust_z        | nan            |      462 |        22 |           21 |       361 |      0.781385 | insufficient_baseline |
| context_derived | baseline_dev_context_flags       | baseline_deviation | delta           | nan            |      110 |        22 |            5 |        65 |      0.590909 | insufficient_baseline |
| context_derived | day_to_day_context_change        | lagged_pairwise    | delta           | nan            |      462 |        22 |           21 |       405 |      0.876623 | missing_value         |
| context_derived | physio_before_after_response     | pairwise_columns   | delta           | nan            |      132 |        22 |            6 |       127 |      0.962121 | missing_value         |
| context_derived | physio_before_after_response     | pairwise_columns   | percent_delta   | nan            |      132 |        22 |            6 |       127 |      0.962121 | missing_value         |
| context_derived | subjective_before_after_response | pairwise_columns   | delta           | nan            |      308 |        22 |           14 |       275 |      0.892857 | missing_value         |
| context_derived | training_load_previous_7d        | rolling_aggregate  | rolling_7d_mean | nan            |      198 |        22 |            9 |       184 |      0.929293 | insufficient_window   |
| context_derived | training_load_previous_7d        | rolling_aggregate  | rolling_7d_sum  | nan            |      198 |        22 |            9 |       184 |      0.929293 | insufficient_window   |
| ecg_derived     | baseline_dev_sit_before          | baseline_deviation | delta           | nan            |      176 |        22 |            8 |       152 |      0.863636 | insufficient_baseline |
| ecg_derived     | baseline_dev_sit_before          | baseline_deviation | robust_z        | nan            |      176 |        22 |            8 |       152 |      0.863636 | insufficient_baseline |
| ecg_derived     | breath_in_response_before        | pairwise           | delta           | nan            |      176 |        22 |            8 |       110 |      0.625    | missing_value         |
| ecg_derived     | breath_in_response_before        | pairwise           | percent_delta   | nan            |      176 |        22 |            8 |       101 |      0.573864 | missing_value         |
| ecg_derived     | orthostatic_after                | pairwise           | delta           | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_after                | pairwise           | percent_delta   | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_after                | pairwise           | ratio           | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_before               | pairwise           | delta           | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_before               | pairwise           | percent_delta   | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_before               | pairwise           | ratio           | nan            |      176 |        22 |            8 |       176 |      1        |                       |
| ecg_derived     | orthostatic_training_response    | pairwise_derived   | delta           | delta          |      176 |        22 |            8 |       176 |      1        |                       |

Showing first 20 of 26 rows.

## Targets
| target                  | column                          | label                             | direction     | horizon      | data_type   | scale_type   | present   |   n_total |   n_valid |   valid_ratio |   n_missing |   missing_ratio |   n_unique |     mean |   median |      std |     iqr |       min |    max | usable   | reason   |
|:------------------------|:--------------------------------|:----------------------------------|:--------------|:-------------|:------------|:-------------|:----------|----------:|----------:|--------------:|------------:|----------------:|-----------:|---------:|---------:|---------:|--------:|----------:|-------:|:---------|:---------|
| subjective_strain_delta | target__subjective_strain_delta | Acute subjective strain response  | higher_worse  | acute        | numeric     | ordinal      | True      |        22 |        21 |      0.954545 |           1 |       0.0454545 |         21 | 0.442893 | 0.252525 | 1.04096  | 1.27273 |  -1.78788 |  2.625 | True     | ok       |
| fatigue_delta           | target__fatigue_delta           | Acute subjective fatigue response | higher_worse  | acute        | numeric     | ordinal      | True      |        22 |        21 |      0.954545 |           1 |       0.0454545 |          8 | 0.952381 | 1        | 1.88351  | 2       |  -3       |  5     | True     | ok       |
| reaction_time_delta     | target__reaction_time_delta     | Acute reaction time response      | higher_worse  | acute        | numeric     | continuous   | True      |        22 |        21 |      0.954545 |           1 |       0.0454545 |         14 | 1.85714  | 1        | 7.83126  | 9       | -12       | 17     | True     | ok       |
| readiness_before        | target__readiness_before        | Pre-training readiness            | higher_better | pre_training | numeric     | ordinal      | True      |        22 |        22 |      1        |           0 |       0         |         15 | 5.42045  | 5.41667  | 0.823143 | 1.04167 |   4       |  7     | True     | ok       |

## Feature Availability
| feature                                                | role   | source_table     | feature_family   |   n_total |   n_valid |   valid_ratio |   n_missing |   missing_ratio |   n_unique |          mean |        median |          std |          iqr |           min |           max | usable   | reason            |
|:-------------------------------------------------------|:-------|:-----------------|:-----------------|----------:|----------:|--------------:|------------:|----------------:|-----------:|--------------:|--------------:|-------------:|-------------:|--------------:|--------------:|:---------|:------------------|
| raw_ecg__before__sit__full__hrv_time__MeanHR_bpm       | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  61.4869      |  61.1367      |  3.34616     |  3.83752     |  54.1898      |   68.0797     | True     | ok                |
| raw_ecg__before__sit__full__hrv_time__MeanNN_ms        | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 | 980           | 982.718       | 53.0629      | 60.7428      | 882.465       | 1108.24       | True     | ok                |
| raw_ecg__before__sit__full__hrv_time__SDNN_ms          | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  36.2913      |  34.7199      |  9.69779     |  9.94781     |  22.6693      |   70.1788     | True     | ok                |
| raw_ecg__before__sit__full__hrv_time__RMSSD_ms         | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  33.6675      |  32.5423      |  7.59847     |  5.39897     |  23.8454      |   60.2172     | True     | ok                |
| raw_ecg__before__sit__full__hrv_time__pNN50_percent    | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         21 |  13.7283      |  10.8534      |  9.89607     |  6.24093     |   1.69492     |   49.1228     | True     | ok                |
| raw_ecg__before__sit__full__hrv_freq__LF_power         | input  | raw_ecg_features | hrv_freq         |        22 |        22 |      1        |           0 |        0        |         22 |   0.000368849 |   0.000263468 |  0.000282461 |  0.00029244  |   7.20959e-05 |    0.00110303 | True     | ok                |
| raw_ecg__before__sit__full__hrv_freq__HF_power         | input  | raw_ecg_features | hrv_freq         |        22 |        22 |      1        |           0 |        0        |         22 |   0.000238233 |   0.00019725  |  0.000225955 |  9.04578e-05 |   7.45497e-05 |    0.00119777 | True     | ok                |
| raw_ecg__before__sit__full__hrv_freq__LF_HF_ratio      | input  | raw_ecg_features | hrv_freq         |        22 |        22 |      1        |           0 |        0        |         22 |   2.06387     |   1.63436     |  1.91546     |  1.44874     |   0.302937    |    8.40394    | True     | ok                |
| raw_ecg__before__sit__full__morph_qrs__QRS_duration_ms | input  | raw_ecg_features | morph_qrs        |        22 |        15 |      0.681818 |           7 |        0.318182 |          9 | 102.936       | 102.367       | 23.0747      | 42.653       |  76.7754      |  136.49       | True     | ok                |
| raw_ecg__before__sit__full__morph_qrs__QRS_area        | input  | raw_ecg_features | morph_qrs        |        22 |        15 |      0.681818 |           7 |        0.318182 |         15 |   9.44353     |   9.01005     |  1.42092     |  1.45365     |   7.79666     |   13.5509     | True     | ok                |
| raw_ecg__before__sit__full__morph_qrs__QRS_main_amp    | input  | raw_ecg_features | morph_qrs        |        22 |        15 |      0.681818 |           7 |        0.318182 |         15 | 350.461       | 351.607       |  5.72407     |  8.11223     | 338.124       |  359.353      | True     | ok                |
| raw_ecg__before__sit__full__morph_qrs__R_width_half_ms | input  | raw_ecg_features | morph_qrs        |        22 |        15 |      0.681818 |           7 |        0.318182 |          3 |  32.4163      |  34.1224      |  2.69761     |  4.2653      |  29.8571      |   38.3877     | False    | low_unique_values |
| raw_ecg__before__sit__full__morph_t__T_amp             | input  | raw_ecg_features | morph_t          |        22 |        15 |      0.681818 |           7 |        0.318182 |         15 |  48.5717      |  51.2011      | 24.9883      | 16.4053      | -29.5172      |   81.1786     | True     | ok                |
| raw_ecg__before__sit__full__morph_t__T_area            | input  | raw_ecg_features | morph_t          |        22 |        15 |      0.681818 |           7 |        0.318182 |         15 |   2.59658     |   2.59592     |  1.24484     |  1.07815     |  -0.767488    |    4.54273    | True     | ok                |
| raw_ecg__before__sit__full__morph_t__RT_interval_ms    | input  | raw_ecg_features | morph_t          |        22 |        15 |      0.681818 |           7 |        0.318182 |          8 | 248.809       | 255.918       | 35.271       |  8.5306      | 123.694       |  272.979      | True     | ok                |
| raw_ecg__before__sit__full__morph_t__QT_like_ms        | input  | raw_ecg_features | morph_t          |        22 |        15 |      0.681818 |           7 |        0.318182 |          8 | 330.419       | 332.694       | 20.3197      | 10.6633      | 264.449       |  358.285      | True     | ok                |
| raw_ecg__before__stand__full__hrv_time__MeanHR_bpm     | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  79.2604      |  78.048       |  6.95611     |  9.1509      |  67.6255      |   92.5982     | True     | ok                |
| raw_ecg__before__stand__full__hrv_time__MeanNN_ms      | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 | 768.523       | 780.426       | 63.8977      | 90.1805      | 653.918       |  888.364      | True     | ok                |
| raw_ecg__before__stand__full__hrv_time__SDNN_ms        | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  61.7496      |  58.1458      | 23.3932      | 41.6313      |  31.7356      |  106.982      | True     | ok                |
| raw_ecg__before__stand__full__hrv_time__RMSSD_ms       | input  | raw_ecg_features | hrv_time         |        22 |        22 |      1        |           0 |        0        |         22 |  25.0925      |  23.0918      | 10.4314      |  4.64764     |  11.5752      |   63.3822     | True     | ok                |

Showing first 20 of 510 rows.

## Feature Target Overlap
| target                  | target_column                   | feature                                                |   n_target_valid |   n_feature_valid |   n_overlap |   overlap_ratio | usable   | reason   |
|:------------------------|:--------------------------------|:-------------------------------------------------------|-----------------:|------------------:|------------:|----------------:|:---------|:---------|
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_time__MeanHR_bpm       |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_time__MeanNN_ms        |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_time__SDNN_ms          |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_time__RMSSD_ms         |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_time__pNN50_percent    |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_freq__LF_power         |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_freq__HF_power         |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__hrv_freq__LF_HF_ratio      |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_qrs__QRS_duration_ms |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_qrs__QRS_area        |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_qrs__QRS_main_amp    |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_qrs__R_width_half_ms |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_t__T_amp             |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_t__T_area            |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_t__RT_interval_ms    |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__sit__full__morph_t__QT_like_ms        |               21 |                15 |          15 |        0.681818 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__stand__full__hrv_time__MeanHR_bpm     |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__stand__full__hrv_time__MeanNN_ms      |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__stand__full__hrv_time__SDNN_ms        |               21 |                22 |          21 |        0.954545 | True     | ok       |
| subjective_strain_delta | target__subjective_strain_delta | raw_ecg__before__stand__full__hrv_time__RMSSD_ms       |               21 |                22 |          21 |        0.954545 | True     | ok       |

Showing first 20 of 2040 rows.

## Ml Readiness
| target                  |   n_samples |   n_usable_features_by_overlap |   feature_sample_ratio | recommended_use   | warning                       |
|:------------------------|------------:|-------------------------------:|-----------------------:|:------------------|:------------------------------|
| subjective_strain_delta |          21 |                            397 |                18.9048 | diagnostic_only   | too_many_features_for_samples |
| fatigue_delta           |          21 |                            397 |                18.9048 | diagnostic_only   | too_many_features_for_samples |
| reaction_time_delta     |          21 |                            397 |                18.9048 | diagnostic_only   | too_many_features_for_samples |
| readiness_before        |          22 |                            398 |                18.0909 | diagnostic_only   | too_many_features_for_samples |