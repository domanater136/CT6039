# Predictive Validation Report

## Setup

- Cutoff: **2025-07-01**
- Training maps: **11,037**
- Test maps: **5,658**
- Players in training: **673**

## Metrics

| model                                   |    mae |   rmse |   win_loss_accuracy |
|:----------------------------------------|-------:|-------:|--------------------:|
| Zero Baseline                           | 5.4286 | 6.1083 |              0.4532 |
| Model 1: Individual RAPM                | 5.123  | 6.0417 |              0.571  |
| Model 2: Joint RAPM + Pair Interactions | 5.1537 | 5.9733 |              0.5744 |

## McNemar Test

- M1 correct / M2 wrong: **287**
- M2 correct / M1 wrong: **306**
- Mid-p: **0.4357**
- Significant: **False**

## Bootstrap CIs

| Metric | Improvement | 95% CI | Excludes zero |
|---|---|---|---|
| MAE | -0.0307 | [-0.0562, -0.0043] | True |
| RMSE | 0.0684 | [0.0433, 0.0928] | True |
| ACCURACY | 0.0035 | [-0.005, 0.0111] | False |

## Top Synergistic Pairs

| pair_col       |   pair_coef |   player_a_id |   player_b_id | player_a_label   | player_b_label   |
|:---------------|------------:|--------------:|--------------:|:-----------------|:-----------------|
| pr_15369_21139 |      1.7328 |         15369 |         21139 | redstar          | pr1metapz        |
| pr_3741_19230  |      1.4484 |          3741 |         19230 | niko             | m0nesy           |
| pr_20254_20702 |      1.3011 |         20254 |         20702 | starry           | jee              |
| pr_21763_22011 |      1.1476 |         21763 |         22011 | ultimate         | kadziu           |
| pr_22084_22125 |      0.9497 |         22084 |         22125 | mello            | lich             |
| pr_20219_22085 |      0.9414 |         20219 |         22085 | xns              | spinnie          |
| pr_20519_21638 |      0.9317 |         20519 |         21638 | lcm              | leleo            |
| pr_9031_16705  |      0.8841 |          9031 |         16705 | valde            | altekz           |
| pr_16816_16835 |      0.8773 |         16816 |         16835 | vsm              | ponter           |
| pr_11816_11893 |      0.8604 |         11816 |         11893 | ropz             | zywoo            |
| pr_11816_16693 |      0.8604 |         11816 |         16693 | ropz             | flamez           |
| pr_7322_11816  |      0.8604 |          7322 |         11816 | apex             | ropz             |
| pr_11816_18462 |      0.8604 |         11816 |         18462 | ropz             | mezii            |
| pr_8248_13670  |      0.8241 |          8248 |         13670 | jkaem            | nawwk            |
| pr_20113_20709 |      0.811  |         20113 |         20709 | deko             | r3salt           |
| pr_12521_19692 |      0.8095 |         12521 |         19692 | art              | zevy             |
| pr_7592_16717  |      0.8033 |          7592 |         16717 | device           | br0              |
| pr_15117_18120 |      0.7792 |         15117 |         18120 | keoz             | sinnopsyy        |
| pr_1206_18294  |      0.7402 |          1206 |         18294 | btn              | adron            |
| pr_18891_22163 |      0.7331 |         18891 |         22163 | cutzmeretz       | misfit           |
| pr_19692_20385 |      0.7098 |         19692 |         20385 | zevy             | kye              |
| pr_22040_22930 |      0.7073 |         22040 |         22930 | z1nny            | tex1y            |
| pr_14619_17372 |      0.7054 |         14619 |         17372 | infinite         | fang             |
| pr_1206_20761  |      0.7028 |          1206 |         20761 | btn              | launx            |
| pr_18141_20085 |      0.6958 |         18141 |         20085 | biguzera         | kauez            |

## Top Anti-Synergistic Pairs

| pair_col       |   pair_coef |   player_a_id |   player_b_id | player_a_label   | player_b_label   |
|:---------------|------------:|--------------:|--------------:|:-----------------|:-----------------|
| pr_20577_22084 |     -0.7603 |         20577 |         22084 | bruninho         | mello            |
| pr_15369_17145 |     -0.6713 |         15369 |         17145 | redstar          | rainwaker        |
| pr_19733_21816 |     -0.5992 |         19733 |         21816 | demqq            | gizmy            |
| pr_17132_18891 |     -0.5698 |         17132 |         18891 | leomonster       | cutzmeretz       |
| pr_19231_22076 |     -0.5602 |         19231 |         22076 | h1te             | aw               |
| pr_19231_21692 |     -0.5602 |         19231 |         21692 | h1te             | kalash           |
| pr_19231_20600 |     -0.5602 |         19231 |         20600 | h1te             | sm3t             |
| pr_19231_22075 |     -0.5602 |         19231 |         22075 | h1te             | sfade8           |
| pr_15949_23397 |     -0.5466 |         15949 |         23397 | peppzor          | poiii            |
| pr_10330_22695 |     -0.5248 |         10330 |         22695 | acor             | sirah            |
| pr_10264_22695 |     -0.5248 |         10264 |         22695 | niko             | sirah            |
| pr_14737_21155 |     -0.5145 |         14737 |         21155 | meyern           | naz              |
| pr_9618_13670  |     -0.5133 |          9618 |         13670 | nexa             | nawwk            |
| pr_9219_19869  |     -0.459  |          9219 |         19869 | felps            | try              |
| pr_16848_20794 |     -0.4548 |         16848 |         20794 | hades            | leen             |
| pr_1206_21213  |     -0.4518 |          1206 |         21213 | btn              | ersin            |
| pr_2476_20813  |     -0.4401 |          2476 |         20813 | emi              | vldn             |
| pr_11110_23397 |     -0.4244 |         11110 |         23397 | golden           | poiii            |
| pr_8327_20637  |     -0.4159 |          8327 |         20637 | furlan           | b1elany          |
| pr_22126_22181 |     -0.3884 |         22126 |         22181 | nyezin           | urban0           |
| pr_18875_22126 |     -0.3884 |         18875 |         22126 | ricioli          | nyezin           |
| pr_16954_22126 |     -0.3884 |         16954 |         22126 | yepz             | nyezin           |
| pr_13670_18571 |     -0.3839 |         13670 |         18571 | nawwk            | cypher           |
| pr_16835_17181 |     -0.3821 |         16835 |         17181 | ponter           | naitte           |
| pr_14394_16835 |     -0.3821 |         14394 |         16835 | tuurtle          | ponter           |
