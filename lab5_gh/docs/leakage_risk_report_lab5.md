# Leakage Risk Report – Lab 5

## Split Strategy

Stratified random split was used to preserve class distribution between train, validation and test sets.



Seed: 42



Split sizes:
Train: 6288
Val: 786
Test: 786



TRAIN
label
Real    3668
Fake    2620

label
Real    0.583333
Fake    0.416667



VAL
label
Real    459
Fake    327


label
Real    0.583969
Fake    0.416031


TEST
label
Real    458
Fake    328


label
Real    0.582697
Fake    0.417303


## Leakage Checks

Exact duplicates

train ∩ test: 0
train ∩ val: 0
val ∩ test: 0



Near duplicates (cosine > 0.95)

pairs found: 19



Template leakage

suspicious patterns found: 1



## Remaining Risks

* possible paraphrased duplicates
* news topics may overlap
* class imbalance could affect training



## Next Steps

* train baseline ML model
* evaluate performance on test set
