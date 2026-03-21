# RoBERTa vs Others: Paired Statistical Test (Sequence-level F1)

| Partition | Comparison | N | Mean RoBERTa | Mean Other | Δ | p (Wilcoxon) | Sig |
|-----------|------------|---|--------------|------------|---|---------------|-----|
| TS0 (TEST) | RoBERTa vs ERNIE | 1305 | 0.2046 | 0.5090 | -0.3044 | 1.00e+00 |  |
| TS0 (TEST) | RoBERTa vs RNAFM | 1305 | 0.2046 | 0.1594 | 0.0452 | 3.70e-49 | *** |
| TS0 (TEST) | RoBERTa vs RiNALMo | 1305 | 0.2046 | 0.0623 | 0.1423 | 9.29e-191 | *** |
| TS0 (TEST) | RoBERTa vs One-hot | 1305 | 0.2046 | 0.0385 | 0.1661 | 6.69e-199 | *** |
| TS0 (TEST) | RoBERTa vs RNABERT | 1281 | 0.2050 | 0.0335 | 0.1715 | 6.09e-203 | *** |
| NEW | RoBERTa vs ERNIE | 5401 | 0.2037 | 0.4929 | -0.2893 | 1.00e+00 |  |
| NEW | RoBERTa vs RNAFM | 5401 | 0.2037 | 0.1093 | 0.0943 | 0.00e+00 | *** |
| NEW | RoBERTa vs RiNALMo | 5401 | 0.2037 | 0.0591 | 0.1445 | 0.00e+00 | *** |
| NEW | RoBERTa vs One-hot | 5401 | 0.2037 | 0.0520 | 0.1517 | 0.00e+00 | *** |
| NEW | RoBERTa vs RNABERT | 5380 | 0.2040 | 0.0278 | 0.1762 | 0.00e+00 | *** |
