# Audit Notes

1. The manuscript risk equation was corrected after audit to the implementation actually used:
   G = 0.30*A(S_R) + 0.25*C + 0.30*T + 0.15*U.

2. The controller thresholds actually used in the archived runs are 0.30/0.50/0.70 with hysteresis 0.05 and a T>=0.90 Level-4 override.

3. Replaying the controller on 30,500 archived full-SRACR sequence decisions with these settings reproduced the stored controller states exactly; see `results/reviewer_verification/R1C6_controller_replay_audit.csv`.

4. The v7.3 archived saliency caches are preserved exactly for numerical audit. The v7.3.2 code patch adds explicit RNG seeding before stochastic saliency-feature recomputation. Therefore, exact manuscript-number audit should use the archived caches; fresh deterministic recomputation should be reported as a rebuild, not silently substituted.

5. PMC is an internal policy-concentration metric, not an independent security endpoint. The independent attack-oriented endpoint is supplied under `results/independent_leakage/`.

6. Patient-level independence is not claimed because consistent authoritative patient/study identifiers were unavailable across all source artifacts. The exact/verified-near-duplicate grouping is included in `data_index/exact_split_manifest.csv`.
