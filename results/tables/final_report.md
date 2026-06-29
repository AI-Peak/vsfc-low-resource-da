# Final Experiment Report

## Executive Summary

- Phase 3 PhoBERT no-augmentation baseline was near the original gate: test macro-F1 0.8478 against the 0.85 target.
- Conservative EDA had a mixed effect: it helped 5% and 20%, but hurt 10%.
- The 5% raw LLM-style paraphrase pilot was the strongest augmentation result with test macro-F1 0.7864.
- Phase 6 filtering removed only 7 of 571 paraphrases and reduced the 5% LLM score to 0.7752, so filtering improved QC but did not beat raw LLM.

## Final Results

| phase | ratio | method | macro_f1 | delta_vs_none | notes |
|---|---|---|---|---|---|
| Phase 4 | 0.05 | eda | 0.7641 | 0.0078 | improved |
| Phase 6 | 0.05 | llm_paraphrase_filtered | 0.7752 | 0.0189 | Filtered LLM improves over none and EDA but underperforms the raw LLM pilot |
| Phase 5 | 0.05 | llm_paraphrase_raw | 0.7864 | 0.0301 | paraphrase_local |
| Phase 4 | 0.05 | none | 0.7563 | 0.0000 | low-resource baseline |
| Phase 4 | 0.10 | eda | 0.7956 | -0.0275 | worse |
| Phase 4 | 0.10 | none | 0.8231 | 0.0000 | low-resource baseline |
| Phase 4 | 0.20 | eda | 0.8208 | 0.0059 | improved |
| Phase 4 | 0.20 | none | 0.8149 | 0.0000 | low-resource baseline |
| Phase 3 | 1.00 | none | 0.8478 | 0.0000 | near-gate 100% PhoBERT baseline; gate was 0.85 |

## Best Method Per Ratio

| phase | ratio | method | macro_f1 | delta_vs_none | notes |
|---|---|---|---|---|---|
| Phase 5 | 0.05 | llm_paraphrase_raw | 0.7864 | 0.0301 | paraphrase_local |
| Phase 4 | 0.10 | none | 0.8231 | 0.0000 | low-resource baseline |
| Phase 4 | 0.20 | eda | 0.8208 | 0.0059 | improved |
| Phase 3 | 1.00 | none | 0.8478 | 0.0000 | near-gate 100% PhoBERT baseline; gate was 0.85 |

## Research Question Notes

- RQ1: augmentation can help in low-resource settings, but the effect is method and ratio dependent.
- RQ2: the clearest gain is at 5%, where raw LLM improves over none by 0.0301 macro-F1.
- RQ3: in the available 5% pilot, raw LLM outperforms EDA by 0.0223 macro-F1.
- RQ4: the heuristic filter flagged a low drift-risk rate, dropping 1.23% of raw paraphrases; filtered LLM underperformed raw LLM in macro-F1.
- RQ5: low-resource augmentation does not close the gap to the 100% PhoBERT ceiling in the current single-seed experiments.

## Artifacts

- Macro-F1 figure: `results/tables/figures/phase9_macro_f1_by_ratio.png`
- Drift analysis: `results/tables/drift_analysis.md`

## Limitations

- Gemini quota prevented full API-based LLM generation for all ratios.
- Current LLM results are a 5% pilot using the local paraphrase fallback.
- Statistical testing needs prediction CSVs in the active runtime; if Kaggle reset removed them, rerun significance tests only after regenerating or restoring predictions.
