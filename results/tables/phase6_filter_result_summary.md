# Phase 6 Filtered LLM Result Summary

| Ratio | None Macro-F1 | EDA Macro-F1 | LLM Raw Macro-F1 | LLM Filtered Macro-F1 | Delta vs None | Delta vs EDA | Delta vs Raw |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.7563 | 0.7641 | 0.7864 | 0.7752 | +0.0189 | +0.0111 | -0.0112 |

Filtering kept 564 of 571 raw paraphrases (keep rate 0.9877). The filtered
LLM file still improves over the 5% no-augmentation and EDA baselines, but the
raw LLM pilot remains the strongest 5% augmentation result.
