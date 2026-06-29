# Phase 5 LLM Paraphrase Summary

| Ratio | None Macro-F1 | EDA Macro-F1 | LLM Raw Macro-F1 | Delta vs None | Delta vs EDA | Method |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.7563 | 0.7641 | 0.7864 | +0.0301 | +0.0223 | paraphrase_local |

Conclusion: because Gemini free-tier quota blocked full API generation, Phase 5
was run as a 5% low-resource pilot using the local paraphrase fallback. The
raw LLM-compatible paraphrases improved test macro-F1 over both no augmentation
and conservative EDA at the same ratio.
