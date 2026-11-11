# PlanRec-Agent

> Code-only source snapshot. Models, datasets, run logs, and credentials are excluded.

PlanRec-Agent is a generative recommendation research project with reproducible workflows for semantic planning, candidate generation, and budget-aware ranking evaluation. It emphasizes preregistered gates, full-catalog evaluation, and honest reporting of negative results.

## Contents

- Recommendation models and semantic-ID generation components (`src/planrec/`)
- Training, full-catalog evaluation, and audit scripts (`scripts/`)
- Configurations, unit tests, and traceable report utilities

## Scope & Limitations

This selected source snapshot is intended for code inspection rather than turnkey end-to-end execution. Some retained tests reference omitted private dataset/artifact modules (`planrec.data.amazon`, `preprocess_amazon_2014`, `scripts/fetch_cloud_artifact.py`). Interpret each experiment strictly under its frozen protocol; passing engineering tests does not imply a recommendation-quality gain.

## Language & License

- 中文说明：[README.zh-CN.md](README.zh-CN.md)
- Released for reading and reference only; no open-source redistribution license is granted.
