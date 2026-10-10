# Result tables

Numbers behind the figures of the paper (percent; MolCoT-Paired-4K test split, 422 records).

| File | Content |
|---|---|
| `model_families_epoch4.csv` | Five model families, epoch 4: exact match (EM), validity and similarity at pass@1 and pass@16 (mean and sd over 10 seeds) for Original-CoT and Curated-CoT SFT, and the Curated − Original gap with its 95% interval and p-value. |
| `model_scales_epoch4.csv` | Six Qwen3.5 sizes, epoch 4, same columns (3 seeds). |
| `grpo_stages.csv` | Qwen3.5-4B at the start of GRPO (after SFT, or the base model) and after GRPO epochs 1 and 2 (3 seeds). |
| `all_figure_numbers.csv` | Every point of every result figure: experiment (A = model families, B = model scales, D = GRPO), series (model), condition, stage (SFT epoch 0-4, GRPO epoch), metric, N of pass@N, estimator, mean, sd, number of seeds. |
| `dataset_label_distribution.csv`, `dataset_split_labels.csv` | Expert labels (1-5) of the Original CoTs, overall and per split. |
| `external_testsets.csv` | Qwen3.5-4B models on ChEBI-20 test and SMolInstruct molecule-generation test, restricted to molecules unseen in the MolCoT-Paired-4K training split; strict greedy EM per seed. |

pass@1 is the mean over the 16 sampled answers; pass@N for N > 1 is the unbiased estimate from the 16 samples
(Chen et al., 2021). `eval/score_predictions.py` computes all of these from the prediction files of a run.
