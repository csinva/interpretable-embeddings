"""
Usage: python mteb_meta.py path_to_results_folder

Creates evaluation results metadata for the model card. 
E.g.
---
tags:
- mteb
model-index:
- name: SGPT-5.8B-weightedmean-msmarco-specb-bitfit
  results:
  - task:
      type: classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77
      config: default
      split: test
      revision: 44fa15921b4c889113cc5df03dd4901b49161ab7
    metrics:
    - type: accuracy
      value: 84.49350649350649
---
"""

import json
import logging
import os
import sys

from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


results_folder = sys.argv[1].rstrip("/")
model_name = results_folder.split("/")[-1]

all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        logger.info(f"Skipping non-json {file_name}")
        continue
    with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
        results = json.load(f)
        all_results = {**all_results, **{file_name.replace(".json", ""): results}}

# Use "train" split instead
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
# Use "validation" split instead
VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MultilingualSentiment", "Ocnli"]
# Use "dev" split instead
DEV_SPLIT = ["CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "T2Reranking", "T2Retrieval", "VideoRetrieval"]

MARKER = "---"
TAGS = "tags:"
MTEB_TAG = "- mteb"
HEADER = "model-index:"
MODEL = f"- name: {model_name}"
RES = "  results:"

META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])


ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n      revision: {}\n    metrics:"
ONE_METRIC = "    - type: {}\n      value: {}"

metrics_list = ["ndcg_at_10", "map", "v_measure", "cos_sim_ap", "accuracy", "cos_sim_spearman"]
comments = ["Retrieval", "Reranking", "Clustering", "PairClassification", "Classification", "STS"]
summarization_cos_sim_spearman = None

metrics_to_comments = dict(zip(metrics_list, comments))
average_scores = {f"average_{metric}": [] for metric in metrics_list}

for ds_name, res_dict in sorted(all_results.items()):
    mteb_desc = (
        MTEB(tasks=[ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")]).tasks[0].description
    )
    hf_hub_name = mteb_desc.get("hf_hub_name", mteb_desc.get("beir_name"))
    if "CQADupstack" in ds_name:
        hf_hub_name = "BeIR/cqadupstack"
    mteb_type = mteb_desc["type"]
    revision = res_dict.get("dataset_revision")  # Okay if it's None
    split = "test"
    if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
        split = "train"
    elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
        split = "validation"
    elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
        split = "dev"
    elif "test" not in res_dict:
        logger.info(f"Skipping {ds_name} as split {split} not present.")
        continue
    res_dict = res_dict.get(split)
    for lang in mteb_desc["eval_langs"]:
        mteb_name = f"MTEB {ds_name}"
        mteb_name += f" ({lang})" if len(mteb_desc["eval_langs"]) > 1 else ""
        # For English there is no language key if it's the only language
        test_result_lang = res_dict.get(lang) if len(mteb_desc["eval_langs"]) > 1 else res_dict
        # Skip if the language was not found but it has other languages
        if test_result_lang is None:
            continue
        META_STRING += "\n" + ONE_TASK.format(
            mteb_type, hf_hub_name, mteb_name, lang if len(mteb_desc["eval_langs"]) > 1 else "default", split, revision
        )
        
        # Initialize a dictionary to store the metrics for the current dataset and language
        current_metrics = {}
        for metric, score in test_result_lang.items():
            if not isinstance(score, dict):
                score = {metric: score}
            for sub_metric, sub_score in score.items():
                # Store all metrics, not just the ones in the metrics_list
                formatted_metric_name = f"{metric}_{sub_metric}" if metric != sub_metric else metric
                current_metrics[formatted_metric_name] = sub_score * 100  # Multiply by 100 to convert to percentage
                # Add the score to the average_scores list for later calculation if it's in the metrics_list
                if formatted_metric_name == "cos_sim_spearman" and mteb_type == "Summarization":
                    summarization_cos_sim_spearman = sub_score
                elif metric in metrics_list or (metric == "cos_sim" and f"{metric}_{sub_metric}" in metrics_list):
                    average_key = f"average_{metric}" if metric in metrics_list else f"average_{metric}_{sub_metric}"
                    average_scores[average_key].append(sub_score)
        
        # Add the metrics for the current dataset and language to the META_STRING
        for metric_name, metric_value in current_metrics.items():
            META_STRING += "\n" + ONE_METRIC.format(metric_name, metric_value)

# Append the average scores at the end of the file
for average_key, scores in average_scores.items():
    if scores:  # Check if there are scores to avoid division by zero
        average_score = sum(scores) / len(scores)
        # Include a comment for each metric
        metric_name = average_key.replace("average_", "")
        comment = metrics_to_comments.get(metric_name, "")
        if comment:
            META_STRING += f"\n# {comment}"
        META_STRING += "\n" + ONE_METRIC.format(
            average_key,
            average_score * 100  # Multiply by 100 to convert to percentage
        )

# Append the summarization cos_sim_spearman score at the end of the file separately
if summarization_cos_sim_spearman is not None:
    META_STRING += f"\n# Summarization"
    META_STRING += "\n" + ONE_METRIC.format(
        "cos_sim_spearman",
        summarization_cos_sim_spearman * 100  # Multiply by 100 to convert to percentage
    )
META_STRING += "\n" + MARKER
metadata_filename = f"./{model_name}.md"
if os.path.exists(metadata_filename):
    logger.warning(f"Overwriting {metadata_filename}")
with open(metadata_filename, "w") as f:
    f.write(META_STRING)