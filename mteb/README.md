# instructor

Ran https://github.com/embeddings-benchmark/mteb/blob/main/scripts/merge_cqadupstack.py as well. And used the merged file for final metadata computation. 

This file has the metrics that they used. (at the end of the file)
https://github.com/xlang-ai/instructor-embedding/blob/main/evaluation/MTEB/mteb/evaluation/MTEB.py

## Billboard:

python main.py --cache ./cache --task cnndm --model_name hkunlp/instructor-xl \
cnndm: (0.31735039505172724, 2.119105535852967e-43) \
mt: (0.3054921424524214, 0.0) \
mscoco: (0.3955485262147562, 2.0504146777169997e-94) \
Average: 33.9 

hkunlp/instructor-large \
cnndm: (0.3033043743717928, 1.2948232470051671e-39) \
mt: (0.3884822996867111, 0.0) \
mscoco: (0.41572307048702417, 4.593896974426431e-105) \
Average: 36.9


| Model         | **AVG**   | Retrieval | Reranking | Clustering | PairClassification | Classification | STS   | Summarization |
| ------------- | --------- | --------- | --------- | ---------- | ------------------ | -------------- | ----- | ------------- |
| SimCSE        | **51.09** | 22.3      | 47.96     | 34.1       | 76.06              | 67.85          | 78.21 | 31.17         |
| coCondenser   | **45.49** | 14.9      | 46.31     | 33.38      | 68.13              | 63.38          | 62.73 | 29.6          |
| Contriever    | **56.03** | 41.9      | 53.1      | 40.9       | 82.5               | 66.9           | 76.5  | 30.4          |
| Glove.3B.600d | **44.51** | 21.6      | 43.3      | 27.7       | 70.9               | 57.4           | 61.9  | 28.8          |

