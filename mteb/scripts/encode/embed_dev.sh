#!/bin/bash

python encode.py \
--input_file="../../questions/msmarco/sample_dev_queries_100_beir.json" \
--questions_file="../../questions/msmarco/questions_query.json" \
--prompt_file="prompts/QAembedder_system_prompt_queries.txt" \
--key="query_id" \
--output_file="dev_queries_out.json" \
--tq 50

python encode.py \
--input_file="../../questions/msmarco/sample_dev_corpus_120_beir.json" \
--questions_file="../../questions/msmarco/questions_corpus.json" \
--prompt_file="prompts/QAembedder_system_prompt_corpus.txt" \
--key="corpus_id" \
--output_file="dev_corpus_out.json" \
--tq 60
