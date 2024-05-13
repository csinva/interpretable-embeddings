python gen_questions.py \
--input_file="questions_query_2k.json" \
--prompt_file="query_text_prompt.txt" \
--output_file="questions_corpus_2k.json" \
--model="gpt-4-1106-preview" \
--sort_key="question_id" \
--batch_size=20 \
--mode="local" \
# --tq=10