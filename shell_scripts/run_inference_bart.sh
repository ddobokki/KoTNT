#!/bin/bash

python3 ../bart_inference.py \
	--model_name_or_path="/data2/bart/temp_workspace/nlp/output_dir/checkpoint-30000" \
	--beam_width="100" \
	--min_length="0" \
	--max_length="1206" \
	--do_sample="false" \
	--num_beam_groups="1"