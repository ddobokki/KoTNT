#!/bin/bash

python3 ../bart_inference.py \
	--model_name_or_path="" \
	--beam_width="100" \
	--min_length="0" \
	--max_length="1206" \
	--do_sample="false" \
	--num_beam_groups="1" \
	--direction="backward"