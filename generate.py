#!/usr/bin/env python3

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="True"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./",
    local_files_only=True  # Ensures it only looks in local folder
)

conversation = [
    dict(role='user', content='Hello.'),
    dict(role='assistant', content='<think>\\nThe user is saying hello. I should reply by saying', prefix=True),
]

print(tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False))
