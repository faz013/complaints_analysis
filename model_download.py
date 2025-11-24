import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    dtype=torch.bfloat16
)

input_text = "She walked to the [MASK]."
results = pipe(input_text)
pprint(results)