import numpy as np
data = np.fromfile(
    "/mnt/mnt/zjdx/paloma_c4_100_domain.bin",
    dtype=np.uint16
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:10000]))