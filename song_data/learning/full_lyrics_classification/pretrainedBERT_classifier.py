import transformers
import numpy as np
from dataset import get_dataset_for_BERT

# Get dataset.
dataset = get_dataset_for_BERT('train')
# Tokenize to IDs.
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

print("Tokenization to IDs")
batch = [tokenizer.encode(song) for song in dataset]
print(batch)

model = transformers.TFAutoModel.from_pretrained("bert-base-uncased")

max_length = max(len(sentence) for sentence in batch)
batch_ids = np.zeros([len(batch), max_length], dtype=np.int32)
batch_masks = np.zeros([len(batch), max_length], dtype=np.int32)
for i in range(len(batch)):
    batch_ids[i, :len(batch[i])] = batch[i]
    batch_masks[i, :len(batch[i])] = 1

result = model([batch_ids, batch_masks])
print([component.shape for component in result])