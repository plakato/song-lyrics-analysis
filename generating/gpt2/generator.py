# setup imports to use the model
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Encode input text.
input_ids = tokenizer.encode("Roses are red\nviolets are blue\ntell me Ted\ndo you have a flu?", return_tensors='tf')
print(input_ids)
print()
input_ids = tokenizer.encode("Roses are red violets are blue tell me Ted do you have a flu?", return_tensors='tf')
print(input_ids)

generated_text_samples = model.generate(
    input_ids,
    max_length=150,
    num_return_sequences=5,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,
    top_p=0.92,
    temperature=.85,
    do_sample=True,
    top_k=125,
    early_stopping=True
)

# Print output for each sequence generated above
for i, beam in enumerate(generated_text_samples):
  print("{}: {}".format(i, tokenizer.decode(beam, skip_special_tokens=True)))
  print()