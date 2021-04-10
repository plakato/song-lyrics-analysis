import json

from sklearn.model_selection import train_test_split

with open('../../song_data/data/ENlyrics_cleaned.json', 'r') as data:
  songs = json.load(data)
  dataset = ['\n'.join(song['lyrics']) for song in songs]
  print(f"Selected {len(dataset)} songs.")

train, eval = train_test_split(dataset, train_size=.95, random_state=2020)
print("Training size:" + str(len(train)))
print("Evaluation size: " + str(len(eval)))

with open('data/train.txt', 'w+') as file_handle:
  file_handle.write("<|endoftext|>".join(train))

with open('data/eval.txt', 'w+') as file_handle:
  file_handle.write("<|endoftext|>".join(eval))