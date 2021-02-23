import json

from sklearn.model_selection import train_test_split

with open('../../song_data/data/ENlyrics_cleaned.json', 'r') as data:
  songs = json.load(data)
  # Filter only pop lyrics.
  dataset = ['\n'.join(song['lyrics']) for song in songs if song['genre'] == 'pop']
  print(f"Selected {len(dataset)} pop songs.")

train, eval = train_test_split(dataset, train_size=.9, random_state=2020)
print("Training size:" + str(len(train)))
print("Evaluation size: " + str(len(eval)))

with open('data/train.txt', 'w+') as file_handle:
  file_handle.write("<|endoftext|>".join(train))

with open('data/eval.txt', 'w+') as file_handle:
  file_handle.write("<|endoftext|>".join(eval))