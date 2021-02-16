from sklearn.model_selection import train_test_split

with open('<path to text file>', 'r') as data:
  dataset = ["<|title|>" + x.strip() for x in data.readlines()]

train, eval = train_test_split(dataset, train_size=.9, random_state=2020)
print("training size:" + str(len(train)))
print("Evaluation size: " + str(len(eval))

with open('train_tmp.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(train))

with open('eval_tmp.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(eval))