import json
from datasets import load_dataset


data_dict = load_dataset("rotten_tomatoes", split="train")
print(data_dict)
with open("./data/testData.json", "w") as f:
    json.dump(data_dict, f)