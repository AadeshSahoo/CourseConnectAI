from datasets import load_dataset
import openai
import json
import pandas as pd

openai.api_key = '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'

categories = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]

combined_train_df = pd.DataFrame()
combined_test_df = pd.DataFrame()

split_train = "train"
split_test = "test"

for subject in categories:
    dataset = load_dataset('baber/hendrycks_math', subject)

    data_split_train = dataset[split_train]
    data_split_test = dataset[split_test]

    train_df = pd.DataFrame(data_split_train)
    test_df = pd.DataFrame(data_split_test)

    combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
    combined_test_df = pd.concat([combined_test_df, test_df], ignore_index=True)

training_data = []

for row in combined_train_df.iterrows():
  training_data.append(
      {'prompt':f"generate me a {row[1]['type']} math question",
       'completion':row[1]['problem']
       }
  )

output_file = "data.jsonl"

with open(output_file, "w") as jsonl_file:
    for row in training_data:
        json_line = json.dumps(row)
        jsonl_file.write(json_line + "\n")

openai.File.create(
  file=open("data.jsonl", "rb"),
  purpose='fine-tune'
)

openai.FineTuningJob.create(training_file="file-Cg5KY8YQG6MAJfjQl3X4VvDF", model="davinci-002")