import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset, load_dataset, DatasetDict
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer, get_scheduler, AutoModelForCausalLM, BertGenerationEncoder, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")

model.to_bettertransformer()

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=30)

def preprocess_function(examples):
    return tokenizer(examples["quesiton"], truncation=True)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# keys = {
#     0:'Analytical_Applications_of_Differentiation',
#     1:'Applications_of_Integration',
#     2:'Composite_Implicit_and_Inverse_Functions',
#     3:'Contexual_Applications_of_Differentiation',
#     4:'Differential_Equations',
#     5:'Fundamentals_of_Differentiation',
#     6:'Infinite_Sequences_and_Series',
#     7:'Integration_and_Accumulation_of_Change',
#     8:'Limits_and_Continuity',
#     9:'Parametric_Equations_polar coordinates_and_vector-values_functions',
# }

# dontTrain = []

# df = pd.DataFrame()

# for term in keys:

#     df2 = pd.read_csv(f'{keys[term]}_data.csv')
#     df2 = df2.drop('question',axis=1)
#     df2 = df2.drop('Unnamed: 0',axis=1)
#     df = df.append(df2, ignore_index=True)


# df['topic'] = 'Generate me a calculus question on ' + df['topic']

# df = df.dropna()

# df.to_csv('allCalcData.csv')

local_csv = load_dataset('csv',split='train',data_files='allCalcData.csv')
local_csv = local_csv.train_test_split(test_size=0.1)
filteredDataset = local_csv.shuffle(seed=42)

block_size = 64
def group_texts(examples):
    concatenated_examples = {k: examples[k] for k in examples.keys()}
    result = {
        k: [t[i : i + block_size] for i in range(0, len(concatenated_examples[k]), block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = filteredDataset.map(preprocess_function, batched=True,num_proc=4)

lm_dataset = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

print(lm_dataset['train'])
print(lm_dataset['train']['input_ids'][0])

print(f"Number of training examples: {len(lm_dataset['train'])}")
# lm_dataset = lm_dataset.remove_columns(tokenized_datasets["train"].column_names)

print(f"Number of training examples: {len(lm_dataset['train'])}")
# print(lm_dataset['train']['labels'][0])
# lm_dataset = lm_dataset.rename_column("labels", "input_ids")


# tokenized_datasets = tokenized_datasets.remove_columns(['Unnamed: 0','quesiton'])
# tokenized_datasets = tokenized_datasets.rename_column('topic','labels')
print(tokenized_datasets)
tokenizer.pad_token = tokenizer.eos_token 

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/Users/aadeshsahoo/Documents/CourseConnect/CalcModel/Model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(tokenized_datasets['train'])

trainer.train()

# Calculate the total number of training steps
total_steps = len(trainer.train_dataloader) * training_args.num_train_epochs

# Create a tqdm progress bar
progress_bar = tqdm(total=total_steps, desc="Training")

# Training loop
for epoch in range(int(training_args.num_train_epochs), desc="Epoch"):
    trainer.train()
    progress_bar.update(len(trainer.train_dataloader))

# Close the progress bar
progress_bar.close()
