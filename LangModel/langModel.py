from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tokenizer = AutoTokenizer.from_pretrained('/content/model')
model = AutoModelForSequenceClassification.from_pretrained('/content/model', num_labels=30)

sentence = "This stock is doing really good."

encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

output = model(**encoding)

logits = output.logits

# Get the predicted class index (argmax) for next sentence prediction
predicted_class = torch.argmax(logits, dim=1).item()

# You can also get the probabilities of each class using softmax
probabilities = torch.softmax(logits, dim=1).tolist()[0]

# Print the results
print("Predicted Class:", predicted_class)
print("Probabilities:", probabilities)