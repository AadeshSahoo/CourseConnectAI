from datasets import load_dataset
import openai
import json
import pandas as pd

openai.api_key = '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'

def generateQuestion(category):
  model_id = 'ft:davinci-002:personal::80jYAUx6'

  prompt = f"generate me an {category} math question"
  max_tokens = 100

  while True:
      response = openai.Completion.create(
          model=model_id,
          temperature=0.2,
          max_tokens=max_tokens,
          prompt=prompt
      )

      generated_text = response.choices[0].text

      split_text = generated_text.split(' ')

      try:
        express_index = split_text.index("Express")

      except:
        max_tokens += 10
        continue

      end_index = 'not defined'

      for x in range(express_index,len(split_text)):
        if split_text[x].endswith('.'):
          end_index = x


      else:
        return(' '.join(split_text[0:end_index+1]))
        break


print(generateQuestion('Algebra'))