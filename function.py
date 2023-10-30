from transformers import AutoTokenizer, FalconForCausalLM, AutoModelForSequenceClassification
import torch 
import openai
import os
from dotenv import load_dotenv

calctokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
calcModel = FalconForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")

langtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
langmodel = AutoModelForSequenceClassification.from_pretrained('/Users/aadeshsahoo/Documents/CourseConnect/LangModel/Modelv2', num_labels=30)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

keys = {
    0:'Parallelism',
    1:'Understatement',
    2:'Antithesis',
    3:'Epithet',
    4:'Aphorism',
    5:'Hyperbole',
    6:'Pathos',
    7:'Ethos',
    8:'Periodic_Sentence',
    9:'Anaphora',
    10:'Syllogism',
    11:'Euphemism',
    12:'Cumulative_Sentence',
    13:'Paradox',
    14:'Logos',
    15:'Apostrophe',
    16:'Allusion',
    17:'Balanced_Sentence',
    18:'Epigram'
}

sentence = "We shall not flag or fail. We shall go on to the end. We shall fight in France, we shall fight on the seas and oceans, we shall fight with growing confidence and growing strength in the air, we shall defend our island, whatever the cost may be, we shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills. We shall never surrender."


def generate_question(category, pr):

    if category == 'AP Calculus':

        input_ids = calctokenizer.encode(pr, return_tensors="pt")

        output = calcModel.generate(input_ids, max_length=500)

        generated_text = calctokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text
    
    elif category == 'AP English and Language':
        
        prompt = f'''
    Generate me a well known piece of next that showcases a salient rhetoric term.

    Please only output the piece of text and nothing else

    Limit the text to three sentences maximum
    '''

        max_tokens = 1

        response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens = 500
                )
        
        response_text = response['choices'][0]['text']

        return str(response_text)
        



prompt = 'Generate me a Calculus BC Level question on: Integration'
print(generate_question('AP Calculus', prompt))

