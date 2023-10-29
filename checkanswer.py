import openai
from transformers import AutoTokenizer, FalconForCausalLM, AutoModelForSequenceClassification
import torch 
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

langtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
langmodel = AutoModelForSequenceClassification.from_pretrained('/Users/aadeshsahoo/Documents/CourseConnect/LangModel/Modelv2', num_labels=30)

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

def answer_question(category, question, user_answer):

    if category == 'AP Calculus':
        prompt = f'''
    question: {question}
    answer: {user_answer}

    please only respond with 1 if this is correct and 0 if it is incorrect. do not provide any explanation
    '''

        max_tokens = 1

        response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens = 500
                )
        
        response_text = response['choices'][0]['text']

        return int(response_text)
    
    elif category == 'AP English and Language':
       
        tokens = langtokenizer(question, return_tensors='pt', padding=True, truncation=True)

        outputs = langmodel(**tokens)

        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=1)


        if predicted_class.lower() == user_answer.lower():
           return 1
        else:
           return 0 
       
       
       
