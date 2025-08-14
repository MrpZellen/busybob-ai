from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo, asyncio, torch
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from transformers import pipeline
from judgementLabels import labels



# STEPS FOR DATA PROCESSING: 
# WE RECIEVE THE DATA FROM THE API CALL 
# THEN WE DO PREPROCESSING:
# - then we take this reply, and get a general positivity and negativity rating from each segment
# - take our individual response and give it a reliablility ranking (is this response good)

# Thats the preprocessing we do in this step, store that data.

load_dotenv()

# huggingface signin process
HF_TOKEN = os.getenv("HF_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
if not HF_TOKEN or not MONGODB_URI:
    raise ValueError("HF_TOKEN is missing from environment variables")

login(token=HF_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SurveyData(BaseModel):
    schedulingMeetups: Dict[str, Any]
    workLifeBalance: Dict[str, Any]
    companySpecific: Dict[str, Any]


#|--------------------|
#|    BOB LLM         |
#|     REGISTRY       |
#|--------------------|

# THIS BOB TESTS 
bobOfTest = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    do_sample=False,
)

# THIS BOB READS POSITIVE AND NEGATIVE SENTIMENT
bobEmotivePipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# wrapper
bobOfEmotion = HuggingFacePipeline(pipeline=bobEmotivePipeline)

# THIS BOB READS FOR GIBBERISH ANSWERS TO FILTER OUT
bobGibberishPipeline = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

# THIS BOB TAGS REMAINING RESPONSES WITH RELEVANT MENTIONS
bobJudgementalPipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# wrapper
bobOfJudgement = HuggingFacePipeline(pipeline=bobJudgementalPipeline)
sentimentList = labels

# NO BOB FOR ANSWER QUALITY, MATH THAT MYSELF FOR WEIGHTING OFF OF EXISTING DATA.




@app.post("/busybob")
async def post_root(request: Request):
    intValList = {}
    strValList = {}
    print("POST endpoint hit!")
    body = await request.json()
    print(f"Received data: {body}")
    # first we organize.
    for (key, value) in body.items():
        print(key, value)
        if(type(value) == int):
            intValList[key] = value
        else:
            strValList[key] = value
    # gibberish filter clears out any gibberish responses FIRST
    resultingGibberish = {}
    for key, item in strValList.items():
        print(key + ' Answer: ' + str(item))
        if item: 
            result = bobGibberishPipeline(key + ' Answer: ' + str(item))
            resultingGibberish[key] = result
            print('SCORE! ', result)
    print(resultingGibberish)
    finalGib = {}
    for (key, item) in resultingGibberish.items():
        if item[0]['label'] == 'clean' or (item[0]['label'] == 'mild gibberish' and item[0]['score'] < 0.9):
            finalGib[key] = (item)
    # pos/neg sentiment read off phrases.
    sentimentResult = bobOfEmotion.agenerate(strValList)

    
    return {
        "status": 200,
        "code": "successful completion"
    }

# STEPS FOR RESPONSE GENERATION:
# retrieve our preprocessed data.
# do an aggregate of the data variables over all instances we have retrieved.
# using the word content and reliability rating weight the responses (adjust it here!!)
# aggregate the positivity and negative ratings as well for an overall vibe, as well as any outliers that are notable
# (considerably bad or good experience)

# THANK YOU CHAT GPT FOR FAKE SURVEY TESTING DATA :)
fake_survey_response = {
    "How would you rate your scheduling experience this week?": 9,
    "Were there any unnecessary schedule items?": "Nothing, everything was perfect",
    "What schedule additions would you like?": "More coffee breaks and free snacks please",
    "How fair was the workload this week? (heavily agree to heavily disagre)": "Heavily disagree",
    "Was assistance readily available when needed? (heavily agree to heavily disagre)": "Agree",
    "How accessible was your manager? (heavily agree to heavily disagre)": "Neutral",
    "How comfortable were check-ins with your manager? (heavily agree to heavily disagre)": "Heavily agree",
    "Were your scheduling needs met?": True,
    "Was your PTO respected?": False,
    "Any notes on scheduling issues?": "Manager randomly cancels meetings, confusing",
    
    "How would you rate your work-from-home / hybrid balance? (heavily agree to heavily disagre)": "Neutral",
    "How did working from home affect you?": "Work ate all my time, but also had fun meetings lol",
    "How many breaks did you take this week?": "A few",
    "Do you have a remote or hybrid setup?": True,
    "How engaged were virtual meetings? (1-10)": 7,
    "How could meetings be improved?": "Make them shorter and fun 🤯",
    "Do you feel heard by management? (1-5)": 3,
    "Was management's response timely? (heavily agree to heavily disagre)": "Disagree",
    "Any notes on remote work?": "Zoom fatigue is real",
    "How often do you seek help when needed? (1-10)": 5,
    "Any general work-life notes?": "Everything is fine except sometimes the Wi-Fi fails",
    
    # Company & group specifics
    "What would you like improved in the break room?": "I wish for better snacks in the break room",
    "Any other company feedback?": "",  # empty response
    "Any gibberish or random notes?": "asdf qwer zxcv",
    "Suggestions for more team events?": "We need more team events",
    "Any numeric nonsense for text fields?": "0",
}



async def testFunctions(request):
    intValList = {}
    strValList = {}
    print("POST endpoint hit!")
    print(f"Received data: {request}")
    # first we organize.
    for (key, value) in request.items():
        print(key, value)
        if(type(value) == int):
            intValList[key] = value
        else:
            strValList[key] = value
    # gibberish filter clears out any gibberish responses FIRST
    resultingGibberish = {}
    for key, item in strValList.items():
        print(key + ' Answer: ' + str(item))
        if item: 
            result = bobGibberishPipeline(key + ' Answer: ' + str(item))
            resultingGibberish[key] = result
            print('SCORE! ', result)
    print(resultingGibberish)
    finalGib = {}
    for (key, item) in resultingGibberish.items():
        if item[0]['label'] == 'clean' or (item[0]['label'] == 'mild gibberish' and item[0]['score'] < 0.9):
            finalGib[key] = (item)
    # pos/neg sentiment read off phrases.
    # sentimentResult = await bobOfEmotion.agenerate(strValList)
    # print('the sentiment is felt: ', sentimentResult)
    print(finalGib)

    
    return {
        "status": 200,
        "code": "successful completion"
    }


asyncio.run(testFunctions(fake_survey_response))