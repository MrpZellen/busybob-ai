from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo, asyncio, json, torch
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from transformers import pipeline
from .judgementLabels import labels, labelsLite


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

# THIS BOB READS FOR GIBBERISH ANSWERS TO FILTER OUT
bobGibberishPipeline = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

# THIS BOB TAGS REMAINING RESPONSES WITH RELEVANT MENTIONS
bobJudgementalPipeline = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

# NO BOB FOR ANSWER QUALITY, MATH THAT MYSELF FOR WEIGHTING OFF OF EXISTING DATA.

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
    resultForGibFilter = await getGibberishSort(strValList)
    print('GIBRES: ', resultForGibFilter)
    #resort data, we don't care about the GIB rating value now that we used it to sort out bad responses.
    # item returned 'key': [{label: str, score: num}] CHANGING MY KEY TO BE THE ANSWER AND QUESTION TOGETHER, separated by '|'
    sentimentResult = []
    for (key, value) in resultForGibFilter.items():
        sentSplit = str(key).split('|')
        # only answer input 1 is answer 0 is question, split
        sentimentResult.append(bobEmotivePipeline(sentSplit[1]))
    # pos/neg sentiment read off phrases.
    print('SENTRES: ', sentimentResult)
    #tagging our replies 
    summarySentiment = []
    # it runs way too slow when the AI processes every survey response especially when custom fields are added.
    # combine into one string, average sentiment check on that without checking every label that unnessecarily.
    oneBigGib = ' '.join([str(gibRes) for gibRes in resultForGibFilter.keys()])
    summarySentiment.append(bobJudgementalPipeline(sequences=oneBigGib, candidate_labels=labels))
    print('SUMRES: ', summarySentiment)
    tagResult = []
    tagResult.append(bobJudgementalPipeline(sequences=[str(key) for key in resultForGibFilter.keys()], candidate_labels=labelsLite)) # longer process, lighter labels to help processing time
    print('TAGRES: ', tagResult)

    # run the algorithm to find top 3 responses in text
    flatTagResult = [item for sublist in tagResult for item in sublist]
    print('starting top3')
    top3Results = gatherTopThree(flatTagResult)

    #precleaning done! make item.
    fullQuestionItem = []
    currentIndex = 0
    for (key, value) in resultForGibFilter.items():
        # for each item that exists in our GibFilter, properly assign all results
        print(key, type(value), value)
        print(sentimentResult[currentIndex])
        print(tagResult[0][currentIndex])
        pocketValue = value[0]
        pocketSentiment = sentimentResult[currentIndex][0]
        pocketTag = tagResult[0][currentIndex]
        fullQuestionItem.append({key: {
            'cleanliness': {
                'label': pocketValue['label'],
                'score': pocketValue['score']
            },
            'connotation': {
                'label': pocketSentiment['label'],
                'score': pocketSentiment['score']
            },
            'tags': {
                'labels': pocketTag['labels'],
                'scores': pocketTag['scores']
            }
        }})
        currentIndex = currentIndex + 1
    myNewResult = {
        "summarySentiment": {
            "values": summarySentiment[0]['labels'],
            "scores": summarySentiment[0]['scores']
        },
        "topThreeResults": top3Results,
        "fullQuestions": fullQuestionItem
    }
    # push to mongoDB
    client = pymongo.MongoClient(MONGODB_URI)
    db = client.survey_data
    collection = db.responses
    newResponse = collection.insert_one(myNewResult).inserted_id
    # return code
    if newResponse:
        return {
            "status": 200,
            "code": "successful completion"
        }
    else:
        return {
            "status": 400,
            "code": "failed to push data changes"
        }

async def getGibberishSort(strValList):
    resultingGibberish = {}
    for key, item in strValList.items():
        print(key + ' Answer: ' + str(item))
        if item: 
            result = bobGibberishPipeline(key + ' Answer: ' + str(item))
            resultingGibberish[key + '|' + str(item)] = result
            print('SCORE! ', result)
    print(resultingGibberish)
    finalGib = {}
    for (key, item) in resultingGibberish.items():
        if item[0]['label'] == 'clean' or (item[0]['label'] == 'mild gibberish' and item[0]['score'] < 0.9):
            finalGib[key] = (item)
    return finalGib

def gatherTopThree(itemsToProcess):
    # compile new object
    ourCandidateStorage = {}
    for item in itemsToProcess:
        nums = item['scores']
        theBiggestNumberThatMakesAllOfTheRules = 0 # index 0 is num, index 1 is index
        theIndexThatDoesnt = 0
        for index, num in enumerate(nums):
            if num > theBiggestNumberThatMakesAllOfTheRules:
                theBiggestNumberThatMakesAllOfTheRules = num
                theIndexThatDoesnt = index
                print('newbig: ', theBiggestNumberThatMakesAllOfTheRules, theIndexThatDoesnt)
        # now translate the magic number back
        biggestNumberWithoutDumbRule = [
            theBiggestNumberThatMakesAllOfTheRules,
            labelsLite[theIndexThatDoesnt] # takes the index and finds the label
        ]
        ourCandidateStorage[item['sequence']] = biggestNumberWithoutDumbRule
        print(ourCandidateStorage[item['sequence']])
    # now we have candidates.
    topThree = []
    print('begintop3')
    for (candidateKey, candidateValue) in list(ourCandidateStorage.items()):
        if len(topThree) == 3:
            print('length 3')
            for index, val in list(enumerate(topThree)):
                if val[1] < candidateValue[0]: # if largestVal of current topThree is less than candidateValue
                    topThree.insert(index, [candidateKey, candidateValue[0], candidateValue[1]])
                    topThree.pop()
                    break;
        else:
            print('lower addition')
            if len(topThree) == 0:
                print('added our first var')
                topThree.append([candidateKey, candidateValue[0], candidateValue[1]]) # 1 is label, 0 is value
            else:
                print('adding a second or third')
                currentLen = len(topThree)
                for index, val in list(enumerate(topThree)):
                    if val[1] < candidateValue[0]: # if largestVal of current topThree is less than candidateValue
                        topThree.insert(index, [candidateKey, candidateValue[0], candidateValue[1]])
                        #no pop here
                if currentLen == len(topThree):
                    topThree.append([candidateKey, candidateValue[0], candidateValue[1]]) # add at end of list if not before
        print('currenttop: ', topThree)
    #top three found
    print('top3: ', topThree)
    return topThree


# STEPS FOR RESPONSE GENERATION:
# retrieve our preprocessed data.
# do an aggregate of the data variables over all instances we have retrieved.
# using the word content and reliability rating weight the responses (adjust it here!!)
# aggregate the positivity and negative ratings as well for an overall vibe, as well as any outliers that are notable
# (considerably bad or good experience)

# THANK YOU CHAT GPT FOR FAKE SURVEY TESTING DATA :)
# fake_survey_response = {
#     "How would you rate your scheduling experience this week?": 9,
#     "Were there any unnecessary schedule items?": "Nothing, everything was perfect",
#     "What schedule additions would you like?": "More coffee breaks and free snacks please",
#     "How fair was the workload this week? (heavily agree to heavily disagre)": "Heavily disagree",
#     "Was assistance readily available when needed? (heavily agree to heavily disagre)": "Agree",
#     "How accessible was your manager? (heavily agree to heavily disagre)": "Neutral",
#     "How comfortable were check-ins with your manager? (heavily agree to heavily disagre)": "Heavily agree",
#     "Were your scheduling needs met?": True,
#     "Was your PTO respected?": False,
#     "Any notes on scheduling issues?": "Manager randomly cancels meetings, confusing",
    
#     "How would you rate your work-from-home / hybrid balance? (heavily agree to heavily disagre)": "Neutral",
#     "How did working from home affect you?": "Work ate all my time, but also had fun meetings lol",
#     "How many breaks did you take this week?": "A few",
#     "Do you have a remote or hybrid setup?": True,
#     "How engaged were virtual meetings? (1-10)": 7,
#     "How could meetings be improved?": "Make them shorter and fun 🤯",
#     "Do you feel heard by management? (1-5)": 3,
#     "Was management's response timely? (heavily agree to heavily disagre)": "Disagree",
#     "Any notes on remote work?": "Zoom fatigue is real",
#     "How often do you seek help when needed? (1-10)": 5,
#     "Any general work-life notes?": "Everything is fine except sometimes the Wi-Fi fails",
    
#     # Company & group specifics
#     "What would you like improved in the break room?": "I wish for better snacks in the break room",
#     "Any other company feedback?": "",  # empty response
#     "Any gibberish or random notes?": "asdf qwer zxcv",
#     "Suggestions for more team events?": "We need more team events",
#     "Any numeric nonsense for text fields?": "0",
# }



# async def testFunctions(request):
#     intValList = {}
#     strValList = {}
#     print("POST endpoint hit!")
#     # body = await request.json()
#     print(f"Received data: {request}")
#     # first we organize.
#     for (key, value) in request.items():
#         print(key, value)
#         if(type(value) == int):
#             intValList[key] = value
#         else:
#             strValList[key] = value
#     # gibberish filter clears out any gibberish responses FIRST
#     resultForGibFilter = await getGibberishSort(strValList)
#     print('GIBRES: ', resultForGibFilter)
#     #resort data, we don't care about the GIB rating value now that we used it to sort out bad responses.
#     # item returned 'key': [{label: str, score: num}] CHANGING MY KEY TO BE THE ANSWER AND QUESTION TOGETHER, separated by '|'
#     sentimentResult = []
#     for (key, value) in resultForGibFilter.items():
#         sentSplit = str(key).split('|')
#         # only answer input 1 is answer 0 is question, split
#         sentimentResult.append(bobEmotivePipeline(sentSplit[1]))
#     # pos/neg sentiment read off phrases.
#     print('SENTRES: ', sentimentResult)
#     #tagging our replies 
#     summarySentiment = []
#     # it runs way too slow when the AI processes every survey response especially when custom fields are added.
#     # combine into one string, average sentiment check on that without checking every label that unnessecarily.
#     oneBigGib = ' '.join([str(gibRes) for gibRes in resultForGibFilter.keys()])
#     summarySentiment.append(bobJudgementalPipeline(sequences=oneBigGib, candidate_labels=labels))
#     print('SUMRES: ', summarySentiment)
#     tagResult = []
#     tagResult.append(bobJudgementalPipeline(sequences=[str(key) for key in resultForGibFilter.keys()], candidate_labels=labelsLite)) # longer process, lighter labels to help processing time
#     print('TAGRES: ', tagResult)

#     # run the algorithm to find top 3 responses in text
#     flatTagResult = [item for sublist in tagResult for item in sublist]
#     print('starting top3')
#     top3Results = gatherTopThree(flatTagResult)

#     #precleaning done! make item.
#     fullQuestionItem = []
#     currentIndex = 0
#     for (key, value) in resultForGibFilter.items():
#         # for each item that exists in our GibFilter, properly assign all results
#         print(key, type(value), value)
#         print(sentimentResult[currentIndex])
#         print(tagResult[0][currentIndex])
#         pocketValue = value[0]
#         pocketSentiment = sentimentResult[currentIndex][0]
#         pocketTag = tagResult[0][currentIndex]
#         fullQuestionItem.append({key: {
#             'cleanliness': {
#                 'label': pocketValue['label'],
#                 'score': pocketValue['score']
#             },
#             'connotation': {
#                 'label': pocketSentiment['label'],
#                 'score': pocketSentiment['score']
#             },
#             'tags': {
#                 'labels': pocketTag['labels'],
#                 'scores': pocketTag['scores']
#             }
#         }})
#         currentIndex = currentIndex + 1
#     myNewResult = {
#         "summarySentiment": {
#             "values": summarySentiment[0]['labels'],
#             "scores": summarySentiment[0]['scores']
#         },
#         "topThreeResults": top3Results,
#         "fullQuestions": fullQuestionItem
#     }
#     # push to mongoDB
#     client = pymongo.MongoClient('mongodb://localhost:27017?directConnection=true')
#     db = client.survey_data
#     collection = db.responses
#     newResponse = collection.insert_one(myNewResult).inserted_id
#     # return code
#     if newResponse:
#         return {
#             "status": 200,
#             "code": "successful completion"
#         }
#     else:
#         return {
#             "status": 400,
#             "code": "failed to push data changes"
#         }

# asyncio.run(testFunctions(fake_survey_response))