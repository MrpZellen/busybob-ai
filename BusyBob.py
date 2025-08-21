from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo, asyncio, json, torch, pprint, datetime
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from transformers import pipeline
from judgementLabels import labels, labelsLite
from sendPDF import youveGotMail, CompanyInstance
from ReportGeneration import GenerateDocument


# STEPS FOR DATA PROCESSING: 
# WE RECIEVE THE DATA FROM THE API CALL 
# THEN WE DO PREPROCESSING:
# - then we take this reply, and get a general positivity and negativity rating from each segment
# - take our individual response and give it a reliablility ranking (is this response good)

# Thats the preprocessing we do in this step, store that data.

load_dotenv()

# huggingface signin process
HF_TOKEN = os.getenv("HF_TOKEN")
MONGODB_URI = os.getenv("MONGOPDF_URI")
if not HF_TOKEN or not MONGODB_URI:
    raise ValueError("HF_TOKEN is missing from environment variables")

login(token=HF_TOKEN)


class SurveyData(BaseModel):
    schedulingMeetups: Dict[str, Any]
    workLifeBalance: Dict[str, Any]
    companySpecific: Dict[str, Any]


#|--------------------|
#|    BOB LLM         |
#|     REGISTRY       |
#|--------------------|

# THIS BOB READS POSITIVE AND NEGATIVE SENTIMENT
bobEmotivePipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# THIS BOB READS FOR GIBBERISH ANSWERS TO FILTER OUT
bobGibberishPipeline = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

# THIS BOB TAGS REMAINING RESPONSES WITH RELEVANT MENTIONS
bobJudgementalPipeline = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

# NO BOB FOR ANSWER QUALITY, MATH THAT MYSELF FOR WEIGHTING OFF OF EXISTING DATA.

# NO BOB FOR AGGREGATION, DUH DOY.

# THIS BOB PROVIDES A SUMMARY OF THE INFORMATION ITS FED, AGGREGATE SCORES ON QUESTIONS. 
bobOfTest = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    do_sample=False,
)

async def post_root(request: Request):
    intValList = {}
    strValList = {}
    print("POST endpoint!")
    print('JSONT TIME')
    body = {}
    collType = request['coll']
    for key, nested_dict in request['data'].items():
        print('IN HERE')
        if isinstance(nested_dict, dict):
            for nested_key, nested_value in nested_dict.items():
                body[f"{key}_{nested_key}"] = nested_value
        else:
            body[key] = nested_dict
    print(body, 'flattened?')
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
                'label': str(pocketValue['label']),
                'score': float(pocketValue['score'])
            },
            'connotation': {
                'label': str(pocketSentiment['label']),
                'score': float(pocketSentiment['score'])
            },
            'tags': {
                'labels': [str(string) for string in pocketTag["labels"]],
                'scores': [float(fl) for fl in pocketTag["scores"]],
            }
        }})
        currentIndex = currentIndex + 1
    myNewResult = {
        "summarySentiment": {
            "values": [str(string) for string in summarySentiment[0]['labels']],
            "scores": [float(fl) for fl in summarySentiment[0]['scores']]
        },
        "topThreeResults": top3Results,
        "fullQuestions": fullQuestionItem
    }
    # push to mongoDB
    client = pymongo.MongoClient(MONGODB_URI)
    db = client.survey_data
    collection = db[collType]
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


async def processResults(request: Request):
    # this is called when the cron weekly crons all over and decides its time to process results and close the survey.
    print('CRON HAS PROCESSED. WE ARE BOBLINE.')
    collType = request['collection']
    print(collType)
    # get mongo items.
    print('past that')
    client = pymongo.MongoClient(MONGODB_URI)
    print('client reached')
    db = client.survey_data
    print('db connected: ', db)
    collection = db[collType]
    print('collection found: ', collection)
    allMyData = []
    for doc in collection.find():
        doc["_id"] = str(doc["_id"])
        allMyData.append(doc)
    aggregatedNumData = aggregateNums(allMyData) # returns aggregated values for each number based response.
    #holy, what a lot of data to process.
    # now that we have processed all of that info, we can feed it into our last AI.
    strippedColl = collType[7:] + "_previousResults"
    collection = db[strippedColl]
    previousData = []
    for doc in collection.find().sort('_id', pymongo.DESCENDING).limit(2):
        doc["_id"] = str(doc["_id"])
        previousData.append(str(doc))
    # SEARCHING THE COMPANY
    db = client.company_storage
    collection = db.companies
    resultingObject = collection.find_one(request['companyID'])
    bobDeets = resultingObject["bobInfo"]["specifications"]
    finalItem = await getAIResponse(aggregatedNumData, previousData, bobDeets["avoid"], bobDeets["tone"], bobDeets["description"])
    #send our email and PDF document
    today = datetime.date.today()
    formatDay = today.strftime('%d-%m-%Y')
    company = CompanyInstance(resultingObject["companyInfo"]["name"], formatDay)
    # GENERATING PDF
    pdfString = GenerateDocument(company.date, company.name, finalItem)
    # SENDING MAIL
    pdfName = 'companyResponse_' + formatDay + ".pdf"
    collection = db.users
    for item in resultingObject['adminAccounts']:
        userToSpam = collection.find_one({'_id': item})
        youveGotMail(userToSpam["email"], pdfString, pdfName, company)
    # return a positive response.
    return {
        "response": "Success compiling! PDFs sent, should be recieved soon", 
        "code": 200
        }


async def getAIResponse(currentData, previousData, avoidedWords, personalityDetails, descriptionRating):
    print('recieved!! now getting response for: ', currentData, previousData)

    inputString = f'''Your goal is to iterperet the following data and return a JSON object that meets the specified result structure.
        Your name is Busy Bob, and factor in the following inclusions:
        INCLUSIONS:
        avoid the words: {avoidedWords}
        your personality is: {personalityDetails}, but your primary focus should remain to be accurate and clear in your deductions.
        On a scale of 1-10, you should be this descriptive: {descriptionRating}

        Each dataset contains:
        - summaryAggregate: 0-1, how useful the feedback is
        - answerCleanlinessScore: 0-1, how readable/clear the answers were
        - connotationAggregate: -1 to 1, how negative or positive the sentiment is
        - allTagsSum: a dictionary of tags with scores (0-1) for presence.

        Current dataset:
        {currentData}

        Past 2 datasets:
        {previousData}

        TASKS:
        item0: just this list: {personalityDetails}
        item1: Give a string that is a general description of key points and the most valuable feedback.
        item2: Give me a number 1-100 that tells me the general company health rating.
        item3: Identify the following trends from the last dataset: 
        item3_1: is sentiment improving or worsening? Show this by returning a string response on the sentiment improvement, giving numbers in the reply as well to describe trends. USE PERCENTAGE POINTS!
        item3_2: what are the major changes in tags? show this by describing in percentages the increase or decrease in certain feedback items over time.
        item4: an overall summary string at the bottom for final thoughts, 



        REQUIRED RESULT STRUCTURE:
        data: {
            "generalDescription": item1,
            "toneNotes": personalityDetails,
            "healthRating": item2,
            "dataTrends": {
                "sentimentTrend": item3_1
                "tagTrends": item3_2
            },
            "finalThoughts": item4
        }
        RETURN JUST THIS JSON, NOTHING ELSE. RETURN IT IN THE PROPER SHAPE TO BE FORMATTED INTO A PYTHON DICTIONARY'''
    resultItem = bobOfTest.invoke(inputString)
    print('bob has given us: ', resultItem)
    return dict(resultItem)

def aggregateNums(fullDataObject):
    finalItem = {}
    for item in fullDataObject:
        print(item)
        summaryNumsToCompile = []
        summaryQuestionsToCompile = {}
        for key, value in item.items():
            if key == '_id' or key == 'topThreeResults': # TODO: FIX TOP THREE RESULTS
                continue
            if key == 'summarySentiment':
                summaryNumsToCompile.append(value[1])
            if key == 'fullQuestions':
                for quesKey, quesItem in value.items():
                    summaryQuestionsToCompile.update({
                        "cleanliness": {
                            "label": quesItem['cleanliness']['label'],
                            "score": quesItem['cleanliness']['score']
                        },
                        "connotation": {
                            "label": quesItem['connotation']['label'],
                            "score": quesItem['connotation']['score']
                        },
                        "tags": {
                            "labels": quesItem['tags']['labels'],
                            "scores": quesItem['tags']['scores']
                        }
                    })
        # handle summary numbers
        finalSumNum = []
        for index in range(len(summaryNumsToCompile)):
            summatedValue = 0
            for value in summaryNumsToCompile:
                summatedValue = summatedValue + value[index]
            summatedValue = summatedValue / range(len(summaryNumsToCompile))
            finalSumNum.append(summatedValue)
        print(finalSumNum, 'final sum num')
        # handle summary questions
        finalCleanlinessSum = 0
        for cleanKey, cleanVal in summaryQuestionsToCompile['cleanliness'].items():
            if cleanKey == 'clean':
                finalCleanlinessSum = finalCleanlinessSum + cleanVal*1.5 #weighted for clean full resp
            else: # case of mild gibberish
                finalCleanlinessSum = finalCleanlinessSum + cleanVal*0.75 #slightly lower weight on anything marked mild gibberish
        finalCleanlinessSum = finalCleanlinessSum / len(summaryQuestionsToCompile) #div length of question count
        print('final cleanliness sum: ', finalCleanlinessSum)
        finalPosConSum = 0
        finalNegConSum = 0 
        for rateKey, rateVal in summaryQuestionsToCompile['connotation'].items():
            if rateKey == 'POSITIVE':
                finalPosConSum = finalPosConSum + rateVal
            else: # its negative
                finalNegConSum = finalNegConSum + (rateVal*0.6) #tone down negative weighting
        weightedConnotationSum = finalPosConSum - finalNegConSum
        print('weighted connotation sum: ', weightedConnotationSum)
        finalTagSums = {}
        #handles tag numbers TODO: maybe see if i can get this less than n^3
        for label in labelsLite:
            foundIndex = 0
            labelAvg = 0
            for key, value in summaryQuestionsToCompile['tags']['labels'].items():
                if key == label:
                    for item in summaryQuestionsToCompile:
                        labelAvg = labelAvg + item['tags']['scores'][foundIndex]
                    break
                else:
                    foundIndex = foundIndex + 1
            labelAvg = labelAvg / len(summaryQuestionsToCompile)
            finalTagSums.update({
                "labelAvg": labelAvg,
                "label": label
            })
            # all items handled, aggregation complete. Compile our final object
        finalItem.update({
            "summaryAggregate": finalSumNum,
            "answerCleanlinessScore": finalCleanlinessSum,
            "connotationAggregate": weightedConnotationSum,
            "allTagsSum": finalTagSums
        })
        print(finalItem, "aggregation results")
    print('all results!!!', finalItem)
    return finalItem


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
        if item[0]['label'] == 'clean' or (item[0]['label'] == 'mild gibberish' and item[0]['score'] > 0.6):
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