from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo, asyncio, json, torch, pprint, datetime
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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

tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")
bobOfTest = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct", device_map="auto" if torch.cuda.is_available() else None)
# FLAN IS EVIL GOOGLE

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bobTestPipeline = pipeline(
    "text-generation",
    model="nvidia/Nemotron-Mini-4B-Instruct",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else None,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
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
    print('holy aggregate lois')
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
    print(db.name)
    collection = db.companies
    print(collection.name)
    resultingObject = collection.find_one({"companyID": collType[7:]})
    print(resultingObject, collType[7:])
    if resultingObject:
        bobDeets = resultingObject["bobInfo"]["specifications"]
    else:
        bobDeets = {
            "avoid": "",
            "tone": "serious",
            "description": 5
        }
    finalItem = await getAIResponse(aggregatedNumData, previousData, bobDeets["avoid"], bobDeets["tone"], bobDeets["description"])
    #send our email and PDF document
    today = datetime.datetime.now()
    formatDay = today.strftime('%d-%m-%Y')
    company = CompanyInstance(resultingObject["companyInfo"]["name"], formatDay)
    # GENERATING PDF
    pdfString = await GenerateDocument(company.date, company.name, finalItem)
    # SENDING MAIL
    pdfName = 'companyResponse_' + formatDay + ".pdf"
    collection = db.users
    for item in resultingObject['adminAccounts']:
        userToSpam = collection.find_one({'_id': item})
        youveGotMail(userToSpam["email"], pdfString, pdfName, company)
    # return a positive response.
    print('CAH CAH IM A BIRD PETAH')
    return {
        "response": "Success compiling! PDFs sent, should be recieved soon", 
        "code": 200
    }


async def getAIResponse(currentData, previousData, avoidedWords, personalityDetails, descriptionRating):
    print('recieved!! now getting response for: ', currentData, previousData)

    message = f"""
        You are Busy Bob, a survey analysis expert. Your personality is: {personalityDetails}
        Avoid using these words: {avoidedWords}
        Descriptiveness level: {descriptionRating}/10

        Analyze this survey data and return a JSON object.


        DATA STRUCTURE EXPLANATION:
        - avgSummaryScore: 0-1 (feedback usefulness, even 0.1 is notable)
        - avgCleanlinessScore: 0-1 (readability)  
        - avgConnotationScore: sentiment score (positive > 2, neutral -0.7 to 0.7, negative < -0.7)
        - avgTagScores: tag presence scores (0-1, consiter 0.05 or higher as substantial enough)

        Analyze the following survey data and respond with ONLY a valid JSON object:
        THIS WEEKS DATA: {str(currentData)}
        LAST TWO WEEKS: {str(previousData)}

        RETURN THIS EXACT JSON FORMAT PLEASE, based on the above data and your instructions.
        {{
            "generalDescription": "Brief summary of key feedback points",
            "toneNotes": "{personalityDetails}",
            "healthRating": 75,
            "dataTrends": {{
                "sentimentTrend": "Sentiment improved by X percentage points compared to previous period",
                "tagTrends": "Work-life balance mentions increased by X%, communication decreased by Y%"
            }},
            "finalThoughts": "Overall assessment and recommendations"
        }}
        """
    print(message)
    response = bobTestPipeline({"role": "system", "content": message})
    preJSONResponse = tokenizer.decode(response[0])
    try:
        resultItem = json.loads(preJSONResponse)
    except Exception as e:
        print(e)
        print('the above went wrong!')
        resultItem = preJSONResponse
    print('bob has given us: ', resultItem)
    return resultItem

def aggregateNums(fullDataObject):
    finalItem = []
    print('length: ', len(fullDataObject))
    for item in fullDataObject:
        print('CHECKIN OUR ITEM OUT: ', item)
        summaryNumsToCompile = []
        summaryQuestionsToCompile = {}
        for key, value in item.items():
            print('current key checked: ', key)
            if key == '_id' or key == 'topThreeResults': # TODO: FIX TOP THREE RESULTS
                continue
            if key == 'summarySentiment':
                summaryNumsToCompile.append(value)
            if key == 'fullQuestions':
                for quesItem in value:
                    if not quesItem:
                        continue
                    print(quesItem)
                    print("Trying to unpack quesItem:", quesItem)
                    innerKey, inner = next(iter(quesItem.items()))
                    print("Unpacked:", innerKey)
                    if inner:
                        summaryQuestionsToCompile[innerKey] = {
                            "cleanliness": {
                                "label": inner['cleanliness']['label'],
                                "score": inner['cleanliness']['score']
                            },
                            "connotation": {
                                "label": inner['connotation']['label'],
                                "score": inner['connotation']['score']
                            },
                            "tags": {
                                "labels": inner['tags']['labels'],
                                "scores": inner['tags']['scores']
                            }
                        }
        # handle summary numbers
        print('we outty')
        finalSumNum = []
        for index in range(len(summaryNumsToCompile)):
            summatedValue = 0
            for value in summaryNumsToCompile:
                print(value)
                summatedValue = summatedValue + value["scores"][index]
            summatedValue = summatedValue / len(summaryNumsToCompile)
            finalSumNum.append(summatedValue)
        print(finalSumNum, 'final sum num')
        # handle summary questions
        finalCleanlinessSum = 0
        for _, currentItem in summaryQuestionsToCompile.items():
            cleanVal = currentItem['cleanliness']
            print(cleanVal)
            print(cleanVal['label'])
            if cleanVal['label'] == 'clean':
                finalCleanlinessSum = finalCleanlinessSum + cleanVal['score']*1.5 #weighted for clean full resp
            else: # case of mild gibberish
                finalCleanlinessSum = finalCleanlinessSum + cleanVal['score']*0.9 #slightly lower weight on anything marked mild gibberish
        finalCleanlinessSum = finalCleanlinessSum / (len(summaryQuestionsToCompile)) #div length of question count
        print('final cleanliness sum: ', finalCleanlinessSum)
        finalPosConSum = 0
        finalNegConSum = 0 
        for _, ratedValue in summaryQuestionsToCompile.items():
            rateKey = ratedValue['connotation']
            if rateKey['label'] == 'POSITIVE':
                finalPosConSum = finalPosConSum + rateKey['score']
            else: # its negative
                finalNegConSum = finalNegConSum + (rateKey['score']*0.8) #tone down negative weighting
        weightedConnotationSum = finalPosConSum - finalNegConSum
        print('weighted connotation sum: ', weightedConnotationSum)
        finalTagSums = {label: 0.0 for label in labelsLite}
        for _, item in summaryQuestionsToCompile.items():
            tags = item['tags']
            for label, score in zip(tags['labels'], tags['scores']):
                if label in finalTagSums:
                    finalTagSums[label] = finalTagSums[label] + score
        #handles tag numbers TODO: maybe see if i can get this less than n^3
        numQuestions = len(summaryQuestionsToCompile)
        for label in finalTagSums:
            finalTagSums[label] = finalTagSums[label] / numQuestions
        # all items handled, aggregation complete. Compile our final object
        resultedItem = {
            "summaryAggregate": finalSumNum,
            "answerCleanlinessScore": finalCleanlinessSum,
            "connotationAggregate": weightedConnotationSum,
            "allTagsSum": finalTagSums
        }
        print(f'Item - APPENDING TO FINAL ITEM: ', resultedItem)
        finalItem.append(resultedItem)
        print(f'Item - finalItem length after append: ', len(finalItem))
    #squish even more
    finalConnotation = 0
    finalCleanliness = 0
    finalSummary = 0
    finalTags  = {}
    for item in (finalItem):
        finalConnotation = finalConnotation + (item["connotationAggregate"] * item["answerCleanlinessScore"])
        finalCleanliness = finalCleanliness + item["answerCleanlinessScore"]
        finalSummary = finalSummary + (item['summaryAggregate'][0] * item["answerCleanlinessScore"])
        for key, value in item["allTagsSum"].items():
            print(key, 'CURRENTKEY')
            if key in finalTags:
                finalTags[key] += value
            else:
                finalTags[key] = value
        print('current clean sum con: ', finalCleanliness, finalSummary, finalConnotation)
        print(finalTags)
    finalConnotation /= len(fullDataObject)
    finalCleanliness /= len(fullDataObject)
    finalSummary /= len(fullDataObject)
    for key, value in finalTags.items():
        finalTags[key] /= 3
    ourLastItemForRealThisTime = {
        "avgSummaryScore": finalSummary,
        "avgCleanlinessScore": finalCleanliness,
        "avgConnotationScore": finalConnotation,
        "avgTagScores": finalTags
    }
    print('all results!!!', ourLastItemForRealThisTime)
    return ourLastItemForRealThisTime


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