from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from busybob import BusyBob


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

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    do_sample=False,
)

async def pymongtest():
    print('reached pytest')
    client = pymongo.MongoClient(MONGODB_URI)
    db = client.survey_data
    collection = db.responses
    if client:
        return True
    else:
        return False


@app.post("/")
async def post_root(request: Request):
    print("POST endpoint hit!")
    body = await request.json()
    print(f"Received data: {body}")
    
    llmBIG = ChatHuggingFace(llm=llm)
    testerMong = await pymongtest()

    invokeString = f"you are a mascot named busybob whose catchphrase is rise and grind (he's a little confused but has the spirit). " \
    "Introduce yourself! The user who you are talking to is inserted below, make a comment about their name and what you like about it! " \
    "username is {0}".format(body['user'])

    result = await llmBIG.ainvoke(invokeString)
    
    return {
        "message": result.content,
        "connectedToMongo": testerMong,
        "receivedData": body if body else None
    }
@app.get("/debug")
async def debug_routes():
    routes = []
    for route in app.routes:
        routes.append(f"{route.methods} {route.path}")
    return {"routes": routes}

@app.post('/busybob')
async def pagingBob(request: Request):
    responseObject = BusyBob.post_root(request=request)
    print(responseObject)
    return {
        'response': responseObject
    }