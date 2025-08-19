from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os, pymongo
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from BusyBob import post_root


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



@app.get("/debug")
async def debug_routes():
    routes = []
    for route in app.routes:
        routes.append(f"{route.methods} {route.path}")
    return {"routes": routes}

@app.post('/')
async def pagingBob(request: Request):
    print('REACHED!!!')
    body = await request.json()
    print('BODY:', body)
    try:
        print('request', request)
        responseObject = await post_root(body)
        print(responseObject)
        return {"response": responseObject}
    except Exception as e:
        print(f"Error in pagingBob: {e}")
        return {
            "error": str(e),
            "status": 500
        }