from fastapi import FastAPI;
import os;
# huggingface signin process
from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

login(token=HF_TOKEN)
app = FastAPI()

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
# access token: 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
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    do_sample=False,
)

@app.get("/")
async def root():
    llmBIG = ChatHuggingFace(llm=llm)
    result = await llmBIG.ainvoke("you are a mascot named busybob whose catchphrase is rise and grind. Introduce yourself! The user who you are talking to is inserted below, make a comment about their name and what you like about it! username is tester john")
    return {"message": result.content}