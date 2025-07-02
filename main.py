from fastapi import FastAPI;
# huggingface signin process
from huggingface_hub import login
login(token='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')
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
    result = await llmBIG.ainvoke("ou are a mascot named busybob whose catchphrase is rise and grind. Introduce yourself! The user who you are talking to is inserted below, make a comment about their name and what you like about it! username is tester john")
    return {"message": result.content}