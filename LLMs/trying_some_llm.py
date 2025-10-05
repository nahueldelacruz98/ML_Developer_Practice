# Import pipeline for creating a summarization pipeline

from transformers import pipeline
import torch
from dotenv import load_dotenv
import os

load_dotenv()
                      
def openai_light_model():
    # Use a pipeline as a high-level helper

    pipe = pipeline("text-generation", model="openai/gpt-oss-20b")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    print(pipe(messages))

def ollama_model():
    
    model_id = "meta-llama/Llama-3.1-8B"

    pipe = pipeline(
        "text-generation", model=model_id, use_auth_token=os.getenv("HF_ACCESS_TOKEN"), model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        
    )

    print(pipe("Hey how are you doing today?"))

ollama_model() 