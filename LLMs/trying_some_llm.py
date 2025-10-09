# Import pipeline for creating a summarization pipeline

from transformers import pipeline
import torch
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
                      
def openai_light_model():
    # Use a pipeline as a high-level helper

    model_id = "openai/gpt-oss-120b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


def ollama_model():
    
    model_id = "meta-llama/Llama-3.1-8B"

    pipe = pipeline(
        "text-generation", model=model_id, use_auth_token=os.getenv("HF_ACCESS_TOKEN"), model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        
    )

    print(pipe("Hey how are you doing today?"))

def mistral_model():
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2"
        #token=os.environ.get("HF_ACCESS_TOKEN") # Safely loads token from environment
    )

    messages = [
        {"role": "user", "content": "What are the three most important things to know about Mistral-7B-Instruct-v0.2?"}
    ]

    # Use chat.completions.create for conversational tasks
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2", # May be needed again depending on your client setup
        messages=messages,
        max_tokens=256
    )

    # Extract the generated text
    generated_text = response.choices[0].message.content
    print(generated_text)

#ollama_model() 
#openai_light_model()
mistral_model()  #continue exploring huggingface models: https://www.youtube.com/watch?v=RL1XxStSzgA