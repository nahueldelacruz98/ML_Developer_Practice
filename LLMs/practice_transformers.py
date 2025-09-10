from transformers import pipeline
import torch

def generate_random_text(input_text, model_name):
    # Load a text-generation pipeline
    generator = None

    if model_name == 'gpt2':
        generator = pipeline("text-generation", model="gpt2")
    elif model_name == 'llama':
        generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    else:
        raise(Exception(""))

    # Generate text from a prompt
    result = generator( input_text, 
                   max_length=100, 
                   num_return_sequences=1)


    print(result[0]["generated_text"])


model_name = "llama"
prompt = "For you, what is the favorite color of Messi?"

generate_random_text(prompt, model_name=model_name)