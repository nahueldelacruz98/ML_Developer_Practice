from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model='llama2')

template = '''
You are an expert in answering questions about LLMs basics.
The model to focus on is: {model}
The list of prompts to be aware of are: {prompts}
The question is: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model  #This chain will build the prompt and then take the choosen model all together in the same loop.

#In the template, to replace variables, use a dictionary and invoke the name of the variables as keys in the dictionary.
result = chain.invoke({"model":"gpt3", "prompts":[], "question":"what is the best way to work with this model?"})

print(result)