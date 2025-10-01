from dotenv import load_dotenv
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent


PATH_DATASET_CSV = r".\Introduction\McDonalds-Yelp-Sentiment-DFE.csv"
load_dotenv()

def load_dataset():
    df = pd.read_csv(PATH_DATASET_CSV, delimiter=',', encoding="ISO-8859-1")
    #Get only the reviews column 

    df_return = pd.DataFrame(df['review'])

    return df_return

def get_embeddings(data:pd.DataFrame):
    '''
    Keep in mind that Chroma.from_documents() functions expects a list of Document object (from Langchain)
    So you cannot pass a DataFrame. But you can create a Document object using a DataFrame
    1. Create embeddings
    2. Create document object (has the following properties)
        - page_content
        - metadata
    3. Build embeddings
    4. Save embeddings (so you dont need to load embeddings every time. It takes some minutes to load)
        - To save embeddings, must use "persist_directory" property from Chroma to save them.
        - Then, invoke 'chroma.persist()' function

    RETURNS a VECTOR DATABASE using chroma function (from_documents)
    '''

    embeddings = SentenceTransformerEmbeddings()

    docs = [Document(page_content=row['review'], metadata=row.to_dict()) for _,row in data.iterrows()]

    doc_search = Chroma.from_documents(
        documents = docs, 
        embedding = embeddings, 
        persist_directory='./chroma_db')
    
    doc_search.persist()

    print("embeddings created successfully")

    return doc_search

def load_embeddings():
    '''
    Load embeddings that were built previously in the function 'get_embeddings'
    1. Set embeddings function
    2. load embeddings
    '''

    embeddings = SentenceTransformerEmbeddings()
    print("Loading embeddings...")
    doc_search = Chroma(persist_directory = './chroma_db', embedding_function = embeddings)
    print("Embeddings loaded.")
    return doc_search


def search(search_input):
    '''
    IMPORTANT: this function is created as a new tool for our agent 
    And within this function are the embeddings   
    '''
    doc_search = load_embeddings()
    docs = doc_search.similarity_search_with_score(search_input, k=2)
    return docs


def get_agent():
    '''
    1. Create tools
    2. Create prompt template
    3. Create LLM instance
    4. Create agent using tools and LLM
    '''
    tools = [Tool(
        name='Search',
        func= search,
        description= 'Return reviews about MC Donalds food related to the search term.'
    )]
    print("Tool 'Search' was set.")

    #Get LLM
    #llm = ChatOpenAI(temperature=0.0, openai_api_key= os.getenv('OPENAI_API_KEY'))
    llm = ChatOllama(temperature=0.0,
                     model='llama2')

    print("LLM instance created with OpenAI")

    #Create agent - Use tools and LLM
    agent = initialize_agent(
        tools,
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True  #this property shows the reasoning of the agent.
    )

    template = "How is the quality of {service} at McDonalds according to customer reviews? Highlight positive and negative comments if any and set a rating between 1 to 5 stars"
    prompt = PromptTemplate(
        input_variables=["service"],
        template=template
    )

    #Shows a LLM response without agent capabilities
    response = llm.predict(prompt.format(service="potatoes"))

    print(f"Agent created successfully {response}")

    return agent

def run_prompt_reasoning(agent):
    template = "How is the quality of {service} at McDonalds according to customer reviews? Highlight positive and negative comments if any and set a rating between 1 to 5 stars"
    prompt = PromptTemplate(
        input_variables=["service"],
        template=template
    )

    #
    response = agent.run(prompt.format_prompt(service="potatoes"))
    print(f"Response: {response}")


df_reviews = load_dataset()
print(df_reviews.head(10))
#get_embeddings(df_reviews)
agent = get_agent()
run_prompt_reasoning(agent)