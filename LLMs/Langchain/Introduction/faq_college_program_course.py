import pdfplumber
import os

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

path_pdf_program_course = r".\data\3664_IA_Cronograma_Curso-2025-C2_1.pdf"
path_txt_program_course = r".\data\3664_IA_Cronograma_Curso-2025-C2_1.txt"

def read_pdf_data():
    text = ''

    with pdfplumber.open(path_pdf_program_course, ) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    
    #write pdf text in a txt file
    with open(path_txt_program_course, 'w', encoding="utf-8") as file:
        file.write(text)

def create_embeddings():
    #read data
    text = ''
    with open(path_txt_program_course, "r", encoding='utf-8') as file:
        text = file.read()
    
    #choose embedding model
    emb = SentenceTransformerEmbeddings()

    #split text into chunks (you need to do this in order to pass the data into the vector store)
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_text(text)

    #create vectorstore instance
    doc_store = Chroma.from_texts(chunks, emb, persist_directory = './chroma_db_2')

def load_embeddings():
    emb = SentenceTransformerEmbeddings()
    print("Loading embeddings...")
    doc_store = Chroma(persist_directory = './chroma_db_2',
                       embedding_function = emb)
    
    return doc_store


def tool_get_class_date(topic:str):
    vec_store= load_embeddings()

    vector_search = vec_store.similarity_search_with_score(
        query = f'¿Cuál es la fecha en la que se va a tratar el tema de {topic}?',
        k = 2
    )

    return vector_search

def tool_get_complementary_topics(main_topic:str):
    vec_store = load_embeddings()

    vec_search = vec_store.similarity_search_with_score(
        query = f'¿Cuáles son las tareas complementarias que se van a tratar cuando sea la clase de {main_topic}?',
        k = 2
    )

    return vec_search

def create_agent():
    #first - choose LLM
    llm = ChatOllama(
        temperature = 0.0,
        model = 'mistral'
    )

    #second - create list of tools
    tools = [
        Tool(
            name = 'tool_get_class_date',
            func = tool_get_class_date,
            description = 'returns the date of a specific class topic.'
        ),
        Tool(
            name = 'tool_get_complementary_topics',
            func = tool_get_complementary_topics,
            description = 'returns all topics will be taught during the main topic class.'
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )

    #All ready - Now, to use the agent, prepare your prompt. RECOMMENDED: Use a prompt template always to keep a structure.
    template = "I would like to hear more about the important topics during the whole course and specially, focus on {topic} to keep in mind and be prepared for {exam}. You must always respond to the user in Spanish. When calling tools, use only their exact English names and the required format: Action: <tool_name> Action Input: <input>"
    prompt = PromptTemplate(
        input_variables = ['topic', 'exam'],
        template = template
    )

    result = agent.run(prompt.format_prompt(topic = 'agentes', exam = 'Primer parcial'))

    print(f'Results: {result}' )


#read_pdf_data()
create_embeddings()
#load_embeddings()
#create_agent()