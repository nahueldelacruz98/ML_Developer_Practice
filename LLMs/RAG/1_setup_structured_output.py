from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.output_parsers import StructuredOutputParser,PydanticOutputParser, ResponseSchema

prompt_text_pdf_analysis = '''
Summarize this pdf text:
                      
Data pipelines works well with static and small datasets, but what if you need to
	- schedule this functions as new data arrives? (like triggers)
	- Scale these workflows
	- Share these workflows to downstream applications
	- Monitor these workflows


So we need to breakdown our data pipeline
Some tools to do this is Airflow

Airflow:
	- Open source platform
	- Its also useful for data engineering
	- run any type of workflow (not only python)
	- Users can take advantage of this tool to define pipelines.
	- workflows are defined by DAGs

DAGs
	- Directed Acyclic Graphs
	- Good practice. Keep just one DAG per file.

Useful link to take practice: https://towardsdatascience.com/end-to-end-machine-learning-pipeline-with-docker-and-apache-airflow-from-scratch-35f6a75f57ad/  
                      '''

def choose_llm(class_llm):
    llm = ChatOllama(
        temperature='0.0',
        model='mistral'
    )

    return llm

def parser_using_pydantic_classes():
    class SummaryResponse(BaseModel):
        title: str
        key_points: list[str]
        sentiment: str

    parser = PydanticOutputParser(pydantic_object=SummaryResponse)
    return parser

def parser_using_schemas():
    gift_schema = ResponseSchema(name="title",   
                                 description="Highlight the main topic.")
    delivery_days_schema = ResponseSchema(name="key_points",
                                          description="list the main tasks and subjects the article is talking about.")
    price_value_schema = ResponseSchema(name="sentiment",        
                                        description="Explain your thougts and overview about the topic.")
    
    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    return parser

#Change between using either PYDANTIC or SCHEMAS parsers
def get_llm_formatted_output():
    '''
    In order to use a parser with a OLLAMA llm model, we need to create a CHAIN.

    chain = prompt | llm | parser (parser must be at the end since it needs the llm ouput)
    chain = llm | parser (the prompt must be invoked later)
    '''

    llm = ChatOllama(temperature = 0.0,
                     model = 'mistral')
    
    parser = parser_using_pydantic_classes()
    chain = llm | parser

    #IMPORTANT: when use langchain parsers, you must indicate the output format using parser format instructions AT THE END OF THE PROMPT.
    new_prompt = f'{prompt_text_pdf_analysis} \n\n {parser.get_format_instructions()}'

    response = chain.invoke(new_prompt)

    print('Response of article overview: ')
    print(response)

get_llm_formatted_output()