import logging
import os
from dotenv import load_dotenv
import requests

def setup_logging():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger(__name__)

log = setup_logging()
load_dotenv()

#Now write your code here

def search_advanced(query:str, number_results:int):

    params = {
        "q": query,     # Search query
        "num": number_results
    }

    header = {
        "x-rapiapi-host": os.getenv("RAPIDAPI_HOST"),
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY")
    }

    uri = os.getenv("URI_REAL_TIME_WEB_SEARCH") + "search-advanced-v2"

    try:
        response = requests.get(uri, params = params, headers = header, timeout = 20)
        response.raise_for_status()
    except requests.exceptions.RequestException as ex:
        log.error(f"An exception occurred while requesting the API: {ex}")
        return

    list_results = response.json()["data"]["organic_results"]

    for index, result in enumerate(list_results, start=1):
        title = result["title"]
        link = result["url"]
        source = result["source"]

        log.info(f"Result number {index}: title: {title}; link: {link}; source: {source}")
    
    log.info("End of results")

def batch_search(list_queries:list, number_results:int):

    resource = "search"
    body = {
        "queries": list_queries,    # a query is a string that you can input in your browser 
        "limit": number_results     #number of results per query search
    }

    header = {
        "x-rapiapi-host": os.getenv("RAPIDAPI_HOST"),
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY")
    }

    uri = os.getenv("URI_REAL_TIME_WEB_SEARCH") + resource

    try:
        response = requests.post(url = uri, headers = header, json=body, timeout = 20)
        response.raise_for_status()
    except requests.exceptions.RequestException as ex:
        log.error(f"An exception occurred while requesting a batch of queries: {ex}")
        return
    except requests.exceptions.Timeout as ex:
        log.error("The request of batch queries has timed out.")
        return
    
    list_queries_results = response.json()["data"]

    for query_result in list_queries_results:
        query_name = query_result["query"]
        batch_results = query_result["results"]
        log.info(f"The following query '{query_name}' has returned {len(batch_results)} results:")

        for index, result in enumerate(batch_results, start=1):
            title = result["title"]
            url = result["url"]

            log.info(f'[{index}] title: {title}; link: {url}')
    
    log.info("End of query results")

#search_advanced("how to build a website",5)
batch_search(["Machine Learning", "how to work with APIs in Python","AWS services and costs"], 3)