import requests
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import pandas

load_dotenv()   # Load environment variables from .env file

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, # for debugging, use logging.DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

    return logging.getLogger(__name__)

log = setup_logging()

def get_profile_posts(url:str):
    # IMPORTANT: only works with public user profiles, not company profiles.
    
    log.info("Getting profile posts from Linkedin")

    resource = "get-profile-posts"
    linkedin_url = quote_plus(url)
    query_string = f"linkedin_url={linkedin_url}"

    uri = os.getenv("URI_API_FRESH_LINKEDIN_PROFILE_DATA") + resource + "?" + query_string

    log.info(f'API request uri: {uri}')

    headers = {
        "x-rapidapi-host" : os.getenv("RAPIDAPI_HOST"),
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY")
    }

    try:
        response = requests.get(url= uri, headers = headers)
        response.raise_for_status() # Raise an error for bad responses
        list_posts = response.json()['data']
        #list_posts = response.json()['data'][0]['hello']    #to throw an error
        log.info(f'Amount of posts found: {len(list_posts)} ; Keys for each post: {list_posts[0].keys()}')

        # Example using pandas to manage posts.
        df_posts = pandas.DataFrame(list_posts)
        log.info(f'Overview of posts using pandas: {df_posts.describe()}')

    except requests.exceptions.RequestException as ex:
        log.error(f"Request failed: {ex}")
        return 
    except Exception as ex:
        log.error(f"An error occurred: {ex}")
        return

def get_company_by_url(url_company:str):
    
    resource = "get-company-by-linkedinurl"
    params = {
        "linkedin_url": url_company
    }

    headers = {
        "x-rapidapi-key" : os.getenv("RAPIDAPI_KEY"),
        "x-rapidapi-host": os.getenv("RAPIDAPI_HOST")
    }

    uri = os.getenv("URI_API_FRESH_LINKEDIN_PROFILE_DATA") + resource

    log.info(f"Uri to get company by url: {uri}")
    
    try:
        response = requests.get(url = uri, headers = headers, params = params, timeout = 10)
        response.raise_for_status()
        log.info("Company data was found")

    except requests.exceptions.RequestException as ex:
        log.error(f"Request failed: {ex}")
        return

    log.info(f"Company data keys: {response.json()['data'].keys()}")

#get_profile_posts(os.getenv("LINKEDIN_PROFILE_URL"))
get_company_by_url(os.getenv("COMPANY_LINKEDIN_URL"))