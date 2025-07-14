Link para acceder a un portal que contiene una lista llena de APIs para practicar. Se llama RapidAPI

https://rapidapi.com/hub

**Voy a seleccionar los siguientes para practicar**

- Fresh Linkedin Profile Data
https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/playground

- Real Time Web Search
https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-web-search

# Key tools

1. use ***dotenv library*** to save secrets.\
    - import os ; ***from dotenv import load_dotenv*** ; invoke ***'load_dotenv()'*** function at the beggining of the script.
    - function --> ***os.getenv('YOUR KEY')***

2. handle exceptions using **try** and **except**. Important functions:
    - ***response.raise_for_status()*** --> to raise an exception if status code is not 2XX.
    - ***requests.exceptions.RequestException*** class --> in case a exception is raised.
    - ***set timeout to requests*** --> example: requests.get(url='your_url', timeout= 10) (timeout in seconds)

3. Log and monitor requests using **logging** module.
    - **import logging** and then setup your logging environment.
    - in the script 'fresh_linkedin_profile_data.py', you can find a small example of it.
    - keep in mind the ***logging level*** in the setup (INFO for general logs, DEBUG for dev log purposes)
    - *IMPORTANT*: There is a way to reuse the logic and change the format of your logs (using a class, for example)
    - You can also save all your logs in files.

4. Handle JSON responses
    - keys() --> to get all keys you can access.
    - pandas library --> to work with large and structured data (columns, rows, and so on)

# TODO
1. Next function to do: 'get-company-by-url' (GET) and 'search-jobs' (POST)
2. Next API -> Real Time Web Search
    - Search (advanced) (GET)
    - Batch Search (light) (POST)
    - IMPORTANT: use the same best practices as the first script.
    