Datasets to practice:

1. Student Performance:
https://archive.ics.uci.edu/dataset/320/student+performance

API will have the following functions and structure

main
    - school
        - GET schools                                   DONE
        - GET students_per_school                       DONE
        - GET students_better_performance               DONE
    - student
        - POST new_student                              DONE
        - GET students_by_parent_job                    DONE
        - GET top_students_less_absences                DONE

2. Rocket League Skillshots dataset:
https://archive.ics.uci.edu/dataset/858/rocket+league+skillshots

NEW FEATURES:
- Docker
- SQLite

API will have the following functions and structure:

main
    - 

FAST API basics

1. run FastAPI app
$ fastapi dev main.py

2. common link to access to API docs
http://127.0.0.1:8000/docs#/



DOCKER basics

<before running any command with docker, remember to start Docker Desktop>

1. build image of dockerfile
$ docker build -t fastapi-app .             (fastapi-app is the name of the image)

2. run container
$ docker run -d -p 8000:8000 --name app fastapi-app    (app is the name of the container. fastapi-app is the name of the container)

<STUDY THE DIFFERENCE BETWEEN IMAGES AND CONTAINERS IN DOCKER AND HOW TO WORK WITH THEM>

