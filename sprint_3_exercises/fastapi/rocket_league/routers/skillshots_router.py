from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter()
path_dataset_file = r"./data/rocket_league_skillshots.data"

# columns = ['BallAcceleration', 'Time', 'DistanceWall', 'DistanceCeil', 'DistanceBall', 'PlayerSpeed', 'BallSpeed', 'up', 'accelerate', 'slow', 'goal', 'left', 'boost', 'camera', 'down', 'right', 'slide', 'jump']

@router.get("/skillshot/get_average_player_speed")
def get_average_player_speed():
    df = pd.read_csv(filepath_or_buffer=path_dataset_file, delim_whitespace=True)

    average_player_speed = df['PlayerSpeed'].mean()

    return {
        "message": f"The average player speed [{len(df)} players] is [{average_player_speed}]"
    }
