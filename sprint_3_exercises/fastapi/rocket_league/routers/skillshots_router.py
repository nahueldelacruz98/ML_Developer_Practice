from fastapi import APIRouter, HTTPException
import pandas as pd
from read_df import read_data


router = APIRouter()
path_dataset_file = r"./data/rocket_league_skillshots.data"

# columns = ['BallAcceleration', 'Time', 'DistanceWall', 'DistanceCeil', 'DistanceBall', 'PlayerSpeed', 'BallSpeed', 'up', 'accelerate', 'slow', 'goal', 'left', 'boost', 'camera', 'down', 'right', 'slide', 'jump']

@router.get("/get_average_player_speed")
def get_average_player_speed():
    df = read_data()

    average_player_speed = df['PlayerSpeed'].mean()

    return {
        "message": f"The average player speed [{len(df)} players] is [{average_player_speed}]"
    }

@router.get("/median_ball_acceleration_by_distance")
def get_median_ball_acceleration_by_distance(distance_wall:float, distance_ball:float, player_speed:float):
    df = read_data()

    #filter by distance_wall, distance_ball and player_speed

    df_filtered = df[(df['DistanceWall'] >= distance_wall) & (df['DistanceBall'] >= distance_ball) & (df['PlayerSpeed'] >= player_speed)]

    if len(df_filtered) == 0:
        print(f"There are no results with the following Distance Wall [{distance_wall}], Distance Ball [{distance_ball}] and Player speed [{player_speed}]")
        raise HTTPException(status_code=400, detail=f"There are no results with the following Distance Wall [{distance_wall}], Distance Ball [{distance_ball}] and Player speed [{player_speed}]")
    
    # Get median ball acceleration

    median_ball_acceleration = df_filtered['BallAcceleration'].mean()

    return {

        'median' : abs(median_ball_acceleration),
        'results_considered': len(df_filtered)
    }