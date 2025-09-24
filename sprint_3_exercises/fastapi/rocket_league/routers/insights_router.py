import pandas as pd
from fastapi import APIRouter, HTTPException
from read_df import read_data

router = APIRouter()

@router.get("/others/median_distance_wall")
def get_median_distance_wall(ball_Acceleration:float, player_speed:float):
    #HACER
    return 0 