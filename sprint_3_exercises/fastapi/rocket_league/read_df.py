import pandas as pd

path_file = r".\data\rocket_league_skillshots.data"

def read_data():
    df = pd.read_csv(path_file, delim_whitespace=True)
    
    print(f'columns: {df.columns.tolist()}')
    print(df.head(5))

    return df