import pandas as pd
import os

# get the dataset's data path
RAW_DATA_PATH = "data/raw"

# loads KB dataset
def load_kenpom_barttorvik():
    return pd.read_csv(os.path.join(RAW_DATA_PATH, "kenpom_barttorvik.csv"))

# loads resumes dataset
def load_resumes():
    return pd.read_csv(os.path.join(RAW_DATA_PATH, "resumes.csv"))

# loads TR dataset
def load_team_rankings():
    return pd.read_csv(os.path.join(RAW_DATA_PATH, "team_rankings.csv"))

# loads TM dataset
def load_team_matchups():
    return pd.read_csv(os.path.join(RAW_DATA_PATH, "tournament_matchups.csv"))

# loads all the datasets (returns a dictionary of all raw datasets)
def load_all_datasets():
    return {
        "kenpom_barttorvik": load_kenpom_barttorvik(),
        "resumes": load_resumes(),
        "team_rankings": load_team_rankings(),
        "team_matchups": load_team_matchups()
    }