import pandas as pd
import tmdbsimple as tmdb
from app.api_keys.tmdb_api import API_KEY

tmdb.API_KEY = API_KEY


def data_loadmap():
    df = pd.read_csv('data/ratings.csv')
    print(df.head())
    
data_loadmap()