import pandas as pd
from typing import TypedDict, Optional, Any

class Movie(TypedDict):
    name: str
    rating: float

def data_loadmap() -> dict[str,list[Movie]]:
    df = pd.read_csv('data/letterboxd_data/ratings.csv')
    #print(df.head())

    df['Date'] = pd.to_datetime(df['Date'],errors='coerce').dt.date.astype(str)
    df = df.dropna(subset=['Date'])

    movies: dict[str,list[Movie]] = {}

    for _, row in df.iterrows():
        date = row['Date']
        name = row['Name']
        rating = row['Rating']

        if date not in movies:
            movies[date] = []
        movies[date].append({'name':name,'rating':rating})

    return movies