import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df= pd.read_csv(r'C:\Users\aissa\Downloads\projet_streamlit',sep =';')


# get dummies => on sépare les genres en colonnes contenant des booléens

dummies = pd.get_dummies(df, columns=["genres"])
dummies = dummies.drop(columns=['tconst','title','isAdult', 'startYear', 'runtimeMinutes', 'averageRating','numVotes','overview','poster_path','directorsName','actorsName'], axis=1)
data = pd.concat([df , dummies], axis = 1)

"""# 5. ML Nearest Neighbors sur production FR: scale de tous features"""

X = df.select_dtypes(include=['number','boolean'])

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# On crée une instance du modèle Nearest Neighbors, on veut récupérer 10 voisins, on utilise la distance cosine
# on entraine le modèle
modelNN = NearestNeighbors(n_neighbors = 10, metric='cosine')
modelNN.fit(X_scaled)

# fonction pour récupérer les 10 recommendations en fonction du titre choisi
def recommend_movies(movie_title):
    movie_index = df[df["title"] == movie_title].index[0]
    _, indices = modelNN.kneighbors([X_scaled[movie_index]])

    recommended_movies_index = indices[0][1:]
    print("Recommandations pour le film ", movie_title, " :")
    for index in recommended_movies_index:
        print(df["title"][index])
