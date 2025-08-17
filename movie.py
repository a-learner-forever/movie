# Simple AI Movie Recommendation System
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Sample movie dataset
movies = {
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "The Prestige",
        "Avengers: Endgame",
        "Iron Man",
        "The Lion King",
        "Frozen",
        "Toy Story",
        "Finding Nemo"
    ],
    "description": [
        "Batman faces the Joker in Gotham City",
        "A thief steals corporate secrets through dream-sharing",
        "Explorers travel through a wormhole in space",
        "Two magicians compete with dangerous tricks",
        "Superheroes unite to defeat Thanos",
        "A billionaire builds a powerful iron suit",
        "A lion cub becomes king of the jungle",
        "Two sisters face ice powers and magic",
        "Toys come to life when humans are not around",
        "A clownfish searches for his missing son"
    ]
}

df = pd.DataFrame(movies)

# 2. Convert descriptions to vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])

# 3. Compute similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Function to recommend movies
def recommend(movie_title):
    if movie_title not in df["title"].values:
        return ["Movie not found!"]
    
    idx = df[df["title"] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [df.iloc[i[0]]["title"] for i in scores[1:6]]
    return recommendations

# 5. Example usage
movie = "Inception"
print(f"Movies similar to '{movie}':")
for m in recommend(movie):
    print("👉", m)
