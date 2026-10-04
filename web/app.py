import sys
import os

# Add project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify, render_template
import pandas as pd

from src.collaborative_filtering import train_svd_model
from src.topn_recommendation import get_top_n

app = Flask(__name__)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

DATA_DIR = os.path.join(BASE_DIR, "data")

ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")

ratings = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)

# ---------------------------------------------------------
# Movie information
# ---------------------------------------------------------

movie_id_to_title = dict(
    zip(movies_df["movieId"], movies_df["title"])
)

movie_id_to_genres = dict(
    zip(movies_df["movieId"], movies_df["genres"])
)

# ---------------------------------------------------------
# Train SVD model
# ---------------------------------------------------------

print("Training SVD recommendation model...")
model, predictions = train_svd_model(ratings)

print("SVD model trained successfully.")

# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# Recommendation API
# ---------------------------------------------------------

@app.route("/api/recommend", methods=["GET"])
def api_recommend():

    try:
        user_id = int(request.args.get("userId", 1))
        n = int(request.args.get("n", 10))

        # Limit recommendations
        n = max(1, min(n, 20))

        # Generate Top-N recommendations
        top_n = get_top_n(predictions, n=n)

        recommendations = top_n.get(user_id, [])

        result = []

        for movie_id, score in recommendations:

            title = movie_id_to_title.get(
                movie_id,
                f"Movie {movie_id}"
            )

            genres = movie_id_to_genres.get(
                movie_id,
                "Unknown"
            )

            # Convert pipe-separated genres
            if genres and genres != "(no genres listed)":
                genres = genres.replace("|", " • ")
            else:
                genres = "Movie"

            result.append({
                "movieId": int(movie_id),
                "movieTitle": title,
                "genres": genres,
                "score": round(float(score), 2)
            })

        # -------------------------------------------------
        # User statistics
        # -------------------------------------------------

        user_ratings = ratings[
            ratings["userId"] == user_id
        ]

        if len(user_ratings) > 0:

            movies_rated = len(user_ratings)

            average_rating = round(
                user_ratings["rating"].mean(),
                2
            )

        else:

            movies_rated = 0
            average_rating = 0

        return jsonify({

            "success": True,

            "userId": user_id,

            "moviesRated": movies_rated,

            "averageRating": average_rating,

            "recommendationCount": len(result),

            "recommendations": result

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
            "recommendations": []
        }), 400


# ---------------------------------------------------------
# Run Flask
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        debug=True,
        host="0.0.0.0",
        port=port
    )