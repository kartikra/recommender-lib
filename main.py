from recommender_lib.dataprep.movielens import MovieLens
from recommender_lib.training_job.pipeline_3 import train_recommender, evaluate_recommender

ml = MovieLens(
    ratings_path="/Users/kartik/Downloads/ml-latest-small/ratings.csv",
    movies_path="/Users/kartik/Downloads/ml-latest-small/movies.csv",
    movie_content_path="",
)

print("Loading movie ratings...")
data_ratings = ml.load_dataset()

# Set value of Top-N (N=no of recommendations per user)
N = 10

# Train Model
similarity_svd_model = train_recommender(data_ratings, algorithm="SVD", no_of_recommended_items=N, model_random_state=10,
                                         test_size=0.25, no_of_items_dropped=1, training_split_random_state=1)

# Get Popularity Rankings (needed for evaluating novelty)
print("\nComputing movie popularity ranks so we can measure novelty later...")
rankings = ml.get_popularity_rankings()

# Evaluate Model
evaluate_recommender(data_ratings, similarity_svd_model, no_of_recommended_items=N, rating_threshold=4.0,
                     evaluate_diversity=True, evaluate_novelty=True,
                     baseline_name='pearson_baseline', baseline_user_based=False, 
                     rankings=rankings)
