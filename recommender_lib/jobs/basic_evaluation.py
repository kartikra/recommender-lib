
from recommender_lib.dataprep.movielens import MovieLens
from recommender_lib.training.pipeline import Pipeline

import random
import numpy as np

def load_movielens_data():
    ml = MovieLens(
    ratings_path="/home/kartik/Downloads/ml-latest-small/ratings.csv",
    movies_path="/home/kartik/Downloads/ml-latest-small/movies.csv",
    movie_content_path="",
    )
    print("Loading movie ratings...")
    data_ratings = ml.load_dataset()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_rankings()
    return (data_ratings, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(data_ratings, rankings) = load_movielens_data()

# Construct an Evaluator to, you know, evaluate them
evaluator = Pipeline(data_ratings, rankings,
                     baseline_sim_options_name="pearson",
                     baseline_sim_options_user_based=False,
                     test_size=0.25,
                     no_of_items_dropped=1,
                     training_split_random_state=1)

# Throw in an SVD recommender
evaluator.add_algorithm("SVD", model_random_state=10)

# Throw in an KNN recommender
evaluator.add_algorithm("KNNBasic", sim_options_name="cosine", sim_options_user_based=False)

# Just make random recommendations
evaluator.add_algorithm("Random")

# Fight!
evaluator.run_evaluation(run_top_n=True, no_of_recommended_items=10, verbose=True)
