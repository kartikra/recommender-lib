from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

from recommender_lib.training.recommender_metrics import RecommenderMetrics
from recommender_lib.training.recommender_model import RecommenderModel


def establish_baseline(data_ratings, baseline_name='pearson_baseline', baseline_user_based=False):

    full_train_set = data_ratings.build_full_trainset()

    # Similarity Algorithm
    similarity_baseline_model = RecommenderModel().build(algorithm="KNNBaseline",
                                                         baseline_name=baseline_name, 
                                                         baseline_user_based=baseline_user_based)
    similarity_baseline_model.fit(full_train_set)
    return similarity_baseline_model


def train_recommender(data_ratings, 
                      algorithm="SVD",
                      no_of_recommended_items=10,
                      model_random_state=10,
                      test_size=0.25,
                      no_of_items_dropped=1,
                      training_split_random_state=1):
    """Train Similarity Recommender using surprise package

    Args:
        data_ratings (_type_): Suprise Dataset with Ratings
        algorithm (str, optional): Select algorithm to be used. Defaults to "SVD".
        no_of_recommended_items (int, optional): no of predictions (top-n). Defaults to 10.
        model_random_state (int, optional): random state for model training. Defaults to 10.
        test_size (float, optional): percent of dataset to be reserved for testing. Defaults to 0.25.
        no_of_items_dropped (int, optional): no of items to drop during leave N out validation. Defaults to 1.
        training_split_random_state (int, optional): random state for train test split and leave 1 out validation process. Defaults to 1.

    Returns:
        _type_: _description_
    """

    print(f"\nBuilding recommendation model using {algorithm}...")
    train_set, test_set = train_test_split(data_ratings, test_size=test_size, random_state=training_split_random_state)

    similarity_model = RecommenderModel().build(algorithm=algorithm, model_random_state=model_random_state)
    similarity_model.fit(train_set)

    print("\nComputing recommendations...")
    predictions = similarity_model.test(test_set)

    print("\nEvaluating accuracy of model...")
    print("RMSE: ", RecommenderMetrics.RMSE(predictions))
    print("MAE: ", RecommenderMetrics.MAE(predictions))

    print(f"\nEvaluating top-{no_of_recommended_items} recommendations...")

    # Set aside one rating per user for testing
    LOOCV = LeaveOneOut(n_splits=no_of_items_dropped, random_state=training_split_random_state)

    for train_set, test_set in LOOCV.split(data_ratings):
        print("Computing recommendations with leave-one-out...")

        # Train model without left-out ratings
        similarity_model.fit(train_set)

        # Predicts ratings for left-out ratings only
        print("Predict ratings for left-out set...")
        left_out_predictions = similarity_model.test(test_set)

        # Build predictions for all ratings not in the training set
        print("Predict all missing ratings...")
        bigtest_set = train_set.build_anti_testset()
        all_predictions = similarity_model.test(bigtest_set)

        # Compute top 10 recs for each user
        print(f"Compute top {no_of_recommended_items} recs per user...")
        top_n_predicted = RecommenderMetrics.top_n(all_predictions, n=no_of_recommended_items)

        # See how often we recommended a movie the user actually rated
        print("\nHit Rate: ", RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions))
        # Break down hit rate by rating value
        print("\nrHR (Hit Rate by Rating value): ")
        RecommenderMetrics.rating_hit_rate(top_n_predicted, left_out_predictions)
        # See how often we recommended a movie the user actually liked
        print("\ncHR (Cumulative Hit Rate, rating >= 4): ", RecommenderMetrics.cumulative_hit_rate(top_n_predicted, left_out_predictions, 4.0))
        # Compute ARHR
        print("\nARHR (Average Reciprocal Hit Rank): ", RecommenderMetrics.avg_reciprocal_hit_rate(top_n_predicted, left_out_predictions))

    return similarity_model


def evaluate_recommender(data_ratings, 
                         similarity_model,
                         no_of_recommended_items=10,
                         rating_threshold=4.0,
                         evaluate_diversity=True,
                         evaluate_novelty=True,
                         baseline_name='pearson_baseline', 
                         baseline_user_based=False,
                         rankings=None):

    full_train_set = data_ratings.build_full_trainset()
    # Computing item similarities so we can measure diversity later if needed
    if evaluate_diversity:
        print("\nComputing item similarities so we can measure diversity later...")
        sim_options = {'name': baseline_name, 'user_based': baseline_user_based}
        print("similarity options: " + str(sim_options) + "\n")
        baseline_model = KNNBaseline(sim_options=sim_options)
        baseline_model.fit(full_train_set)

    # Evaluate trained model
    print("\nComputing complete recommendations, no hold outs...")
    similarity_model.fit(full_train_set)
    bigtest_set = full_train_set.build_anti_testset()
    all_predictions = similarity_model.test(bigtest_set)
    top_n_predicted = RecommenderMetrics.top_n(all_predictions, n=no_of_recommended_items)

    # Print user coverage with a minimum predicted rating of 4.0:
    print("\nUser coverage: ", RecommenderMetrics.user_coverage(top_n_predicted, full_train_set.n_users, rating_threshold=rating_threshold))

    # Measure novelty (average popularity rank of recommendations):
    if evaluate_novelty and rankings is not None:
        print("\nNovelty (average popularity rank): ", RecommenderMetrics.novelty(top_n_predicted, rankings))

    # Measure diversity of recommendations:
    if evaluate_diversity:
        print("\nDiversity: ", RecommenderMetrics.diversity(top_n_predicted, baseline_model))

    return
