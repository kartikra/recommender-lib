from recommender_lib.training.recommender_metrics import RecommenderMetrics


class EvaluatedAlgorithm:
    
    def __init__(self, algorithm_name, initial_model):
        self.algorithm_name = algorithm_name
        self.recommender_model = initial_model
        
    def evaluate(self, evaluation_data, run_top_n, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if verbose:
            print("Evaluating accuracy...")
        self.recommender_model.fit(evaluation_data.get_training_set())
        predictions = self.recommender_model.test(evaluation_data.get_test_set())
        metrics["RMSE"] = RecommenderMetrics.root_mean_square_error(predictions)
        metrics["MAE"] = RecommenderMetrics.mean_absolute_error(predictions)
        
        if run_top_n:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.recommender_model.fit(evaluation_data.get_leave_n_out_training_set())
            left_out_predictions = self.recommender_model.test(evaluation_data.get_leave_n_out_test_set())
            # Build predictions for all ratings not in the training set
            all_predictions = self.recommender_model.test(evaluation_data.get_leave_n_out_anti_test_set())
            # Compute top 10 recs for each user
            top_n_predicted = RecommenderMetrics.top_n(all_predictions, n)
            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions)
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.cumulative_hit_rate(top_n_predicted, left_out_predictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.avg_reciprocal_hit_rate(top_n_predicted, left_out_predictions)
        
            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.recommender_model.fit(evaluation_data.get_full_training_set())
            all_predictions = self.recommender_model.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = RecommenderMetrics.top_n(all_predictions, n)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.user_coverage(top_n_predicted,
                                                                   evaluation_data.get_full_training_set().n_users,
                                                                   rating_threshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.diversity(top_n_predicted, evaluation_data.get_similarities())
            
            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.novelty(top_n_predicted,
                                                            evaluation_data.get_popularity_rankings())
        
        if verbose:
            print("Analysis complete.")
    
        return metrics
    
    def get_algorithm_name(self):
        return self.algorithm_name
    
    def get_recommender_model(self):
        return self.recommender_model
