from recommender_lib.training.evaluation_data import EvaluationData
from recommender_lib.training.evaluated_algorithm import EvaluatedAlgorithm
from recommender_lib.model.surprise_model import SurpriseModel


class Pipeline:
    
    list_recommender_models = []
    
    def __init__(self, dataset, 
                 rankings,
                 baseline_sim_options_name="pearson",
                 baseline_sim_options_user_based=False,
                 test_size=0.25,
                 no_of_items_dropped=1,
                 training_split_random_state=1):

        sim_options = {'name': baseline_sim_options_name, 'user_based': baseline_sim_options_user_based}
        evaluation_data = EvaluationData(dataset, rankings, sim_options,
                                         test_size=test_size, 
                                         no_of_items_dropped=no_of_items_dropped, 
                                         training_split_random_state=training_split_random_state)
        self.dataset = evaluation_data
        
    def add_algorithm(self, algorithm_name, **kwargs):
        initial_model = SurpriseModel().build(algorithm_name, **kwargs)
        recommender_model = EvaluatedAlgorithm(algorithm_name, initial_model)
        self.list_recommender_models.append(recommender_model)
        
    def run_evaluation(self, run_top_n, no_of_recommended_items=10, verbose=True):
        results = {}
        for recommender_model in self.list_recommender_models:
            print("Evaluating ", recommender_model.get_algorithm_name(), "...")
            results[recommender_model.get_algorithm_name()] = \
                recommender_model.evaluate(self.dataset, run_top_n, 
                                           n=no_of_recommended_items,
                                           verbose=verbose)

        # Print results
        print("\n")
        
        if run_top_n:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if run_top_n:
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
        
    def sample_top_n_recs(self, ml, test_subject=85, k=10):
        
        for recommender_model in self.list_recommender_models:
            print("\nUsing recommender ", recommender_model.get_algorithm_name())
            
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.get_full_training_set()
            recommender_model.get_algorithm().fit(trainSet)
            
            print("Computing recommendations...")
            testSet = self.dataset.get_anti_test_set_for_user(test_subject)
        
            predictions = recommender_model.get_algorithm().test(testSet)
            
            recommendations = []
            
            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
