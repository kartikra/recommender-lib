from surprise import SVD
from surprise import KNNBaseline



class RecommenderModel:

    def __init__(self):
        return

    def build(self, algorithm, **kwargs):

        if algorithm == "SVD":
            return SVD(random_state=kwargs.get("model_random_state", 10))
        elif algorithm == "KNNBaseline":
            # KNNBaseline option creates a baseline
            sim_options = dict()
            sim_options["name"] = kwargs.get("baseline_name", "pearson_baseline")
            sim_options["user_based"] = kwargs.get("baseline_user_based", False)
            return KNNBaseline(sim_options)
