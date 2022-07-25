from surprise import NormalPredictor
# KNN Based
from surprise import KNNBaseline
from surprise import KNNBasic
# Matrix Facorization Based
from surprise import SVD
from surprise import SVDpp
from surprise import NMF


class SurpriseModel:
    """Suprise Package provides number of models
    Matrix Factorzation: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
    KNN Based: https://surprise.readthedocs.io/en/stable/knn_inspired.html
    """

    def __init__(self):
        return

    def build(self, algorithm, **kwargs):

        # https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#similarity-measures-configuration
        sim_options = {
            "name": kwargs.get("sim_options_name", "MSD"),
            "user_based": kwargs.get("sim_options_user_based", False),
        }
        if "sim_options_min_support" in kwargs:
            sim_options["min_support"] = kwargs.get("sim_options_min_support", 0)
        if "sim_options_shrinkage" in kwargs:
            sim_options["shrinkage"] = kwargs.get("sim_options_shrinkage", 100)
        
        if algorithm == "KNNBaseline":
            return KNNBaseline(sim_options=sim_options)
        elif algorithm == "KNNBasic":
            return KNNBasic(sim_options=sim_options)
        elif algorithm == "SVD":
            return SVD(random_state=kwargs.get("model_random_state", 10))
        elif algorithm == "SVDpp":
            return SVDpp(random_state=kwargs.get("model_random_state", 10))
        elif algorithm == "NMF":
            return NMF(random_state=kwargs.get("model_random_state", 10))
        elif algorithm == "Random":
            return NormalPredictor()
        else:
            raise ValueError (f"Unknown algorithm name {algorithm}")
