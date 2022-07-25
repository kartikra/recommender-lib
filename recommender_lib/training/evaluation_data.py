from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline


class EvaluationData:
    
    def __init__(self, data, 
                 popularity_rankings,
                 sim_options={},
                 test_size=0.25,
                 no_of_items_dropped=1,
                 training_split_random_state=1):
        
        self.rankings = popularity_rankings
        
        # Build a full training set for evaluating overall properties
        self.full_training_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_training_set.build_anti_testset()
        
        # Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=test_size, 
                                                       random_state=training_split_random_state)
        
        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        leave_n_out_cv = LeaveOneOut(n_splits=no_of_items_dropped, 
                                     random_state=training_split_random_state)
        for train, test in leave_n_out_cv.split(data):
            self.leave_n_out_train = train
            self.leave_n_out_test = test
            
        self.leave_n_out_anti_test = self.leave_n_out_train.build_anti_testset()
        
        # Compute similarity matrix between items so we can measure diversity
        self.baseline_similarity_model = KNNBaseline(sim_options=sim_options)
        self.baseline_similarity_model.fit(self.full_training_set)
            
    def get_full_training_set(self):
        return self.full_training_set
    
    def get_full_anti_test_set(self):
        return self.full_anti_test_set
    
    def get_anti_test_set_for_user(self, test_subject):
        training_set = self.full_training_set
        fill = training_set.global_mean
        anti_test_set = []
        u = training_set.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in training_set.ur[u]])
        anti_test_set += [(training_set.to_raw_uid(u), training_set.to_raw_iid(i), fill) for
                         i in training_set.all_items() if
                         i not in user_items]
        return anti_test_set

    def get_training_set(self):
        return self.trainSet
    
    def get_test_set(self):
        return self.testSet
    
    def get_leave_n_out_training_set(self):
        return self.leave_n_out_train
    
    def get_leave_n_out_test_set(self):
        return self.leave_n_out_test
    
    def get_leave_n_out_anti_test_set(self):
        return self.leave_n_out_anti_test
    
    def get_similarities(self):
        return self.baseline_similarity_model
    
    def get_popularity_rankings(self):
        return self.rankings
