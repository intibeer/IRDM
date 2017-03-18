class LearningMethod:
    def __init__(self, eta, weights):
        """
        Base class for Learning Methods
        INPUT:
            eta - double
                learning parameter
            weights - array of doubles
                weights of parameters which will be adjusted as the methods 
                learn from the data
        """
        self.eta = eta
        self.weights = weights
        
    def update_weight(self, weight_index, increment):
        """
        Method to update a single weight value
        INPUT:
            weight_index - int
                index of the parameter whose weight is to be adjusted
            increment - double
                amount by which the weight is to be adjusted
        """
        self.weights[weight_index] += self.eta*increment