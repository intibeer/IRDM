from ..Features import cost_gradient, cost, Features
from learningmethod import LearningMethod

class LambdaRank(LearningMethod):
    def __init__(self, eta):
        """
        LambdaRank method
        INPUT:
            eta - double
                The learning parameter for the method
        """
        LearningMethod.__init__(eta)