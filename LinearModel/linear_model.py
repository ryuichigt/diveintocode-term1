import numpy as np
import scipy as sc
import pandas as  pd
from sklearn.preprocessing import StandardScaler

class LinearModel():

    def __init__(self,train, target, theta, iterations, alpha):
        self.train = train
        self.target = target
        self.theta = theta
        self.iterations = iterations
        self.alpha = alpha
    
#     def compute_cost(self):
#         h = np.dot(self.train.T,theta)
#         j = sum((h - self.target)**2)/(len(self.target)*2)
#         return j
    def compute_cost(train,test,theta):
        h = np.dot(train.T,theta)
        j = sum((h - target)**2)/(len(target)*2)
        return j

    def gradient_descent(self):
        
        """
        args:
        alpha: Step size/Learning rate
         iterations: No. of iterations(Number of iterations)
        """
        past_costs = []
        past_thetas = []

        J = compute_cost(self.train,self.target,self.theta)
        #J = self.compute_cost()
        past_costs.append(J)
        past_thetas.append(self.theta)
    
        for i in range(self.iterations):
            h = np.dot(self.train.T, self.theta)
            self.theta = self.theta - (self.alpha/len(self.target)) * np.dot(self.train, h - self.target)
            J = compute_cost(self.train,self.target,self.theta)
            past_costs.append(J)
            past_thetas.append(self.theta)
    
        self.cost = past_costs
        self.theta = past_thetas
        return past_costs,past_thetas
        
    def plot_learning_curve(self):
        cost,theta = lm.gradient_descent()
        return pd.DataFrame(cost).plot.line()

