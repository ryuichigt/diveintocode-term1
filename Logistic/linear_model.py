class LinearModel():

    def __init__(self,x,y,theta,h):
        self.x = x
        self.y = y
        self.theta = theta
        self.h = h

 	def hx(x,theta):
    	z = np.dot(x,theta.T)
    	return z

	def sigmoid(x,theta):
    	z = hx(x,theta)
    	sig = 1 / (1 + np.exp(-z))
    	return sig

    def compute_cost(self):
    	x = sigmoid(self.x,theta)
    	sl = np.dot(-self.y,np.log(x))
    	sr = np.dot((1-self.y),np.log(1-x))
    	s = 1/len(x)*(sl-sr)
    	r = h/(2*len(x))*sum(theta**2)
    	reslt = s+r
    	return reslt
        

    def gradient_descent(self,iterations = 1000,alpha = 0.01):
        """
        args:
          alpha: Step size/Learning rate
          iterations: No. of iterations(Number of iterations)
        """
    	x = np.array(self.x)
    	y = np.array(self.y)
    	theta = self.theta
    	past_costs = []
	    past_thetas = []
    	J = compute_cost(x,y,theta,alpha)
    	past_costs.append(J)
    	past_thetas.append(theta)
    	for i in range(iterations):
    	    h = sigmoid(x,theta)
    	    theta = theta - alpha *(1/len(y))*(x.T.dot(h-y))
    	    cost = (compute_cost(x,y,theta,alpha))
    	    past_costs.append(cost)
    	    past_thetas.append(theta)
    	    print(cost,theta)

    	return past_thetas,past_costs


    # 確率を求める
    def predict_probs(self):
        pred = sigmoid(self.x,self.theta)
        return pred
    
	# 分類を行う。
    def predict(self, threshold=0.5):
        pred = sigmoid(self.x,self.theta) >= 0.5
        return pred*1
