class svm(w,x,y,b):
	"""docstring for ClassName"""
	def __init__(self):
		self.w = w
		self.x = x
		self.y = y
		self.b = b
	def sign(self):
		sig = np.dot(self.x,self.w.T) + self.b
		return sig
    
	def f(self):
		sig = np.dot(self.x,self,w.T) + self.b

		y = []
		for i in sig:
			if i.sum() >= 0:
				y.append(1)
			elif i.sum() <0:
				y.append(-1)
    
		return np.array([y])

	def gradient_descent(self,reg = 0.1):
    	#重みベクトルを０として初期化する
		dW = np.zeros(self.w.shape) 
    
    	#クラス数と特徴数分回す
		num_classes = self.w.shape[1]
		num_train = self.x.shape[0]
		loss = 0.0
    	#データ数分回す
    	#margin = loss(X,y,W)
    
		for i in range(num_train):
			scores = self.x[i,:].dot(self.w)
			correct_class_score = scores[self.y[i]]
        	#クラス分回す
			for j in range(num_classes):
				if j == self.y[i]:
					continue
				margin = scores[j] - correct_class_score + 1 
				if margin > 0:
					loss += margin
					dW[:,y[i]] -= self.x[i,:] 
					dW[:,j] += self.x[i,:]

    	# すべての例を平均化する
		loss /= num_train
		dW /= num_train

	    # 正則化を加える
		loss += 0.5 * reg * np.sum(self.w * self.w)
		dW += reg*self.w
    	# 損失関数と最適化したベクトル
		return loss, dW
