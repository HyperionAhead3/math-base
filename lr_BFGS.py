'''
将逻辑回归（梯度下降）改造为拟牛顿法（BFGS）
'''
def lr_BFGS(data,alpha):
	n = len(data[0]) - 1
	w0 = np.zeros(n)	#上一个权值
	w = np.zeros(n)		#当前权值
	g0 = np.zeros(n)	#上一个梯度
	g = np.zeros(n)
	C = np.identity(n)	#Hessian阵的逆矩阵，初始化为单位阵(n*1)
	for times in range(1000):
		for d in data:
			x = np.array(d[:-1])	#输入数据(n为列向量)
			y = d[-1]				#期望输出
			C = hessian_r(C, w-w0, g-g0)
			g0 = g					#记录上一次梯度
			w0 = w					#记录上一次权值
			g = (y - pred(w, x))*x	#梯度(n维列向量)
			w = w + alpha * C.dot(g)#梯度下降
		print(times, w)
	return w

def hessian_r(C, w, g):	#此处w和g是变化量
	#BFGS算法近似计算Hessian阵的逆
	n = w.size
	d = g.dot(w)
	if math.fabs(d) < 1e-5:		#分母为0,返回单位阵
		return np.identity(n)
	A = np.zeros((n,n))
	B = np.zeros((n,n))
	w2 = C.dot(w)
	for i in range(n):
		A[i] = w[i] * g
		B[i] = w[i] * w
	A /= d
	At = A.transpose()
	B /= d
	I = np.identity(n)
	return (I-A).dot(C).dot(I-At) + B