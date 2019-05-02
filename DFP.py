def hessian_r2(C, w, g):	#此处w和g是变化量
	#DFP算法近似计算Hessian阵的逆
	n = w.size
	d = g.dot(w)
	if math.fabs(d) < 1e-4:
		return np.identity(n)
	A = np.zeros((n,n))
	for i in range(n):
		A[i] = g[i] * g
	A /= d

	B = np.zeros((n,n))
	w2 = C.dot(w)
	for i in range(n):
		B[i] = w2[i] * w2
	d2 = w.dot(w2)
	if math.fabs(d2) < 1e-4:
		return np.identity(n)
	B /=d2
	return C + A - B