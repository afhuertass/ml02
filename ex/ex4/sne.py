import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
from  sklearn.metrics.pairwise import euclidean_distances
np.random.seed(41)

import mnist
from  sklearn.decomposition import PCA
#mnist.init()

x_train, y_train, x_test, y_test = mnist.load()



def loss( p , q ):
	# KL divergence
	#print( p.min() )

	#print( q.min() )
	return np.sum(np.where(   p != 0  , p * np.log( ( (p +1 ) / (q+1) )   ), 0 )   )
def softmax( X  ):
	# Apply softmax transformation to etch row 

	expp = np.exp( X )

	np.fill_diagonal( expp , 0.)
	
	p = expp /  expp.sum(1).reshape( [ -1 , 1 ] )
	return  p 


def q_matrix( z ):

	D = euclidean_distances( z , z , squared = True )
	D = -D/2.0
	exp_d = np.exp( D )
	np.fill_diagonal( exp_d , 0.0 )
	#print( np.sum(exp_d))
	return exp_d/ np.sum( exp_d )



def probabilities_from_matrix( X , sigma2 = 1000**2  ):
	# X [ N , dim ]
	# should return NxN matrix with the probabilities

	D = euclidean_distances( X , X , squared = True  )
	X = -D/(2.0*sigma2)
	
	s = softmax( X )
	#print("probasss" ,  s.max() )
	return s

def gradient(   p , q , z   ):

	# Z [ N  , 2 ]
	# pij - Qij 
	# z [ N , 2 ]

	grads = np.zeros( ( len(z) , 2 ))

	for i in range( len(z)):


		zi = z[i] # [ 2 ]
		p_minus_q = p[ i , : ] - q[ i , : ]
		p_minus_q_trans = p[ : , i ] - q[: , i ]
		term = p_minus_q + p_minus_q_trans
		term = term.reshape( (len(z) , 1 ))
		#print( "foka")

		#print( p_minus_q.shape )

		zsum = 2*np.sum( (zi - z)*term , axis = 0  )
		#zsum shape [N0


		#print( zsum.shape )
		grads[i  , : ] = zsum
	 
	#prod_diffs = np.expand_dims( diff1   , 2 ) 

	# Z i - Z j 
	#Zdiff = np.expand_dims( z , 1 ) - np.expand_dims( z , 0 )

	#grad = 4.0*( Zdiff*prod_diffs ).sum( 1 )
	#print( Zdiff )
	return grads



def step_size(eta, tau, grads_squared, n_iters, method="determ"):
    if method == "determ":
        return eta / (1 + eta * tau * n_iters)
    
    if method == 'adagrad':
        return eta / (tau + np.sqrt(grads_squared))

    return 1e-3


def sgd( X , eta= 0.01, tau=.99, method="determ", batch_size=25, zdim = 2 ):


	losses = []
	epoch = 0
	N = len(X)
	z =  np.random.normal( loc = 0.0 , scale = 0.1 , size=[ N , 2 ] ) 
	z_init = z.copy()
	#return z , []
	p = probabilities_from_matrix( X )
	print( p.shape )
	while epoch < 120:
		# each epoch is a full pass, and as minibatch is not trivial 
		for i in range(1):

		# matrix representations 
			#print("epooocs " , epoch)
			q = q_matrix( z  )
			grad = gradient( p , q ,  z )

			z = z - eta*grad 
			if epoch % 100 == 0:
				print(epoch)
			loss_t = loss( p , q)
			losses.append( loss_t )
		epoch += 1 



	return z , z_init


N = 3000
shuffled_inds = np.random.randint( low = 0 , high =  50000 ,  size = N  )
X_shuffled, y_shuffled = x_train[shuffled_inds], y_train[shuffled_inds]

print( np.unique( y_shuffled))
indexes = y_shuffled[ (  ( y_shuffled == 1)  | ( y_shuffled == 8 )  ) | ( y_shuffled == 2 ) ]
X = X_shuffled[ indexes ]
Y = y_shuffled[ indexes ]
print( X.shape )
pca = PCA( 50 )

pca.fit( X )

x_new = pca.transform( X )

z , z_init = sgd( x_new )

#plt.plot( losses )
#plt.show()

plt.scatter( z[: , 0 ] , z[: , 1 ] , c = Y  )
plt.show()
plt.scatter( z_init[: , 0 ] , z_init[: , 1 ] , c = Y  )
plt.show()