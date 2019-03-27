import pandas as pd 
import numpy as np 

import math
from sklearn.model_selection import train_test_split

import  matplotlib.pyplot as plt 

def lr_decay( t , tau = 0.5 , ini = 0.0005):
	

	return  ini/(1 +  ini*tau*t )
def lr_agrad(   past_gradients , tau =  0.99 , ini = 1.5 ):
	
	return  ini/( tau+ np.sqrt( past_gradients ))

def loss_f( preds , actual  ):
	m = len( preds )
	return ( 1.0/m)*np.sum( ( preds - actual)**2 , axis  = 0 )


def sgd( thetas , X , Y , X_test , Y_test  ,  iterations = 20000 , learning_rate = 0.001 , batch_size = 100 , agrad = False  ):
	# runs a step of the gradient descent 
	# X is vector of data 
	L = X.shape[0]
	
	losses  = {}
	losses["train"] = []
	losses["test"] = []
	losses["iterations"] = []

	grads_accum = np.zeros( ( X.shape[1] , 1 ) ) # one gradient for each dimension
	for i in range(iterations ):
		loss = 0.0 

		random_index = np.random.randint( 0 , L , batch_size )
		
		x_batch = X[ random_index , :  ]
		y_batch = Y[random_index , : ]

		y_pred = np.dot( x_batch , thetas ).reshape( ( -1 , 1 ))

		Y_test_preds = np.dot( X_test , thetas ).reshape( ( -1 , 1 ))
		#print( y_batch.shape )
		#print( y_pred.shape )

		grads = (2.0/batch_size)*x_batch.T.dot(  y_pred - y_batch )
		#print("asdasdas")
		#print( grads.shape    )
		grads_accum += grads**2

		if agrad:

			learning_rate = lr_agrad( grads_accum )
		else:
			learning_rate = lr_decay( i )

		thetas = thetas - learning_rate*grads
		
		if i% 100 == 0 :
			#Y_train_preds = np.dot( X , thetas ).reshape( (-1 , 1 ) )

			loss_train = loss_f( y_pred , y_batch )
			loss_test = loss_f( Y_test_preds   , Y_test  )
			#print( loss_train )
			#print( loss_test )
			losses["train"].append( loss_train )
			losses["test"].append( loss_test )
			losses["iterations"].append( i )

	return losses

def prepare_data( split = 0.2 ):

	df = pd.read_csv("./winequality-white.csv" , sep = ";")

	Y = df["quality"].values.reshape( (-1 , 1 ))
	df = df.drop( ["quality"] , axis = 1 )
	X = df.values 

	X = (X - X.min()) / ( X.max() - X.min() )

	x_train , x_test , y_train , y_test  = train_test_split(  X , Y , test_size = split )
	

	return x_train , y_train , x_test , y_test 

def main():

	x_train , y_train , x_test , y_test = prepare_data()
	print( y_train.shape )
	theta = np.random.rand( ( x_train.shape[1] )  ).reshape( ( -1 , 1 ))
	print( theta.shape )
	batch_sizes = [ 1 , 10 , 100 , 1000]
	for i in batch_sizes:
		losses = sgd( theta , x_train , y_train , x_test , y_test  , batch_size = i  )
		losses_agrad = sgd( theta , x_train , y_train , x_test , y_test , batch_size = i , agrad = True  )


		plt.plot( losses["iterations"] , losses["train"] )
		plt.plot( losses["iterations"] , losses["test"]  )
		plt.legend( ['Train loss - minibatch {}'.format( i ) , "Test loss"])
		plt.title( "Decay learning rate ")
		plt.savefig("./decay_batch-{}.png".format( i ) )
		plt.clf()


		plt.plot( losses_agrad["iterations"] , losses_agrad["train"] )
		plt.plot( losses_agrad["iterations"] , losses_agrad["test"]  )
		plt.legend( ['Train loss - minibatch {}'.format( i ) , "Test loss"])
		plt.title( "Agrad learning rate ")
		plt.savefig("./agrad_batch-{}.png".format( i ) )
		plt.clf()




if __name__=='__main__':
	#prepare_data()
	main()