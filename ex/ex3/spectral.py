import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


import itertools


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def read_data(  path ):

	df = pd.read_csv(path , header = None )

	return df

def get_degree( M ):
	# M [ n , n ]

	 degrees = M.sum( axis = 1 )

	 R = np.diag( degrees )
	 return R


def plot_vectors( evectors ):

	evectors = evectors.T
	# 
	for i in range( evectors.shape[0] ):

		vec = evectors[ i , : ]

		plt.plot(  range( vec.shape[0] ) , vec  )
		plt.savefig("./eigenvec_{}.png".format( i ) )
		plt.clf()


	return 

def do_thing( L  , k = 4 , prefix = "" ):

	eigenvalues, eigenvectors = np.linalg.eig( L )

	print( eigenvectors.shape )
	k_smallest = np.argsort( eigenvalues )[1:k+1]
	print( "Smallest eigenvalues ")
	print( eigenvalues [k_smallest].min() )
	#eigenvalues[ np.argsort( eigenvalues )[k:] ]

	#print( eigenvalues )

	smallest_vectors = eigenvectors [  : , k_smallest  ]
	plot_vectors( smallest_vectors )

	i = 0 
	j = 1
	# Print all combinations of eigenvectors 
	combs = list( itertools.combinations( range( smallest_vectors.shape[ 1 ] ) , 2) )

	for  i , j in combs:
		name = "{}_i_{}_j_{}.png".format( prefix, i,j)
		plt.scatter( smallest_vectors[ : ,  i ] , smallest_vectors[: , j ]  )
		plt.savefig("{}".format(name))
		plt.clf()
	print(combs)

	return smallest_vectors 


def main( epsilon = 0.5  , A = 8):

	A = A +1 
	df = read_data("./exercise3data.csv")

	# get euclidean distance matrix 
	D = euclidean_distances( df.values , df.values )
	print( D.shape )
	print( D[:5 , :5 ].shape  )
	# Plotting data
	plt.scatter( df[0] , df[1] )
	plt.xlabel( "X")
	plt.ylabel("Y")
	plt.savefig( "./plotdata.png")
	plt.clf()
	#plt.show()

	# first adjacency matrix 
	W1 =  np.zeros_like( D )
	W1[ D <= epsilon ] = 1.0 
	np.fill_diagonal( W1 , 0.0 )

	W2 = np.zeros_like( D )

	for i in range( D.shape[0] ):
		distances = D[ i , : ]
		indexs = np.argsort( distances )[:A]
		indexs = list( indexs )
		indexs.remove( i )
		
		W2[ i , indexs ] = 1.0
		W2[ indexs , i ] = 1.0 # for the symetry 

	D1 = get_degree( W1 )
	D2 = get_degree( W2 )

	# Build laplacian matrix
	L1  = D1 - W1
	L2 = D2 - W2
	# the 
	newX1 = do_thing(L1 , prefix = "L1")
	newX2 = do_thing(L2 , prefix = "L2")


	thing( df , newX1 , newX2 , prefix1 = "e_{}".format( epsilon ) , prefix2 = "A_{}".format( A ) )

def thing( df , newX1 , newX2 , prefix1 , prefix2  ):
	kmeans = KMeans(n_clusters=2, random_state=0 , init = "k-means++")
	kmeans.fit( df.values )
	#print( kmeans.labels_ )
	preds = kmeans.predict( df.values )
	#print(preds)
	plt.scatter( df[0][  preds == 1 ] ,  df[1][preds == 1 ] )
	plt.scatter( df[0][  preds == 0 ] ,  df[1][preds == 0 ] )
	plt.legend( ["Cluster 1" , "Cluster 2 "])
	plt.title( "K means - Naive ")
	plt.savefig( "NaiveKmeans.png")
	plt.clf()



	kmeans = KMeans(n_clusters=2, random_state=0 , init = "k-means++")
	kmeans.fit( newX1 )
	preds = kmeans.predict( newX1 )
	#print(preds)
	plt.scatter( df[0][  preds == 1 ] ,  df[1][preds == 1 ] )
	plt.scatter( df[0][  preds == 0 ] ,  df[1][preds == 0 ] )
	plt.legend( ["Cluster 1" , "Cluster 2 "])
	plt.title( "K means - L1 ")
	plt.savefig("Spectral-L1-{}.png".format(prefix1) )
	plt.clf()



	kmeans = KMeans(n_clusters=2, random_state=0 , init = "k-means++")
	kmeans.fit( newX2 )
	preds = kmeans.predict( newX2 )
	#print(preds)
	plt.scatter( df[0][  preds == 1 ] ,  df[1][preds == 1 ] )
	plt.scatter( df[0][  preds == 0 ] ,  df[1][preds == 0 ] )
	plt.legend( ["Cluster 1" , "Cluster 2 "])
	plt.title( "K means - L2 ")
	plt.savefig( "Spectral-L2-{}.png".format(prefix2))
	plt.clf()





if __name__ == "__main__":

	main()
