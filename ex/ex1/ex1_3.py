import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 



def reco_error( X , W ):

	Transformed = np.matmul( X , W.T  )
	Back = np.matmul( Transformed , W )

	error = np.sum(   np.sum( (X - Back)**2  , axis = 1) , axis = 0  )
	return error


df = pd.read_csv("./ex_1_data.csv" , header = None )
print("Size original data")
print(df.shape)

print("Number of dimensions(D):" , df.shape[1] )
print("Data points(N):" , df.shape[0])

print( "Calculate convariance matrix" )
Cov = df.cov()
print( "Shape covariance matrix: " , Cov.shape )

eigenvals , eigenvec = np.linalg.eig( Cov )

print( "Sorted EigenValues ")

sorted_eigenvals = sorted( eigenvals , reverse = True)
print( sorted_eigenvals )

vs = []

L = 2
W = np.zeros( ( 2 , df.shape[1])  )
for i in range(L):
	vs = eigenvec[  eigenvals == sorted_eigenvals[i]   ]
	#print(vs.shape)
	W[i , : ]  =  vs 


a = np.matmul(  df.values , W.T )

plt.scatter(  a[ : , 0 ] , a[: , 1 ] )
plt.ylabel( " First Principal component ")
plt.xlabel( "Second principal component")

plt.savefig("./first2components.png")
plt.clf()


errors = []
for l in range(  1 , 6 ):
	
	W = np.zeros( ( l , df.shape[i]))
	for j in range(l):
		v = eigenvec[  eigenvals == sorted_eigenvals[j]   ]
		W[j , : ] = v

	error = reco_error( df.values , W )
	errors.append( error )

plt.scatter( range(1,6) , errors)
plt.xlabel("Reconstruction Error")
plt.ylabel( "L ( number of reconstructed components) ")
plt.savefig("./reco_error.png")
plt.clf()


#reco_error( df.values , W )