#Contient l'implementation de differentes fonctions utiles pour le projet

"""
Les fonctions presentes sont:
- copulaCreation qui va creer la copule pour X
- distributionCreation qui va creer les distributions de X
- la fonction f, permettant d'obtenir la variable Y a partir du vecteur aleatoire X
- getConditional
"""

from openturns import *
import numpy as np
from math import *

def copulaCreation():
	R = CorrelationMatrix(2)
	R[1,0] = R[0,1] = 2*np.sin((pi/6)*0.17)
	R[1,1] = R[0,0] = 2*np.sin(pi/6)
	copulas = ComposedCopula([IndependentCopula(2),NormalCopula(R)])
	return(copulas)
	


def distributionCreation(situation):
	D1 = Gamma(3.6239,(1./134.827),0)
	D2 = TruncatedDistribution(Gumbel(6.8951*np.exp(-4),663.75),0,TruncatedDistribution.LOWER)
	D3 = RandomMixture([D1,D2],[0.85,0.15])
	if (situation == 1):
		Q = D1
	if (situation == 2):
		Q = D2
	if (situation == 3):
		Q = D3
	K = TruncatedDistribution(Normal(30,7.5),0,TruncatedDistribution.LOWER)
	Zv = Triangular(49,50,51)
	Zm = Triangular(54,55,56)
	return([Q,K,Zm,Zv])



f = NumericalMathFunction(["Q","K","Zm","Zv"], ["y"], ["Zv + (Q/(K*10*sqrt((Zm-Zv)/100)))^(3.0/5.0)"])

def getConditional(X,Y):
	Xout = []
	for i in range(len(Y)):
		if (Y[i][0] > 58.):
			Xout.append(X[i])
	return(NumericalSample(np.array(Xout)))
			
