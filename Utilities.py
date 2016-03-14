#Contient l'implementation de differentes fonctions utiles pour le projet

"""
Les fonctions presentes sont:
- copulaCreation qui va creer la copule pour X
- distributionCreation qui va creer les distributions de X
- la fonction f, permettant d'obtenir la variable Y a partir du vecteur aleatoire X
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
	return([Q,K,Zv,Zm])



def f(X,L,B):
	Y = (X.getMarginal(3) - X.getMarginal(2))/L
	Y.sqrt()
	Y = ProductDistribution(Y, X.getMarginal(1)*B)
	inv = NumericalMathFunction('x','1.0/x')
	b0 = Y.getRange().getLowerBound()
	bN = Y.getRange().getUpperBound()
	a = [b0[0],bN[0]]
	g = [SpecFunc.MaxNumericalScalar, inv(bN)[0]]
	Y = CompositeDistribution(inv,Y,a,g)
	Y = ProductDistribution(Y,X.getMarginal(0))
	a0 = Y.getRange().getLowerBound()
	aN = Y.getRange().getUpperBound()
	puiss = NumericalMathFunction('x','x^(3./5.)')
	a = [a0[0],aN[0]]
	g = [puiss(a0)[0], puiss(aN)[0]]
	Y = CompositeDistribution(puiss,Y,a,g)
	return(Y)
	
			
