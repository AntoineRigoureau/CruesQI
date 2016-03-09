#Contient l'implementation de differentes fonctions utiles pour le projet

"""
Les fonctions presentes sont:
- copulaCreation
- distributionCreation
"""

from openturns import *
import numpy as np
from math import *

def copulaCreation():
	R = CorrelationMatrix(2)
	R[1,0] = R[0,1] = 2*np.sin((pi/6)*0.17)
	R[1,1] = R[0,0] = 2*np.sin(pi/6)
	copulas = ComposedCopula([IndependantCopula(2),NormalCopula(2,R])
	return(copulas)
	


def distributionCreation(situation):
	if (situation == 1):
		Q = Gamma(3.6239,1/134.827,0)
	if (situation == 2):
		Q = TruncatedDistribution(Gumbel(6.8951*np.exp(-4),663.75),0,side=TruncatedDistribution.LOWER)
	if (situation == 3) 
		Q = RandomMixture
		
	
			
