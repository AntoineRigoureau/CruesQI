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
	D2 = TruncatedDistribution(Gumbel(np.exp(-4)*6.8951,663.75),0.,TruncatedDistribution.LOWER)
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
	
	
class ControlVariableVarianceReduction:
	
	def __init__(self,event1,d1,d2,t):
		self.eventR = event1
		self.YR = d1
		self.Y = d2
		self.N = 1000
		self.P = 1
		self.log = False
		self.ProbabilityEstimation = 0
		self.MaxC = [-1,-1]
		self.threshold = t
		#Moyenne du modele reduit, du terme de controle
		self.meanR = 0.
		self.mean = 0.
		#Variance du modele reduit, du terme de controle
		self.VarianceR = 0.
		self.Variance = 0.
		#Ecart-types du modele reduit, du terme de controle
		self.sdR = 0.
		self.sd = 0.
		#Coefficient de variation du modele reduit, du terme de controle
		self.cR = 0.
		self.c = 2.
		
	def setIterationNumber(self,n):
		self.N = n
	
	def setBlockNumber(self,p):
		self.P = p
	
	def setMaximumCoefficientOfVariation(self, cmaxR = -1, cmax = -1):
			self.MaxC[0] = cmaxR
			self.MaxC[1] = cmax
	
	def getMean(self):
		return([self.meanR,self.mean])
	
	def getVariance(self):
		return([self.VarianceR,self.Variance])
		
	def getStandardDeviation(self):
		return([self.sdR,self.sd])
		
	def getVariationCoefficient(self):
		return([self.cR,self.c])
	
	def getProbabilityEstimate(self):
		return(self.ProbabilityEstimation)
		
		
	def run(self):
		#Estimation du premier terme par Monte-Carlo
		monte_carlo = MonteCarlo(self.eventR)
		monte_carlo.setMaximumOuterSampling(self.N)
		monte_carlo.setBlockSize(self.P)
		if (self.MaxC[0] >0):
			monte_carlo.setMaximumCoefficientOfVariation(self.MaxC[0])
		monte_carlo.run()
		self.meanR = monte_carlo.getResult().getProbabilityEstimate()
		self.VarianceR = monte_carlo.getResult().getVarianceEstimate()
		self.sdR = monte_carlo.getResult().getStandardDeviation()
		self.cR = monte_carlo.getResult().getCoefficientOfVariation()
		
		
		#Estimation du second terme
		value = 0.
		varValue = 0.
		i = 0
		while(i < self.N*self.P and self.c > self.MaxC[1]):
			"""
			#Tests d'arrets en fonction des coeffs de variation
			if (self.MaxC[1] > 0):
				if (self.MaxC[1] > self.c):
					break
			"""
			
			#Evaluation des sommes
			iterValue = 0.
			Yvalue = self.Y.getRealization()
			YRvalue = self.YR.getRealization()
			if (self.threshold < YRvalue[0]):
				iterValue = iterValue - 1
			if (self.threshold < Yvalue[0]):
				iterValue = iterValue + 1
			#Mise a jour des valeurs
			if (iterValue != 0):
				value = value +iterValue
				self.mean = value/(i+1)
				varValue = varValue + iterValue*iterValue
				self.Variance = varValue/(i+1) - self.mean*self.mean
				if (self.Variance >0):
					self.sd = sqrt(self.Variance)
				if (self.mean != 0):
					self.c = abs(self.sd/self.mean)
			i = i+1
		self.ProbabilityEstimation = self.meanR + self.mean
		print("Estimation du terme approche: {0} avec une variance de: {1} et un coefficient de variation de {2}".format(self.meanR,self.VarianceR,self.cR))
		print("Estimation du terme de controle: {0} avec une variance de: {1} et un coefficient de variation de {2} en {3} iterations".format(self.mean,self.Variance,self.c,i))
		print("Estimation de probabilite: {0}".format(self.ProbabilityEstimation))		
