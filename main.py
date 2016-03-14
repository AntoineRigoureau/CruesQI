#Fichier principal pour le projet

from openturns import *
import numpy as np
from math import *
from Utilities import *
import matplotlib.pyplot as plt
from openturns.viewer import View

#Creation de la copule de X
copula = copulaCreation()

#Creation des distributions de X
situation = 1
distributions = distributionCreation(situation)

#Creation de X
X = ComposedDistribution(distributions,copula)

#Moyennes et Variances des loies marginales
for i in range(len(distributions)):
	mean = X.getMarginal(i).getMean()[0]
	var =  X.getMarginal(i).getMoment(2)[0] - mean*mean
	print("Moyenne de la marginale " + str(i) + " :"+ str(mean))
	print("Variance de la marginale " + str(i) + " :"+ str(var))


#Graph des densites marginales
#TODO: faire des jolis graph
#Pour voir les graph, ajouter .show() a la fin de la derniere linge du bloc
for i in range (len(distributions)):
	pdf_graph = X.drawMarginal1DPDF(i,0,100,1000)
	fig = plt.figure(figsize=(10, 5))
	pdf_axis = fig.add_subplot(111)
	View(pdf_graph, figure=fig, axes=[pdf_axis], add_legend=False).show()
	

#Calcul des matrices de cavariances, de Spearman et tau de Kendall de X
Xcov = X.getCovariance()
Xspea = X.getSpearmanCorrelation()
Xk = X.getKendallTau()
print("Matrice de covariance:")
print(Xcov)
print("Matrice de Spearman:")
print(Xspea)
print("Matrice tau de Kendall:")
print(Xk)


#Creation de la valeur d'interet: variable aleatoire Y
Y = f(X,100,10)

#Evaluations de Y et calcul de moyenne/ecart-type
Yi = Y.getSample(10000)
mean = sum(np.array(Yi))/10000
print(mean)

