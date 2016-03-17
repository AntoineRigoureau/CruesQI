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
situation = 3
distributions = distributionCreation(situation)

#Creation de X
X = RandomVector(ComposedDistribution(distributions,copula))
print(X.getMarginal(1))

#Moyennes et Variances des loies marginales
for i in range(len(distributions)):
	mean = X.getDistribution().getMarginal(i).getMean()[0]
	var =  X.getDistribution().getMarginal(i).getMoment(2)[0] - mean*mean
	print("Moyenne de la marginale " + str(i) + " :"+ str(mean))
	print("Variance de la marginale " + str(i) + " :"+ str(var))


#Graph des densites marginales
#TODO: faire des jolis graph
#Pour voir les graph, ajouter .show() a la fin de la derniere linge du bloc
for i in range (len(distributions)):
	pdf_graph = X.getDistribution().drawMarginal1DPDF(i,0,100,1000)
	fig = plt.figure(figsize=(10, 5))
	pdf_axis = fig.add_subplot(111)
	View(pdf_graph, figure=fig, axes=[pdf_axis], add_legend=False)
	

#Calcul des matrices de cavariances, de Spearman et tau de Kendall de X
Xcov = X.getDistribution().getCovariance()
Xspea = X.getDistribution().getSpearmanCorrelation()
Xk = X.getDistribution().getKendallTau()
print("Matrice de covariance:")
print(Xcov)
print("Matrice de Spearman:")
print(Xspea)
print("Matrice tau de Kendall:")
print(Xk)


#Creation de la valeur d'interet: variable aleatoire Y
Y = RandomVector(f,X)

#Evaluations de Y et calcul de moyenne/ecart-type
Yi = Y.getSample(10000)
mean = sum(np.array(Yi))/10000
var = sum(np.array(Yi)*np.array(Yi))/10000 - mean*mean
print("Moyenne de 10000 evaluations de Y: "+str(mean[0]))
print("Variance de 10000 evaluations de Y: "+str(var[0]))

#Estimation par noyaux de la densite:
#Par defaut, le noyau utilise est une loi normale N(0,1)
#TODO: tester d'autres noyaux, faire un beau graph
kernel = KernelSmoothing()
fittedDistribution = kernel.build(Yi)
graph = fittedDistribution.drawPDF()
View(graph)


#Creation de l'evenement redoute:
fearedEvent = Event(Y,Greater(),58.)
