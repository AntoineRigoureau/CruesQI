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
situation = 2
distributions = distributionCreation(situation)

#Creation de X
DX = ComposedDistribution(distributions,copula)
X = RandomVector(DX)

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

"""
#Estimation de la probilite:
#Methode de Monte-Carlo:
monte_carlo = MonteCarlo(fearedEvent)
monte_carlo.setMaximumOuterSampling(1000)
monte_carlo.setBlockSize(10)
monte_carlo.setMaximumCoefficientOfVariation(0.1)
Log.Show(Log.INFO)
f.enableHistory()
monte_carlo.run()
resultMC = monte_carlo.getResult()
print("L'estimation est: %",resultMC.getProbabilityEstimate())


#Estimation de la densite de probabilite de X conditionne a E:
#Par defaut, le noyau utilise est une loi normale N(0,1)
#TODO: tester d'autres noyaux, faire un beau graph
Xi = f.getHistoryInput().getSample()
Yeval = f.getHistoryOutput().getSample()
Xcond = getConditional(Xi,Yeval)
kernel = KernelSmoothing()
fittedX = kernel.build(Xcond)


#Methode d'importance sampling
importance = ImportanceSampling(fearedEvent,fittedX)
importance.setMaximumOuterSampling(100)
importance.setBlockSize(100)
importance.setMaximumCoefficientOfVariation(0.1)
Log.Show(Log.INFO)
importance.run()
resultIS = importance.getResult()
print("L'estimation est: %",resultIS.getProbabilityEstimate())

"""
#Approxiamtion de modeles:
#creation d'une base de polynomes:
d = 4
p = 3
DZ = ComposedDistribution(distributions)
enumerateFunction = LinearEnumerateFunction(d)
H = [StandardDistributionPolynomialFactory(AdaptiveStieltjesAlgorithm(DZ.getMarginal(i))) for i in range(d)]
productBasis = OrthogonalProductPolynomialFactory(H,LinearEnumerateFunction(d))
m = enumerateFunction.getStrataCumulatedCardinal(p)
print(m)

#Approximation par la methode LASSO avec LARS leave-one-out:
algoSelection = LeastSquaresMetaModelSelectionFactory(LAR(),CorrectedLeaveOneOut())
n = 1000
Xsample = DZ.getSample(n)
Ysample = f(Xsample)
fCCompute = FunctionalChaosAlgorithm(Xsample,Ysample,DZ,FixedStrategy(productBasis,m),LeastSquaresStrategy(algoSelection))
fCCompute.run()
fCFResult = fCCompute.getResult()
fC = fCFResult.getMetaModel()


YC = RandomVector(fC,X)
fearedEventBis = Event(YC,Greater(),58.)
"""
#Monte-Carlo applique a fC:
#Ici, comme il s'agit d'un modele d'approximation, on n'a pas de contraintes de calcul donc on peut faire beaucoup d'evaluations
monte_carloBis = MonteCarlo(fearedEventBis)
monte_carloBis.setMaximumOuterSampling(10000)
monte_carloBis.setBlockSize(1000)
monte_carloBis.setMaximumCoefficientOfVariation(0.01)
Log.Show(Log.INFO)
fC.enableHistory()
monte_carloBis.run()
resultMCBis = monte_carloBis.getResult()
print("L'estimation est: %",resultMCBis.getProbabilityEstimate())

#Utilisation de variables de controle:


control = ControlVariableVarianceReduction(fearedEventBis,YC,Y,58.0)
control.setIterationNumber(10000)
control.setBlockNumber(1000)
control.setMaximumCoefficientOfVariation(0.01,0.01)
control.run()
"""

#Analyse de sensibilite
sobol = FunctionalChaosRandomVector(fCFResult)
for i in range(4):
    print("Sobol {0} = {1}".format(i,sobol.getSobolIndex(i)))
    print("Sobol Total {0} = {1}".format( i, sobol.getSobolTotalIndex(i)))

#Estimation de l'esperence conditionelle a E chapeau:
k = 10000
conditionedSample = getConditional(X.getSample(k),YC.getSample(k))
conditionedMean = conditionedSample.computeMean()
print(fC(conditionedMean))

#On analyse la sensibilite en utilisant un algo FORM
#L'esperence estimee est utilisee comme point de depart
algo = AbdoRackwitz()
formAlgo = FORM(algo,fearedEventBis,conditionedMean)
formResults = formAlgo.run()
importance = formResults.getResult().getImportanceFactors()
