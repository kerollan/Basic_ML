# -*- coding: utf-8 -*-
# Python Project Template

# 1. Prepare Problem
# a) Load libraries
import numpy as np
from pandas import read_csv #on utilise cette librairie pour differencier les differentes "classes"
from pandas import set_option
from matplotlib import pyplot as plt
from pandas.tools.plotting import scatter_matrix

# b) Load dataset
url = 'https://goo.gl/mLmoIz'
noms = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'Classe']
data = read_csv(url, names=noms)

print 'Taille totale (lignes*colonnes) :', data.shape
print '\nLectures des 20 premières lignes : ', (data.head(20))
print '\nTypes des attributs :\n', data.dtypes


# 2. Summarize Data
# a) Descriptive statistics
#Class distribution
class_cpt = data.groupby('Classe').size()
print '\nNombre de données / classes : \n', class_cpt
#Correlation entre attributs
set_option('display.width', 100)
set_option('precision', 5)
correlations = data.corr(method='pearson')
print correlations
#Distorsion des valeurs (skew)
print '\nDistorsions des valeurs :\n', data.skew()

# b) Data visualizations
#'Histogramme des valeurs des IRIS'
data.hist(color = 'red')
plt.show()

#Affichage des densités des valeurs
data.plot(kind='density', subplots=False,  sharex=True)
#On les affiche sur le même graphique puisqu'ils ont la même échelle
plt.show()
#Boites et tracés
data.plot(kind='box', subplots=True, layout = (2,2), sharex=False, sharey=False)
plt.show()

#Matrice de correlation
def test():
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    figure.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(noms)
    ax.set_yticklabels(noms)
    plt.show()

#Scatter plot
scatter_matrix(data)
plt.show()

# 3. Prepare Data
#a) Data Cleaning
#b) Feature Selection
#c) Data Transforms
#No needs, already normalized

# 4. Evaluate Algorithms
#a) Split-out validation dataset
#b) Test options and evaluation metric
#c) Spot Check Algorithms
#d) Compare Algorithms



# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

#6. Finalize Model
#a) Predictions on validation dataset
#b) Create standalone model on entire training dataset
#c) Save model for later use
#
#
#
#