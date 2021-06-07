"""
Effet du mimétisme müllérien sur la dynamique des populations (stage M1)

Question : le mimétisme limite-t-il le risque d'extinction ?

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (fixé à 1.1)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique, 0.1 pour interspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation (fixé à 0.50)
    - L -> degré de nocivité (fixé à 0.001)
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05)

Version Python : 3.9
Auteur : Maxime Boutin
"""
##### importation des bibliothèques #####
import numpy as np # bibliothèque NumPy pour les matrices
import csv # permet de manipuler des jeux de données csv

##### définition des paramètres #####
K = 2500.0
p = 0.5
L = 0.001
c = np.array([[1,   0.1,    0.1,    0.1,    0.1],
              [0.1,  1,     0.1,    0.1,    0.1],
              [0.1, 0.1,     1,     0.1,    0.1],
              [0.1, 0.1,    0.1,     1,     0.1],
              [0.1, 0.1,    0.1,    0.1,     1]]) # np.array permet de définir une matrice
m1 = np.array([[1,   0,  0,  0,  0],
               [0,   1,  0,  0,  0],
               [0,   0,  1,  0,  0],
               [0,   0,  0,  1,  0],
               [0,   0,  0,  0,  1]]) # matrice pour la communauté sans motif d'avertissement commun
m2 = np.array([[1,   1,  1,  1,  1],
               [1,   1,  1,  1,  1],
               [1,   1,  1,  1,  1],
               [1,   1,  1,  1,  1],
               [1,   1,  1,  1,  1]]) # matrice pour la communauté mimétique
d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05]])
ID = 0

##### Création du fichier csv #####
file = open("AppartenanceCercle.csv", "w", newline="") # genère un fichier nommé AppartenanceCercle.csv
writer = csv.writer(file) # démarre le mode "écriture"
writer.writerow(('ID', 'Communauté_1', 'Communauté_2', 'Nb_sp(sans cercle)', 'Nb_sp(avec cercle)')) # écrit la première ligne


for _ in range(0,5000):
    ID += 1
    n = np.array(5 * [[25]], dtype="float32")
    r = np.round((np.random.uniform(0.5, 1.5, 5)).reshape(5, 1), 3) # tire aléatoirement des r pour les 5 espèces

    t = 0 # permet de réinitialiser la variable "temps"

    m = m1 # la communauté est considérée sans motif d'avertissement commun

##### calcul du premier équilibre #####
    while np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001):
        n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n # np.dot(a,b) : produit matriciel a.b
        n[n<2]=0 # remplace tous les effectifs inférieurs à 2 par 0 (extinction)
        t+=1

    T = t # enregistre la valeur de t
    N1 = np.count_nonzero(n) # calcul du nombre d'espèces non-éteintes
    S1 = np.sum(n) # calcul du nombre d'individus dans la communauté

    n = np.array(5*[[25]], dtype="float32") # réinitialise les effectifs initiaux
    m = m2 # la communauté est considérée avec un cercle mimétique

    t = 0 # permet de réinitialiser la variable "temps"

##### calcul du nouvel équilibre, avec cercle #####
    while t < T:
        n += r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
        n[n < 2] = 0
        t+=1

    S2 = np.sum(n)
    N2 = np.count_nonzero(n)

    writer.writerow((ID, S1, S2, N1, N2)) # écrit une ligne de données

file.close() # ferme le fichier


