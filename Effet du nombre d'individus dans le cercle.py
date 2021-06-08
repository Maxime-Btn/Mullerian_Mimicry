"""
Effet du mimétisme müllérien sur la dynamique des populations (stage M1)

Question : Les cercles contenant plus d'individus sont-ils moins sensibles à la perturbation ?

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (tirés aléatoirement)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique, 0.1 pour interspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation
    - L -> degré de nocivité (fixé à 0.001)
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05)

Version Python : 3.9
Auteur : Maxime Boutin
"""
### importation des bibliothèques ###
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
from matplotlib.ticker import MultipleLocator # bibliothèque Matplotlib pour mettre en forme les axes
import random

Final = [] # enregistre tous les les données obtenues

Bl = [] # liste des états "maintien du cercle"
Ol = [] # liste des états "extinction de l'espèce perturbée"
Rl = [] # liste des états "extinction du cercle"
Nl = [] # liste des états "extinction au premier équilibre"

### définition des paramètres ###
m = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
n = np.array(10*[[25]], dtype="float32") # np.array permet de définir une matrice
c = np.array([[1,   0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1],
              [0.1,  1,     0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1],
              [0.1, 0.1,     1,     0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1],
              [0.1, 0.1,    0.1,     1,     0.1,    0.1,    0.1,    0.1,    0.1,    0.1],
              [0.1, 0.1,    0.1,    0.1,     1,     0.1,    0.1,    0.1,    0.1,    0.1],
              [0.1, 0.1,    0.1,    0.1,    0.1,     1,     0.1,    0.1,    0.1,    0.1],
              [0.1, 0.1,    0.1,    0.1,    0.1,    0.1,     1,     0.1,    0.1,    0.1],
              [0.1, 0.1,    0.1,    0.1,    0.1,    0.1,    0.1,     1,     0.1,    0.1],
              [0.1, 0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,     1,     0.1],
              [0.1, 0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,     1]])
K = 2500.0
L = 0.001
d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])

B = 0
O = 0
R = 0
N = 0

for p in range(0,152,1):
    p = p/100

    for _ in range(0,100): # permet de faire la simulation 100 fois pour une valeur de p, avec des r aléatoires
        n = np.array(10*[[25]], dtype="float32")
        r = np.round((np.random.uniform(0.2, 2, 10)).reshape(10, 1), 3) # tire aléatoirement les r

        t = 0 # permet de réinitialiser la variable "temps"

        ### calcul du premier équilibre ###
        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
            n[n<2]=0
            t+=1

        a = (n[0] + n[1]) # effectif du petit cercle
        A = float(np.round(a / 250) * 250) # permet d'associer à l'effectif du petit cercle, le multiple de 250 le plus proche
        if A == 0 : X = 3 # si le multiple est 0, on considère le cercle éteint

        else :

            t = 0  # permet de réinitialiser la variable "temps"
            x = [t]
            save = n

            ### calcul du second équilibre après la perturbation ###

            while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.001)) and t < 1000:
                # application de la perturbation pour t de 1 à 20
                if t <= 20 and t >= 1: d = np.array([[0.7], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.7]])
                else: d = np.array([[0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05]])
                n += r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
                n[n < 2] = 0
                save = np.dstack((save, n))  # ajout de la matrice "n" à la matrice "save"
                t += 1
                x.append(t)  # ajout de la valeur de t à la liste x

            if n[0]==0:
                if n[1]==0 :
                    X = 2 # enregistre l'état final '2' -> extinction du cercle
                else :
                    X = 1 # enregistre l'état final '1' -> extinction de l'espèce touchée seulement
            else:
                X = 0 # enregistre l'état final '0' -> pas d'extinction

        Paire = [A,X] # associe le multiple de 250 avec l'état final
        Final.append(Paire) # ajoute la paire à la liste Final
Final.sort() # trie la liste Final par ordre croissant de A


### calcul la fréquence de chaque état, pour chaque tranche de 250 ###
z = []
for ù in range(0, int(max(Final)[0] + 250), 250):
    z.append(ù)
    B = 0
    O = 0
    R = 0
    N = 0
    for w in range(0, len(Final)-1, 1):
        Coord = Final[w]
        if Coord[0] == ù :
            if Coord[1] == 0 : B += 1
            elif Coord[1] == 1 : O += 1
            elif Coord[1] == 2 : R += 1
            elif Coord[1] == 3: N += 1

    Tot = B+O+R+N

    if Tot != 0 :
        Bl.append(100*B/Tot)
        Ol.append(100*O/Tot)
        Rl.append(100*R/Tot)
        Nl.append(100*N/Tot)

    else :
        Bl.append(Tot)
        Ol.append(Tot)
        Rl.append(Tot)
        Nl.append(Tot)


### mise en forme  du diagramme en batons ###
BOl = [x1+y1 for x1,y1, in zip(Bl,Ol)]

plt.bar(z, Bl, label="Pas d'extinction", width=180, color="Royalblue")
plt.bar(z, Ol, bottom=Bl, label="Extinction de l'espèce perturbée", width=180, color="orange")
plt.bar(z, Rl, bottom=BOl, label="Extinction du cercle", width=180, color="Firebrick")

plt.legend(loc="lower right")

plt.xlabel("Effectif du cercle")
plt.ylabel("Fréquence des états finaux (en %)")
plt.xlim(100, int(max(Final)[0])+100)
ax = plt.axes()
ax.xaxis.set_major_locator(MultipleLocator(250))
ax.xaxis.set_minor_locator(MultipleLocator(250))

plt.show()











