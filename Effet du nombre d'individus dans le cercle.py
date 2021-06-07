"""
Descriptif : ce programme est composé de 2 parties : tout d'abord le calcul des effectifs à l'équilibre pour les 10
    populations, puis une perturbation est appliquée (augmentation du taux de mortalité "delta") sur une pop de chaque
    cercle mimétique afin de voir s'il y a une répercussion sur les autres pop. Les 10 populations sont réparties selon
    deux cercles mimétiques :
        - cercle A -> pop 1 et pop 2 (2 populations)
        - cercle B -> pop 3 à pop 10 (8 populations)

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (fixé à 1.1)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique, 0.1 pour interspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation (fixé à 0.50)
    - L -> degré de nocivité (fixé à 0.001)
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05, ou 0.7 pour la perturbation)

Version :
"""
### importation des bibliothèques ###
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
from matplotlib.ticker import MultipleLocator
import random

Final = []

Bl = []
Ol = []
Rl = []
Nl = []


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

    print(p)
    for _ in range(0,100):
        print(_)
        n = np.array(10*[[25]], dtype="float32")
        r = np.round((np.random.uniform(0.2, 2, 10)).reshape(10, 1), 3)


        t = 0 # permet de réinitialiser la variable "temps"

        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
            n[n<2]=0
            t+=1

        a = (n[0] + n[1])
        A = float(np.round(a / 250) * 250)
        if A == 0 : X = 3
        else :

            t = 0  # permet de réinitialiser la variable "temps"
            x = [t]
            save = n

            while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.001)) and t < 1000:
                if t <= 20 and t >= 1:
                    d = np.array([[0.7], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.7]])
                else:
                    d = np.array([[0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05]])
                n += r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
                n[n < 2] = 0
                save = np.dstack((save, n))  # ajout de la matrice "n" à la matrice "save"
                t += 1
                x.append(t)  # ajout de la valeur de t à la liste x

            if n[0]==0:
                if n[1]==0 :
                    X = 2
                else :
                    X = 1
            else:
                X = 0

        Test = [A,X]
        print(Test)
        Final.append(Test)
Final.sort()
print(Final)

print(max(Final)[0]+250)
print(len(Final))

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
            elif Coord[1] == 3 : N +=1
    Tot = B+O+R+N
    print("-----------------------"+str(Tot))
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

print(Bl)
print(Ol)
print(Rl)
print(Nl)
print("-----------------------")
BOl = [x1+y1 for x1,y1, in zip(Bl,Ol)]
print(BOl)
plt.bar(z, Bl, label="Pas d'extinction", width=180, color="Royalblue")
plt.bar(z, Ol, bottom=Bl, label="Extinction de l'espèce perturbée", width=180, color="orange")
plt.bar(z, Rl, bottom=BOl, label="Extinction du cercle", width=180, color="Firebrick")

plt.legend(loc="lower right")

plt.xlabel("Effectif du cercle")
plt.ylabel("Fréquence des états finaux (en %)")
plt.xlim(100, int(max(Final)[0])+100)
ax = plt.axes()
#ax.yaxis.set_major_locator(MultipleLocator(0.5))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(250))
ax.xaxis.set_minor_locator(MultipleLocator(250))

plt.show()











