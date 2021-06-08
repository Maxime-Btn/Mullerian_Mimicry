"""
Effet du mimétisme müllérien sur la dynamique des populations (stage M1)

Question : Quel est l'effet du degré de nocivité sur la résistance à la perturbation ?

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (fixé à 1.1)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique, 0.1 pour interspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation
    - L -> degré de nocivité
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05, ou 0.7 pour la perturbation)

Version Python : 3.9
Auteur : Maxime Boutin
"""
### importation des bibliothèques ###
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
import seaborn as sns # bibliothèque Seaborn pour réaliser les heatmaps

### définition des paramètres ###
Etat = np.array([], dtype="float32")
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

for z in range(0,5):
    X = np.array([], dtype="float32") # stocke les états finaux
    P=[] # stocke les valeurs de p
    if z == 0: # L = 0.005 pour le cercle A et L = 0.001 pour le cercle B
        L = np.array([[0.005],[0.005],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001]])
    elif z == 1: # L = 0.004 pour le cercle A et L = 0.001 pour le cercle B
        L = np.array([[0.004],[0.004],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001]])
    elif z == 2: # L = 0.003 pour le cercle A et L = 0.001 pour le cercle B
        L = np.array([[0.003],[0.003],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001]])
    elif z == 3: # L = 0.002 pour le cercle A et L = 0.001 pour le cercle B
        L = np.array([[0.002],[0.002],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001]])
    elif z == 4: # L = 0.001 pour le cercle A et L = 0.001 pour le cercle B
        L = np.array([[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001],[0.001]])

    for p in range(0,151,5):
        p=p/100
        P.append(p) # enregistre la valeur de p
        n = np.array(10*[[25]], dtype="float32")
        r = 1.1
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
        d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])

        ### calcul du premier équilibre ###
        t = 0 # permet de réinitialiser la variable "temps"

        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n # np.dot(a,b) : produit matriciel a.b
            n[n<2]=0 # remplace tous les effectifs inférieurs à 2 par 0 (extinction)
            t+=1

        if np.any(n==0): # si le cercle est éteint, on enregistre l'état final "3" et on arrête les calculs pour cette valeur de L
            X = np.append(X, [3])
        else:
            ### calcul du second équilibre après la perturbation ###

            t = 0 # permet de réinitialiser la variable "temps"

            while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)):
                # application de la perturbation pour t de 1 à 20
                if t <= 20 and t>=1 : d = np.array([[0.7], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.7]])
                else : d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])
                n += r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
                n[n < 2] = 0
                t+=1

            if n[0]==0:
                if n[1]==0 :
                    X = np.append(X, [2]) # enregistre l'état final '2' -> extinction du cercle
                else:
                    X = np.append(X, [1]) # enregistre l'état final '1' -> extinction de l'espèce touchée seulement
            else:
                X = np.append(X, [0]) # enregistre l'état final '0' -> pas d'extinction
    if z == 0:
        Etat = X
    else:
        Etat=np.vstack((Etat,X)) # stocke toutes les valeurs de X pour tracer le graphique

### mise en forme de la heatmap ###
ax = sns.heatmap(Etat, square=True,linewidth=0.10,cbar_kws={"ticks":range(4),"shrink": 0.3}, linecolor="black",cmap=["Royalblue","Orange","Firebrick","Black"],xticklabels=P, yticklabels=["5 (0.005)","4 (0.004)","3 (0.003)","2 (0.002)","1 (0.001)"])
c_bar = ax.collections[0].colorbar
c_bar.set_ticks([0.30, 1, 1.85, 2.6])
c_bar.set_ticklabels(["Pas d'extinction", "Extinction de l'espèce perturbée", "Extinction du cercle", "Extinction avant la perturbation"])
plt.xlabel("Taux de prédation")
plt.ylabel("ratio "+u"$ \lambda_A  /  \lambda_B$")
plt.show()

