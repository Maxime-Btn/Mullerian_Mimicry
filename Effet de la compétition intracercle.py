"""
Effet du mimétisme müllérien sur la dynamique des populations (stage M1)

Question : Quel est l'effet de la compétition intracercle sur la résistance à la perturbation ?

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (fixé à 1.1)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation
    - L -> degré de nocivité (fixé à 0.001)
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05, ou 0.7 pour la perturbation)

Version Python : 3.9
Auteur : Maxime Boutin
"""
### importation des bibliothèques ###
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
import seaborn as sns # bibliothèque Seaborn pour réaliser une heatmap

##### définition des paramètres #####
Etat = np.array([], dtype="float32")
L = 0.001
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
    if z == 0: # compétition intracercle de 0.5 et intercercle de 0.1
        c = np.array([[1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.5, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5],
                      [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]])
    elif z == 1: # compétition intracercle de 0.4 et intercercle de 0.1
        c = np.array([[1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.4, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 1, 0.4, 0.4, 0.4, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 0.4, 1, 0.4, 0.4, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 1, 0.4, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 1, 0.4, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1, 0.4],
                      [0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1]])
    elif z == 2: # compétition intracercle de 0.3 et intercercle de 0.1
        c = np.array([[1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.3, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 0.3],
                      [0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1]])
    elif z == 3: # compétition intracercle de 0.2 et intercercle de 0.1
        c = np.array([[ 1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.2,  1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1]])
    elif z == 4: # compétition intracercle de 0.1 et intercercle de 0.1
        c = np.array([[1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1]])

    for p in range(0,151,5):
        p=p/100
        P.append(p) # enregistre la valeur de p

        ### définition des paramètres initiaux ###
        n = np.array(10*[[25]], dtype="float32")
        r = 1.1
        K = 2500.0
        d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])

        ### calcul du premier équilibre ###
        t = 0 # permet de réinitialiser la variable "temps"


        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n # np.dot(a,b) : produit matriciel a.b
            n[n<2]=0 # remplace tous les effectifs inférieurs à 2 par 0 (extinction)
            t+=1
        if np.any(n==0):
            X = np.append(X, [3]) # si le cercle est éteint, on enregistre l'état final "3" et on arrête les calculs
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
ax = sns.heatmap(Etat, square=True,linewidth=0.10,cbar_kws={"ticks":range(4),"shrink": 0.3}, linecolor="black",cmap=["Royalblue","Orange","Firebrick","Black"],xticklabels=P, yticklabels=[u"$c_{intraring}=0.5$",u"$c_{intraring}=0.4$",u"$c_{intraring}=0.3$",u"$c_{intraring}=0.2$",u"$c_{intraring}=0.1$"])
c_bar = ax.collections[0].colorbar
c_bar.set_ticks([0.30, 1, 1.85, 2.6])
c_bar.set_ticklabels(["Pas d'extinction", "Extinction de l'espèce perturbée", "Extinction du cercle", "Extinction avant la perturbation"])
plt.xlabel("Taux de prédation")
plt.ylabel("Compétition intra-cercle")
plt.show()

