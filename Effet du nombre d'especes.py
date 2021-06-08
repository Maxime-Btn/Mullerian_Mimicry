"""
Effet du mimétisme müllérien sur la dynamique des populations (stage M1)

Question : Les cercles contenant plus d'espèces sont-ils moins sensibles à la perturbation ?

Paramètres :
    - n -> matrice des effectifs initiaux (fixés à 25)
    - r -> taux de reproduction (fixé à 1.1)
    - c -> matrice des coefficients de compétition (1 pour intraspécifique, 0.1 pour interspécifique)
    - K -> capacité de charge du milieu (fixée à 2500)
    - p -> taux de prédation
    - L -> degré de nocivité (fixé à 0.001)
    - m -> matrice des taux de ressemblance entre 2 espèces (0 ou 1)
    - d -> matrice des taux de mortalité hors-prédation (0.05, ou 0.7 pour la perturbation)

Version Python : 3.9
Auteur : Maxime Boutin
"""
##### importation des bibliothèques #####
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
import seaborn as sns # bibliothèque Seaborn pour réaliser une heatmap

##### définition des paramètres #####
Etat = np.array([], dtype="float32")
m1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) # A = 1 espèce et B = 9 espèces
m2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]) # A = 2 espèces et B = 8 espèces
m3 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]) # A = 3 espèces et B = 7 espèces
m4 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]) # A = 4 espèces et B = 6 espèces
m5 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]) # A = 5 espèces et B = 5 espèces


for z in range(0,5):
    X = np.array([], dtype="float32") # stocke les états finaux
    P=[] # pour une taille de cercle, stocke les valeurs de p
    if z == 0:
        m=m5
    elif z == 1:
        m=m4
    elif z == 2:
        m=m3
    elif z == 3:
        m=m2
    elif z == 4:
        m=m1

    for p in range(0,151,5):
        p=p/100
        P.append(p) # enregistre la valeur de p

        ### définition des paramètres initiaux ###
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
        L = 0.001
        d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])

        ### calcul du premier équilibre ###
        t = 0 # permet de réinitialiser la variable "temps"

        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n # np.dot(a,b) : produit matriciel a.b
            n[n<2]=0 # remplace tous les effectifs inférieurs à 2 par 0 (extinction)
            t+=1

        if np.any(n==0): # si le cercle est éteint, on enregistre l'état final "3" et on arrête les calculs pour cette richesse spécifique
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
                if n[1]==0 or z==4 :
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
ax = sns.heatmap(Etat, square=True,linewidth=0.05,cbar_kws={"ticks":range(4),"shrink": 0.3}, linecolor="black",cmap=["Royalblue","Orange","Firebrick","Black"],xticklabels=P, yticklabels=[u"$N_A=5$",u"$N_A=4$",u"$N_A=3$",u"$N_A=2$",u"$N_A=1$"])
c_bar = ax.collections[0].colorbar
c_bar.set_ticks([0.30, 1, 1.85, 2.6])
c_bar.set_ticklabels(['Pas d\'extinction', 'Extinction de l\'espèce perturbée' , 'Extinction du cercle', "Extinction avant la perturbation"])
plt.xlabel("Taux de prédation")
plt.ylabel("Richesse spécifique \n du cercle")
plt.show()

