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

Version : 2 (24/02/2021)
"""
### importation des bibliothèques ###
import numpy as np # bibliothèque NumPy pour les matrices
import matplotlib.pyplot as plt # bibliothèque Matplotlib pour les plots
import seaborn as sns

### définition des paramètres ###
Y = np.array([], dtype="float32")
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
    X = np.array([], dtype="float32")
    P=[]
    if z == 0:
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
    elif z == 1:
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
    elif z == 2:
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
    elif z == 3:
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
    elif z == 4:
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
        P.append(p)
        n = np.array(10*[[25]], dtype="float32") # np.array permet de définir une matrice
        r = 1.1
        K = 2500.0
        d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])

########################## 1. CALCUL DES EQUILIBRES ##########################
        t = 0 # permet de réinitialiser la variable "temps"


        while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)) and t < 500:
            n += r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n # np.dot(a,b) : produit matriciel a.b
            n[n<2]=0 # remplace tous les effectifs inférieurs à 2 par 0 (extinction)
            t+=1
        if np.any(n==0):
            X = np.append(X, [3])
            print("----------------------------------------------")
            print("p = ", p, " et taille du cercle = ", np.sum(m, axis=1)[0])
            print("3")
        else:



############################## 2. PERTURBATION ###############################

            t = 0 # permet de réinitialiser la variable "temps"
            x = [t] # enregistre toutes les valeurs de t (liste), pour tracer la courbe
            save = n # enregistre tous les effectifs (matrice), pour tracer la courbe
            ne = n + (r * n * (1 - np.dot(c,n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n)

            while (t < 100 or np.any(abs(r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n) > 0.0001)):
    # applique la perturbation pour la pop1 et la pop10, de la génération 1 à la génération 20
                if t <= 20 and t>=1 : d = np.array([[0.7], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.7]])
                else : d = np.array([[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05],[0.05]])
                n += r * n * (1 - np.dot(c, n) / K) - (p / (1 + np.dot(m, L * n))) * n - d * n
                n[n < 2] = 0
                save = np.dstack((save, n)) # ajout de la matrice "n" à la matrice "save"
                t+=1
                x.append(t) # ajout de la valeur de t à la liste x

            print("----------------------------------------------")
            print("p = ",p," et taille du cercle = ",np.sum(m, axis=1)[0])
            if n[0]==0:
                if n[1]==0 :
                    print("2")
                    X = np.append(X, [2])
                else:
                    print("1")
                    X = np.append(X, [1])
            else:
                print("0")
                X = np.append(X, [0])
    if z == 0:
        Y = X
    else:
        Y=np.vstack((Y,X))

print(Y)
ax = sns.heatmap(Y, square=True,linewidth=0.10,cbar_kws={"ticks":range(4),"shrink": 0.3}, linecolor="black",cmap=["Royalblue","Orange","Firebrick","Black"],xticklabels=P, yticklabels=[u"$c_{intraring}=0.5$",u"$c_{intraring}=0.4$",u"$c_{intraring}=0.3$",u"$c_{intraring}=0.2$",u"$c_{intraring}=0.1$"])
c_bar = ax.collections[0].colorbar
c_bar.set_ticks([0.30, 1, 1.85, 2.6])
c_bar.set_ticklabels(["Pas d'extinction", "Extinction de l'espèce perturbée", "Extinction du cercle", "Extinction avant la perturbation"])
plt.xlabel("Taux de prédation")
plt.ylabel("Compétition intra-cercle")
plt.show()

