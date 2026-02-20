import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# 1. Création des 3 globules 
# Note : centers doit être une liste de listes [[x, y]]
x1, _ = make_blobs(n_samples=20, centers=[[2, 2]], cluster_std=.5, random_state=42)
x2, _ = make_blobs(n_samples=20, centers=[[6, 2]], cluster_std=.5, random_state=42)
x3, _ = make_blobs(n_samples=20, centers=[[12, 2]], cluster_std=.5, random_state=42)

x4, _ =make_blobs(n_samples=5, centers=[[2, 2]], cluster_std=.5, random_state=42)
x5, _ = make_blobs(n_samples=20, centers=[[7, 2]], cluster_std=.5, random_state=42)

# 2. Fusion des données (Features)
features1 = np.vstack([x1, x2, x3])
labels1 = np.repeat([0, 1, 2], 20)
features2= np.vstack([x4, x2, x5])
labels2= np.concatenate((np.repeat([0], 5),np.repeat([1,2], 20)))

# 4. Construction du DataFrame
jeu1 = pd.DataFrame(features1, columns=['X', 'Y'])
jeu1['label'] = labels1

jeu2 = pd.DataFrame(features2, columns=['X', 'Y'])
jeu2['label'] = labels2

print (jeu2['label'])

# Vérifications
print(f"Dimensions : {jeu1.shape}")
print(jeu1.head(3))

print(f"Dimensions : {jeu2.shape}")
print(jeu2.head(3))
                
# 5. Visualisation correcte
visuOrig=False
if (visuOrig==True):
    plt.figure(figsize=(15, 15))
    plt.scatter(jeu1['X'], jeu1['Y'], c=jeu1['label'], cmap='viridis', edgecolors='k')
    plt.title("Jeu1 : Visualisation des 3 globules décalés")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.scatter(jeu2['X'], jeu2['Y'], c=jeu2['label'], cmap='viridis', edgecolors='k')
    plt.title("Jeu 2 Visualisation des 3 globules : 1 décalée avec peu d'objets et deux qui se chevauchent")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

#####################
## Analyse kmeans
#####################

km1= KMeans(n_clusters=3,n_init=10,max_iter=20,verbose=True)
km1.fit(features1)

km2= KMeans(n_clusters=3,n_init=10,max_iter=20,verbose=True)
km2.fit(features2)

visuKmeans=False
if (visuKmeans==True):
    plt.figure(figsize=(15, 15))
    plt.scatter(features1[:, 0], features1[:, 1], c=km1.labels_,marker='+')
    plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],marker='+',c='red') 
    plt.title(f"k-means jeu1", fontdict={"fontsize": 12})
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    plt.figure(figsize=(15, 15))
    plt.scatter(features2[:, 0], features2[:, 1], c=km2.labels_,marker='+')
    plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],marker='+',c='red') 
    plt.title(f"k-means jeu2", fontdict={"fontsize": 12})
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

print("confusion kmeans jeu 1")
print(pd.crosstab(labels1, km1.labels_))

print("confusion kmeans jeu2")
print(pd.crosstab(labels2, km2.labels_))


#####################
## Analyse Hierachique agglomerative
#####################

aggloK3 = AgglomerativeClustering(n_clusters=3, metric = 'euclidean', linkage = 'complete')
HClabels1=aggloK3.fit_predict(features1)

aggloK3 = AgglomerativeClustering(n_clusters=3, metric = 'euclidean', linkage = 'complete')
HClabels2=aggloK3.fit_predict(features2)

print("confusion HC jeu 1")
print(pd.crosstab(labels1, HClabels1))

print("confusion HC jeu2")
print(pd.crosstab(labels2, HClabels2))

visuHC=True
if (visuHC==True):
    plt.figure(figsize=(15, 15))
    plt.scatter(features1[:, 0], features1[:, 1], c=HClabels1,marker='+')
    plt.title(f"HC jeu1", fontdict={"fontsize": 12})
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    plt.figure(figsize=(15, 15))
    plt.scatter(features2[:, 0], features2[:, 1], c=HClabels2,marker='+')
    plt.title(f"HC jeu2", fontdict={"fontsize": 12})
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

###############
# dbscan

from sklearn.cluster import DBSCAN

# 1. Configuration de DBSCAN
# eps : Rayon de recherche. Si trop petit -> beaucoup de bruit (-1). Si trop grand -> 1 seul bloc.
# min_samples : Nombre de points minimum pour valider un "cœur" de cluster.
dbscan = DBSCAN(eps=1.1, min_samples=4)
db_labels1 = dbscan.fit_predict(features1)
n_clusters1_ = len(set(db_labels1)) - (1 if -1 in db_labels1 else 0)
n_noise1_ = list(db_labels1).count(-1)

db_labels2 = dbscan.fit_predict(features2)
n_clusters2_ = len(set(db_labels2)) - (1 if -1 in db_labels2 else 0)
n_noise2_ = list(db_labels2).count(-1)

print("Jeu 1")
print(f"Nombre de clusters estimés : {n_clusters1_}")
print(f"Nombre de points considérés comme bruit : {n_noise1_}")
print(pd.crosstab(labels1, db_labels1, rownames=['Réel'], colnames=['Prédit']))

print("Jeu 2")
print(f"Nombre de clusters estimés : {n_clusters2_}")
print(f"Nombre de points considérés comme bruit : {n_noise2_}")
print(pd.crosstab(labels2, db_labels2, rownames=['Réel'], colnames=['Prédit']))


plt.figure(figsize=(15, 5))
scatter = plt.scatter(features1[:, 0], features1[:, 1], c=db_labels1, cmap='viridis', marker='o', edgecolors='k')
plt.title(f"DBSCAN : {n_clusters1_} clusters trouvés (eps=1.1, min_samples=4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.colorbar(scatter, label='ID Cluster (-1 = Bruit)')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 5))
scatter = plt.scatter(features2[:, 0], features2[:, 1], c=db_labels2, cmap='viridis', marker='o', edgecolors='k')
plt.title(f"Jeu 2 DBSCAN : {n_clusters2_} clusters trouvés (eps=1.1, min_samples=4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.colorbar(scatter, label='ID Cluster (-1 = Bruit)')
plt.grid(True, alpha=0.3)
plt.show()



