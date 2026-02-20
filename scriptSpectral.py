import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

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


#####################
## Analyse Spectrale
#####################

# Configuration du modèle
spectral_model = SpectralClustering(n_clusters=3, 
                                    affinity='nearest_neighbors', 
                                    assign_labels='kmeans', 
                                    random_state=42)

# Application sur le Jeu 1
specLabels1 = spectral_model.fit_predict(features1)

# Application sur le Jeu 2
specLabels2 = spectral_model.fit_predict(features2)

# --- Affichage des matrices de confusion ---
print("\n--- Confusion Spectral Clustering Jeu 1 ---")
print(pd.crosstab(labels1, specLabels1, rownames=['Réel'], colnames=['Prédit']))

print("\n--- Confusion Spectral Clustering Jeu 2 ---")
print(pd.crosstab(labels2, specLabels2, rownames=['Réel'], colnames=['Prédit']))

# --- Visualisation ---
visuSpectral = True
if visuSpectral:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Graphique Jeu 1
    ax1.scatter(features1[:, 0], features1[:, 1], c=specLabels1, cmap='tab10', marker='o', edgecolors='k')
    ax1.set_title("Spectral Clustering - Jeu 1")
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # Graphique Jeu 2
    ax2.scatter(features2[:, 0], features2[:, 1], c=specLabels2, cmap='tab10', marker='o', edgecolors='k')
    ax2.set_title("Spectral Clustering - Jeu 2")
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.show()


from sklearn.manifold import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D # Nécessaire pour le rendu 3D

# 1. Calcul de l'embedding spectral en 3 dimensions
embed_3d = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', random_state=42)
features2_spectral_3d = embed_3d.fit_transform(features2)

# 2. Création de la figure 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Affichage des points
# On utilise c=labels2 pour vérifier si les vrais groupes sont bien isolés
scatter = ax.scatter(features2_spectral_3d[:, 0], 
                     features2_spectral_3d[:, 1], 
                     features2_spectral_3d[:, 2], 
                     c=labels2, 
                     cmap='viridis', 
                     s=60, 
                     edgecolors='k', 
                     alpha=0.8)

# Configuration des axes
ax.set_title("Projection du Jeu 2 dans l'Espace Spectral 3D")
ax.set_xlabel("Vecteur propre 1")
ax.set_ylabel("Vecteur propre 2")
ax.set_zlabel("Vecteur propre 3")

# Ajout d'une légende pour les clusters réels
legend1 = ax.legend(*scatter.legend_elements(), title="Groupes Réels")
ax.add_artist(legend1)

plt.show()
