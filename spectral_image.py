import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
from scipy.linalg import eigh

def compute_k(pixels, max_k=10, sigma=1.0):
    # 1. Similarité
    distances = cdist(pixels, pixels)
    W = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # 2. Degré
    D = np.diag(W.sum(axis=1))

    # 3. Laplacien
    L = D - W

    # 4. Valeurs propres
    eigenvalues, _ = eigh(L)

    # 5. Eigengap
    gaps = np.diff(eigenvalues[:max_k])
    k = np.argmax(gaps) + 1

    return k

# 1. Configuration du chemin
dossier = ""
fichier = "img1.JPG"
chemin_complet = os.path.join(dossier, fichier)

# 2. Chargement et conversion
try:
    img = Image.open(chemin_complet)
except FileNotFoundError:
    print("Erreur : fichier non trouvé.")
    exit()

img_np = np.array(img)
img_np = img_np[300:900, 0:]

lambda_ = 0.2
k = 3
taille_bloc = 300
stride = 100

h, w, c = img_np.shape

image_sum = np.zeros_like(img_np, dtype=float)
image_count = np.zeros((h, w, 1), dtype=float)

#####################
## Analyse KMeans
#####################

for i in range(0, h, stride):
    for j in range(0, w, stride):

        block = img_np[i:i+taille_bloc, j:j+taille_bloc]
        bh, bw, bc = block.shape

        # éviter les blocs incomplets
        if bh < taille_bloc or bw < taille_bloc:
            continue

        pixels = block.reshape(-1, bc)

        k = compute_k(pixels[:500]) 
        k = max(2, min(k, 6))

        model = KMeans(n_clusters=k, n_init=10)


        coords = np.indices((bh, bw)).reshape(2, -1).T
        coords = coords * lambda_

        pixels = np.hstack((pixels, coords))

        scaler = StandardScaler()
        pixels = scaler.fit_transform(pixels)

        labels = model.fit_predict(pixels)

        centres = np.zeros((k, bc))
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                centres[cluster_id] = block.reshape(-1, bc)[mask].mean(axis=0)
            else:
                centres[cluster_id] = np.random.randint(0, 255, bc)

        block_segmented = centres[labels].reshape(bh, bw, bc)

        # accumulation
        image_sum[i:i+bh, j:j+bw] += block_segmented
        image_count[i:i+bh, j:j+bw] += 1

        print(i, j, h, w)

# moyenne finale
image_segmentee = (image_sum / image_count).astype(np.uint8)

# 5. Affichage
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.title("Image Originale")
plt.imshow(img_np)

plt.subplot(1, 2, 2)
plt.title("Image Segmentée")
plt.imshow(image_segmentee)

plt.show()