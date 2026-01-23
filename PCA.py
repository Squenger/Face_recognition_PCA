# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:19:10 2026

@author: beddo
"""


import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image





def load_dataset(root, target_size=(168, 192)):
    images = []
    labels = []

    for person in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person)

        if not os.path.isdir(person_dir):
            continue

        # Extraction robuste du label (fonctionne pour s1 et yaleB01)
        digits = ''.join(filter(str.isdigit, person))
        if not digits: continue
        label = int(digits) - 1

        for img_name in sorted(os.listdir(person_dir)):
            if not img_name.lower().endswith(".pgm"):
                continue

            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert("L")
                if target_size:
                    img = img.resize(target_size)
                images.append(np.array(img).flatten())
                labels.append(label)
            except Exception as e:
                print(f"Erreur chargement {img_path}: {e}")

    return np.array(images), np.array(labels)



def compute_eigenfaces(X, K):
    # Expect X shape: (nb_images, N) where each row is a flattened image.
    # Compute mean face (length N) across the image rows.
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    # Perform SVD on the transposed centered data so principal components
    # are vectors of length N (same as an image). U will have shape (N, nb_images).
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)

    # Take first K principal components (columns of U) and return them
    # as rows so eigenfaces has shape (K, N).
    eigenfaces = U[:, :K].T
    return mean_face, eigenfaces


def plot_eigenfaces_and_mean(mean_face, eigenfaces, h, w, K=10):
    plt.figure(figsize=(15, 5))
    
    # Plot mean face
    plt.subplot(1, K + 1, 1)
    plt.imshow(mean_face.reshape(h, w), cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")

    # Plot top K eigenfaces
    for i in range(K):
        plt.subplot(1, K + 1, i + 2)
        plt.imshow(eigenfaces[i].reshape(h, w), cmap="gray")
        plt.title(f"EF {i+1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()



def project(face, mean_face, eigenfaces):
    return eigenfaces @ (face - mean_face)



def recognize(test_face, projections, labels, mean_face, eigenfaces):
    test_proj = project(test_face, mean_face, eigenfaces)
    distances = np.linalg.norm(projections - test_proj, axis=1)

    idx = np.argmin(distances)
    return labels[idx], distances[idx]



def reconstruct_face(weights, mean_face, eigenfaces):
    return mean_face + eigenfaces.T @ weights


def recognize_face(test_face, projections, mean_face, eigenfaces, threshold=None):
    test_proj = project(test_face, mean_face, eigenfaces)

    distances = np.linalg.norm(projections - test_proj, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    if threshold is not None and min_dist > threshold:
        return None, min_dist  # visage inconnu

    return min_idx, min_dist


def recognize_class(test_face, class_centers, mean_face, eigenfaces):
    test_proj = project(test_face, mean_face, eigenfaces)
    distances = np.linalg.norm(class_centers - test_proj, axis=1)
    return np.argmin(distances), np.min(distances)

def train_test_split(X, y, n_train=5):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_train.extend(X[train_idx])
        y_train.extend(y[train_idx])
        X_test.extend(X[test_idx])
        y_test.extend(y[test_idx])

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test)
    )


# Charger les données
X, y = load_dataset("CroppedYale")

# Calcul PCA
K = 50  # typiquement 40–100
mean_face, eigenfaces = compute_eigenfaces(X, K)

# Visualisation du visage moyen et des eigenfaces
h, w = 192, 168  # taille de l'image (CroppedYale)
plot_eigenfaces_and_mean(mean_face, eigenfaces, h, w, K=10)


projections = np.array([
    project(x, mean_face, eigenfaces)
    for x in X
])



# X : (nb_images, N)
# mean_face : (N,)
# eigenfaces : (K, N)

idx = 0  # index du visage à reconstruire

original = X[idx]

# projection
weights = eigenfaces @ (original - mean_face)

# reconstruction
reconstructed = reconstruct_face(weights, mean_face, eigenfaces)



class_centers = []

for label in np.unique(y):
    class_centers.append(np.mean(projections[y == label], axis=0))

class_centers = np.array(class_centers)

# plt.figure()
# plt.imshow(X[300].reshape(h,w))
# plt.show()

# %% Reconstruction de visage

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original.reshape(h, w), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Reconstruit (K={eigenfaces.shape[0]})")
plt.imshow(reconstructed.reshape(h, w), cmap="gray")
plt.axis("off")

plt.show()


for K in [5, 10, 20, 40]:
    eigenfaces_k = eigenfaces[:K]
    weights = eigenfaces_k @ (original - mean_face)
    recon = mean_face + eigenfaces_k.T @ weights

    plt.figure()
    plt.title(f"K = {K}")
    plt.imshow(recon.reshape(h, w), cmap="gray")
    plt.axis("off")
    plt.show()



# %% Test de reonnaissance 



test_path = "CroppedYale/yaleB10/yaleB10_P00A+000E+00.pgm"

test_img = Image.open(test_path).convert("L")
test_face = np.array(test_img).flatten()

pred_label, dist = recognize_class(
    test_face,
    class_centers,
    mean_face,
    eigenfaces
)


print("Personne reconnue : Sujet", pred_label + 1)

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Visage test")
plt.imshow(test_face.reshape(h, w), cmap="gray")
plt.axis("off")

# on prend une image représentative de la classe reconnue
recognized_face = X[y == pred_label][0]

plt.subplot(1, 2, 2)
plt.title(f"Reconnu : Sujet{pred_label + 1}")
plt.imshow(recognized_face.reshape(h, w), cmap="gray")
plt.axis("off")

plt.show()

# %%  calcul du taux de reconaissance

X_train, y_train, X_test, y_test = train_test_split(X, y)

mean_face, eigenfaces = compute_eigenfaces(X_train, K)

train_proj = np.array([
    project(x, mean_face, eigenfaces)
    for x in X_train
])

class_centers = []

for label in np.unique(y_train):
    class_centers.append(np.mean(train_proj[y_train == label], axis=0))

class_centers = np.array(class_centers)


correct = 0

for face, true_label in zip(X_test, y_test):
    pred_label, _ = recognize_class(
        face,
        class_centers,
        mean_face,
        eigenfaces
    )
    if pred_label == true_label:
        correct += 1

accuracy = correct / len(y_test)
print(f"Taux de reconnaissance : {accuracy * 100:.2f} %")

