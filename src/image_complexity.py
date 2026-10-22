"""Image complexity score based on persistent entropy (topological data analysis).

Patches are extracted from the grayscale image, a Vietoris-Rips persistence diagram is
computed on a random sample of patches, and its normalised persistent entropy is used as
the complexity score. In this project, scores were used to split test images into
low / medium / high complexity groups.

Usage:
    python src/image_complexity.py --image path/to/mri.jpg
"""
import argparse

import numpy as np
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


def extract_patches(img, kernel_size, stride, padding):
    if img.ndim < 3:
        img = np.expand_dims(img, axis=-1)
    img_padded = np.pad(img, [(padding, padding), (padding, padding), (0, 0)], mode="constant")
    patches = extract_patches_2d(img_padded, (kernel_size, kernel_size))
    patches = patches[::stride]
    return patches.reshape(patches.shape[0], -1)


def persistent_entropy(patches, n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    sample = patches[rng.permutation(len(patches))[:n_samples]]

    diagrams = VietorisRipsPersistence().fit_transform([sample])
    n_distinct = np.sum([pt[-1] == 0 for pt in diagrams[0]])
    norm = np.log2(min(n_distinct, len(sample))) + 1e-5

    pe = PersistenceEntropy(normalize=False, nan_fill_value=norm)
    return float(pe.fit_transform(diagrams)[0, 0] / norm)


def image_complexity(image_path, kernel_size=16, stride=2, padding=4, n_samples=200):
    img = np.array(Image.open(image_path).convert("L"))
    patches = extract_patches(img, kernel_size, stride, padding)
    return persistent_entropy(patches, n_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--kernel-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    score = image_complexity(args.image, args.kernel_size, args.stride, n_samples=args.n_samples)
    print(f"Complexity (persistent entropy): {score:.4f}")


if __name__ == "__main__":
    main()
