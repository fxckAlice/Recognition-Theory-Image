from skimage import io, color
from skimage.filters import gaussian, sobel, unsharp_mask
import matplotlib.pyplot as plt

# 1. Зчитування зображення
image = io.imread("img.png")

# Переведення у відтінки сірого

if image.shape[-1] == 4:
    image = image[:, :, :3]

gray = color.rgb2gray(image)

# ---------------- GAUSSIAN FILTER ----------------

sigmas = [1, 3, 5]

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, sigma in enumerate(sigmas):
    filtered = gaussian(gray, sigma=sigma)
    plt.subplot(1, 4, i + 2)
    plt.imshow(filtered, cmap="gray")
    plt.title(f"Gaussian sigma={sigma}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("gaussian_results.png")
plt.show()

# ---------------- SOBEL FILTER ----------------

edges = sobel(gray)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Sobel edges")
plt.axis("off")

plt.tight_layout()
plt.savefig("sobel_results.png")
plt.show()

# ---------------- UNSHARP MASKING ----------------

params = [
    (10, 1),
    (40, 3),
    (90, 5)
]

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, (radius, amount) in enumerate(params):
    sharp = unsharp_mask(gray, radius=radius, amount=amount)
    plt.subplot(1, 4, i + 2)
    plt.imshow(sharp, cmap="gray")
    plt.title(f"r={radius}, a={amount}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("unsharp_results.png")
plt.show()