import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.ndimage import rank_filter


def adaptive_median_filter(img, smax=7):
    # Адаптивний медіанний фільтр для grayscale-зображення
    padded = np.pad(img, pad_width=smax // 2, mode='edge')
    result = np.zeros_like(img)

    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            window_size = 3
            pixel_done = False

            while not pixel_done:
                half = window_size // 2
                local = padded[i + smax // 2 - half:i + smax // 2 + half + 1,
                               j + smax // 2 - half:j + smax // 2 + half + 1]

                z_min = int(np.min(local))
                z_max = int(np.max(local))
                z_med = int(np.median(local))
                z_xy = int(padded[i + smax // 2, j + smax // 2])

                a1 = z_med - z_min
                a2 = z_med - z_max

                if a1 > 0 and a2 < 0:
                    # Область B
                    b1 = z_xy - z_min
                    b2 = z_xy - z_max

                    if b1 > 0 and b2 < 0:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    pixel_done = True
                else:
                    window_size += 2
                    if window_size > smax:
                        result[i, j] = z_xy
                        pixel_done = True

    return result.astype(np.uint8)


def show_four(title, images, titles):
    # Допоміжна функція для виводу 4 зображень
    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=14)

    for k in range(4):
        plt.subplot(2, 2, k + 1)
        plt.imshow(images[k], cmap='gray')
        plt.title(titles[k])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 1. Зчитування зображення у градаціях сірого
img = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Не вдалося знайти файл image.jpg. Перевір назву і шлях.")

plt.gray()

# 2. Додавання гаусівського шуму
img_gaussian = random_noise(img, mode='gaussian', var=0.001)
img_gaussian = (255 * img_gaussian).astype(np.uint8)

# 3. Звичайний медіанний фільтр
img_median = cv2.medianBlur(img_gaussian, 3)

# 4. Адаптивний медіанний фільтр
img_adaptive_median = adaptive_median_filter(img_gaussian, smax=7)

# 5. Виведення: оригінал / шум / медіанний / адаптивний медіанний
show_four(
    "Медіанні фільтри",
    [img, img_gaussian, img_median, img_adaptive_median],
    ["Оригінальне", "Гаусівський шум", "Median 3x3", "Adaptive Median Smax=7"]
)

# 6. Фільтри мінімуму і максимуму
img_min = rank_filter(img_gaussian, rank=0, size=3)
img_max = rank_filter(img_gaussian, rank=8, size=3)

# 7. Виведення: оригінал / шум / мінімум / максимум
show_four(
    "Фільтри мінімуму та максимуму",
    [img, img_gaussian, img_min, img_max],
    ["Оригінальне", "Гаусівський шум", "Minimum filter", "Maximum filter"]
)

# 8. Усереднюючі фільтри
img_blur_3 = cv2.blur(img_gaussian, (3, 3))
img_blur_7 = cv2.blur(img_gaussian, (7, 7))

# 9. Виведення: оригінал / шум / blur 3x3 / blur 7x7
show_four(
    "Усереднюючі фільтри",
    [img, img_gaussian, img_blur_3, img_blur_7],
    ["Оригінальне", "Гаусівський шум", "Blur 3x3", "Blur 7x7"]
)