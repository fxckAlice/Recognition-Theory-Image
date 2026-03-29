import cv2
import numpy as np
import matplotlib.pyplot as plt

# зчитування напівтонового зображення
img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# перевірка, чи файл успішно завантажився
if img is None:
    raise FileNotFoundError("Не вдалося знайти або зчитати файл зображення")

# -------------------------------
# 1. Оригінальне зображення і його гістограма
# -------------------------------

# обчислення гістограми через OpenCV
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Оригінальне зображення')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(np.arange(256), hist[:, 0], width=1.0)
plt.title('Гістограма оригіналу')
plt.xlim([0, 255])

plt.tight_layout()
plt.show()

# -------------------------------
# 2. Еквалізація гістограми
# -------------------------------

# вирівнювання гістограми
eq_img = cv2.equalizeHist(img)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Оригінальне зображення')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма оригіналу')

plt.subplot(2, 2, 3)
plt.imshow(eq_img, cmap='gray')
plt.title('Еквалізоване зображення')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(eq_img.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма після еквалізації')

plt.tight_layout()
plt.show()

# -------------------------------
# 3. Функція гамма-корекції
# -------------------------------

def gamma_correction(img, gamma=1.0, in_range=(0, 1), out_range=(0, 1)):
    # нормалізація зображення до діапазону [0, 1]
    img_norm = img.astype(np.float32) / 255.0

    # обмеження вхідного діапазону
    low_in, high_in = in_range
    img_clipped = np.clip(img_norm, low_in, high_in)

    # перенесення вхідного діапазону до [0, 1]
    if high_in - low_in == 0:
        img_scaled = np.zeros_like(img_clipped)
    else:
        img_scaled = (img_clipped - low_in) / (high_in - low_in)

    # гамма-корекція яскравості
    img_gamma = np.power(img_scaled, gamma)

    # перенесення у вихідний діапазон
    low_out, high_out = out_range
    img_out = low_out + img_gamma * (high_out - low_out)

    # повернення до діапазону [0, 255]
    img_out = np.clip(img_out, 0, 1)
    return (img_out * 255).astype(np.uint8)

# -------------------------------
# 4. Гамма-корекція при gamma = 1, 0.5, 2.1
# in_range=(0,1), out_range=(0,1)
# -------------------------------

img_g1 = gamma_correction(img, gamma=1.0, in_range=(0, 1), out_range=(0, 1))
img_g05 = gamma_correction(img, gamma=0.5, in_range=(0, 1), out_range=(0, 1))
img_g21 = gamma_correction(img, gamma=2.1, in_range=(0, 1), out_range=(0, 1))

plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.imshow(img_g1, cmap='gray')
plt.title('Gamma = 1')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.hist(img_g1.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма Gamma = 1')

plt.subplot(3, 2, 3)
plt.imshow(img_g05, cmap='gray')
plt.title('Gamma = 0.5')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.hist(img_g05.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма Gamma = 0.5')

plt.subplot(3, 2, 5)
plt.imshow(img_g21, cmap='gray')
plt.title('Gamma = 2.1')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.hist(img_g21.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма Gamma = 2.1')

plt.tight_layout()
plt.show()

# -------------------------------
# 5. Гамма-корекція з регулюванням діапазонів
# gamma = 1
# -------------------------------

# стискання вихідного діапазону яскравості
img_out_02_06 = gamma_correction(img, gamma=1.0, in_range=(0, 1), out_range=(0.2, 0.6))

# розтягнення вхідного піддіапазону
img_in_02_06 = gamma_correction(img, gamma=1.0, in_range=(0.2, 0.6), out_range=(0, 1))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_out_02_06, cmap='gray')
plt.title('in=(0,1), out=(0.2,0.6)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img_out_02_06.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма out=(0.2,0.6)')

plt.subplot(2, 2, 3)
plt.imshow(img_in_02_06, cmap='gray')
plt.title('in=(0.2,0.6), out=(0,1)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(img_in_02_06.ravel(), bins=256, range=(0, 256))
plt.title('Гістограма in=(0.2,0.6)')

plt.tight_layout()
plt.show()