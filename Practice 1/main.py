import cv2
import matplotlib.pyplot as plt

# зчитування зображення без змін
img = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)


# розміри: (висота, ширина, канали)
print("Розміри зображення:", img.shape)

# спосіб 1: ручний розрахунок об'єму
height, width, channels = img.shape
volume1 = height * width * channels
print("Об'єм (спосіб 1):", volume1, "байт")

# спосіб 2: через numpy
volume2 = img.size * img.itemsize
print("Об'єм (спосіб 2):", volume2, "байт")

# конвертація BGR → RGB для коректного відображення
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# показ оригінального зображення
plt.imshow(img_rgb)
plt.title("Оригінальне зображення")
plt.axis('off')
plt.show()

# перевід у відтінки сірого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# створення двох підграфіків
fig, axes = plt.subplots(1, 2)

# стандартне сіре зображення
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Звичайне сіре")
axes[0].axis('off')

# сіре з ручним масштабуванням (контраст)
axes[1].imshow(gray, cmap='gray', vmin=100, vmax=200)
axes[1].set_title("З масштабуванням")
axes[1].axis('off')

plt.show()  # відображення обох зображень

# збереження з різною якістю JPEG
cv2.imwrite('img_q75.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
cv2.imwrite('img_q30.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])
cv2.imwrite('img_q0.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 0])