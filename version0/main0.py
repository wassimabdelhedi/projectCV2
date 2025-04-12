import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:\\GI1 S2\\projectCV2\\version0\\coins.jpg")
output = image.copy()

# Prétraitement
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Seuillage Otsu
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphologie légère pour corriger les défauts
kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Détection des contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Compter les objets
num_objects = len(contours)
print(f"Nombre d'objets détectés : {num_objects}")

# Dessiner les résultats
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, f"Objet {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Afficher le nombre total d'objets sur l'image
cv2.putText(output, f"Total Objets : {num_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Affichage
cv2.imshow("Image Originale", image)
cv2.imshow("Image Seuillage", closing)
cv2.imshow("Detection d'objets", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
