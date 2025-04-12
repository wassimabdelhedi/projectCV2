from ultralytics import YOLO
import cv2
from collections import Counter

# Charger un modèle pré-entrainé
model = YOLO('yolov8n.pt')  # Version rapide et légère

# Charger image
image_path = "C:\\GI1 S2\\projectCV2\\version1\\coins.jpg"

image = cv2.imread(image_path)

# Prédiction
results = model(image_path)

# Récupérer les noms des classes détectées
detected_classes = []

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]
        detected_classes.append(class_name)

# Compter le nombre d'occurrences par objet
counter = Counter(detected_classes)

# Afficher les résultats
for obj, count in counter.items():
    print(f"{obj}: {count}")

# Affichage image annotée
annotated_image = results[0].plot()
cv2.imshow("Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
