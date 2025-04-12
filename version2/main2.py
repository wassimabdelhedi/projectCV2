from ultralytics import YOLO
import cv2
from collections import Counter

# Charger le modèle YOLO
model = YOLO('yolov8n.pt')  # Modèle rapide et léger

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 = webcam par défaut

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des objets sur le frame courant
    results = model(frame)

    detected_classes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detected_classes.append(class_name)

    # Compter les objets détectés
    counter = Counter(detected_classes)

    # Générer un commentaire texte selon les objets détectés
    if len(counter) == 0:
        comment = "Searching for objects..."
    else:
        comment = "Detected: " + ", ".join([f"{obj}({count})" for obj, count in counter.items()])

    # Dessiner les annotations sur l'image
    annotated_frame = results[0].plot()

    # Ajouter le commentaire textuel
    cv2.putText(annotated_frame, comment, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Afficher le résultat en temps réel
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Sortir en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
