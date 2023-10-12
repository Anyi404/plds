#cuarto semestre

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Pulgar
thumb_points = [1, 2, 4]
# Índice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Colores
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        fingers_counter = ["_", "_"]
        thickness = [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])

                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)

                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)
                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counter[idx] = str(np.count_nonzero(fingers == True))
                for i, finger in enumerate(fingers):
                    if finger == True:
                        thickness[idx][i] = -1

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Mostrar el número de dedos en la pantalla
                cv2.putText(frame, f"Fingers: {fingers_counter[idx]}", (20 + idx * 350, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualización
        cv2.rectangle(frame, (0, 0), (160, 80), (125, 220, 0), -1)
        cv2.putText(frame, f"Left: {fingers_counter[0]}", (15, 65), 1, 5, (255, 255, 255), 2)
        cv2.putText(frame, f"Right: {fingers_counter[1]}", (120, 65), 1, 5, (255, 255, 255), 2)

        for idx, (hand_color, hand_thickness) in enumerate(zip([PEACH, PURPLE], [thickness[0], thickness[1]])):
            # Pulgar
            cv2.rectangle(frame, (160 + idx * 200, 10), (210 + idx * 200, 60), hand_color, hand_thickness[0])
            cv2.putText(frame, "Pulgar", (160 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Índice
            cv2.rectangle(frame, (220 + idx * 200, 10), (270 + idx * 200, 60), hand_color, hand_thickness[1])
            cv2.putText(frame, "Indice", (220 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Medio
            cv2.rectangle(frame, (280 + idx * 200, 10), (330 + idx * 200, 60), hand_color, hand_thickness[2])
            cv2.putText(frame, "Medio", (280 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Anular
            cv2.rectangle(frame, (340 + idx * 200, 10), (390 + idx * 200, 60), hand_color, hand_thickness[3])
            cv2.putText(frame, "Anular", (340 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Menique
            cv2.rectangle(frame, (400 + idx * 200, 10), (450 + idx * 200, 60), hand_color, hand_thickness[4])
            cv2.putText(frame, "Menique", (400 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
