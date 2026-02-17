import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


class DetectHands():
    def __init__(self, max_hands=2, trust_detection=0.5, trust_track=0.5, color_points=(0,0,255), color_connections=(0,255,0)):
        self.max_hands = max_hands
        self.trust_detection = trust_detection
        self.trust_track = trust_track
        self.color_points = color_points
        self.color_connections = color_connections
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.trust_detection,
            min_tracking_confidence=self.trust_track,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def find_hands(self, img, draw=True):
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # Polegar
            (0, 5), (5, 6), (6, 7), (7, 8),    # Indicador
            (5, 9), (9, 10), (10, 11), (11, 12), # Médio
            (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Mínimo e Palma
        ]

        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        timestamp_ms = int(time.time() * 1000)
        self.result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if draw and self.result.hand_landmarks:
            for idx, landmarks in enumerate(self.result.hand_landmarks):
                # 1. Desenhar conexões
                for connection in HAND_CONNECTIONS:
                    p1, p2 = landmarks[connection[0]], landmarks[connection[1]]
                    cx1, cy1 = int(p1.x * w), int(p1.y * h)
                    cx2, cy2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(img, (cx1, cy1), (cx2, cy2), self.color_connections, 2)

                # 2. Desenhar pontos e Handedness (Esquerda/Direita)
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, self.color_points, cv2.FILLED)

                if self.result.handedness:
                    handedness = self.result.handedness[idx][0]
                    label = handedness.category_name
                    tx, ty = int(landmarks[0].x * w), int(landmarks[0].y * h) - 20
                    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img
    
    def find_points(self, img, num_hand=0, draw=True):
        self.points_list = []
        if self.result.hand_landmarks:
            my_hand = self.result.hand_landmarks[num_hand]
            h, w, c = img.shape
            
            for id, lm in enumerate(my_hand):
                # Convertendo coordenadas normalizadas para pixels
                cx, cy = int(lm.x * w), int(lm.y * h)
                # O Z continua sendo um valor flutuante de profundidade
                cz = lm.z 
                self.points_list.append([id, cx, cy, cz])
            
        return self.points_list
    
    def fingers_up(self):
        fingers = []
        # IDs das pontas dos dedos: Polegar(4), Indicador(8), Médio(12), Anelar(16), Mínimo(20)
        tip_ids = [4, 8, 12, 16, 20]
        
        if self.result.hand_landmarks:
            landmarks = self.result.hand_landmarks[0] # Focando na primeira mão detectada

            # --- Lógica para o Polegar ---
            # No polegar, comparamos o X (horizontal) em vez do Y, 
            # verificando se ele está "aberto" para o lado.
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers.append(1) # Aberto
            else:
                fingers.append(0) # Fechado

            # --- Lógica para os outros 4 dedos (Indicador ao Mínimo) ---
            for id in range(1, 5):
                # No MediaPipe, o Y cresce para baixo. 
                # Então, se o Y da ponta (tip) for MENOR que o Y da junta anterior (pip),
                # significa que o dedo está mais "alto", ou seja, levantado.
                if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
        return fingers 
    
    def get_distance(self, p1_id, p2_id, img):
        if self.result.hand_landmarks:
            h, w, _ = img.shape
            lm = self.result.hand_landmarks[0]
            
            # Coordenadas em pixels
            x1, y1 = int(lm[p1_id].x * w), int(lm[p1_id].y * h)
            x2, y2 = int(lm[p2_id].x * w), int(lm[p2_id].y * h)
            
            # Teorema de Pitágoras: d = sqrt((x2-x1)^2 + (y2-y1)^2)
            distance = math.hypot(x2 - x1, y2 - y1)
            
            return distance, (x1, y1), (x2, y2)
        return 0, (0,0), (0,0)

    def close(self):
        self.detector.close()

def main():
    cap = cv2.VideoCapture(0)
    deteccao = DetectHands()

    while cap.isOpened():
        success, img = cap.read()
        if not success: break

        # img = cv2.flip(img, 1)
        # O img aqui é o retorno da função find_hands
        img = deteccao.find_hands(img, draw=True)

        # lista_pontos = deteccao.find_points(img,0, True)
        # print(lista_pontos)

        dedos = deteccao.fingers_up()

        if dedos:
            if dedos == [1, 1, 1, 1, 1]:
                cv2.putText(img, "PAPEL", (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


            elif dedos == [0, 1, 1, 0, 0]:
                cv2.putText(img, "TESOURA", (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


            elif dedos == [0, 0, 0, 0, 0]:
                cv2.putText(img, "PEDRA", (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # dist, p1, p2 = deteccao.get_distance(4, 8, img)
        # if dist < 30: # Se a distância for menor que 30 pixels
        #     cv2.circle(img, p1, 15, (255, 0, 255), cv2.FILLED)
        #     print("CLICK!")

        cv2.imshow('MediaPipe Tasks', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    deteccao.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()