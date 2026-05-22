#!/usr/bin/env python3
"""
Facial Expression Recognition - Real-time Webcam Demo
Consente di valutare in tempo reale l'approccio classico (SVM + HOG + LBP)
e l'approccio Deep Learning (CNN custom in PyTorch).

Controlli:
- Premere 'c' per selezionare la CNN (PyTorch)
- Premere 's' per selezionare l'SVM (HOG + LBP)
- Premere 'q' o ESC per uscire
"""

import os
import cv2
import time
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.feature import hog, local_binary_pattern

# =====================================================================
# 1. Definizione dell'Architettura CNN (deve essere identica a quella addestrata)
# =====================================================================
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.fc2(x)
        return x

# =====================================================================
# 2. Estrazione Feature Classiche (identica a notebook 2 e 4)
# =====================================================================
def extract_classical_features(img):
    hog_feats = hog(
        img, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=False
    )
    
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    
    cell_size = 8
    lbp_feats = []
    for i in range(0, 48, cell_size):
        for j in range(0, 48, cell_size):
            cell = lbp[i:i+cell_size, j:j+cell_size]
            hist, _ = np.histogram(cell, bins=10, range=(0, 10))
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            lbp_feats.extend(hist)
            
    return np.concatenate([hog_feats, lbp_feats])

# Mappatura etichette
EMOTIONS = {
    0: "Rabbia (Angry)",
    1: "Disgusto (Disgust)",
    2: "Paura (Fear)",
    3: "Felicita (Happy)",
    4: "Tristezza (Sad)",
    5: "Sorpresa (Surprise)",
    6: "Neutro (Neutral)"
}

def main():
    print("Inizializzazione Demo in tempo reale...")
    
    # -----------------------------------------------------------------
    # Percorsi dei file
    # -----------------------------------------------------------------
    models_dir = "models"
    svm_file = os.path.join(models_dir, "svm_model.pkl")
    scaler_file = os.path.join(models_dir, "svm_scaler.pkl")
    cnn_file = os.path.join(models_dir, "cnn_model.pth")
    
    # Verifica esistenza file dei modelli
    if not (os.path.exists(svm_file) and os.path.exists(scaler_file) and os.path.exists(cnn_file)):
        print(f"ERRORE: Uno o piu file dei modelli non sono stati trovati in '{models_dir}/'.")
        print("Assicurati di aver eseguito con successo i notebook 2 e 3 prima di lanciare la demo.")
        return

    # -----------------------------------------------------------------
    # Caricamento dei Modelli
    # -----------------------------------------------------------------
    print("Caricamento dei modelli in corso...")
    
    # SVM & Scaler
    svm_model = joblib.load(svm_file)
    scaler = joblib.load(scaler_file)
    print("-> SVM & Scaler caricati con successo.")
    
    # CNN (Target GPU se disponibile, altrimenti CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"-> Utilizzo dispositivo Deep Learning: {device}")
    
    cnn_model = EmotionCNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_file, map_location=device))
    cnn_model.eval()
    print("-> CNN caricata con successo.")
    
    # Transform per CNN
    cnn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # -----------------------------------------------------------------
    # Inizializzazione Rilevatore Volti (OpenCV Haar Cascade)
    # -----------------------------------------------------------------
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"ERRORE: Impossibile caricare il classificatore Haar Cascade da {cascade_path}")
        return
    print("-> Haar Cascade caricato correttamente.")

    # -----------------------------------------------------------------
    # Apertura Webcam
    # -----------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRORE: Impossibile aprire la webcam (ID 0).")
        print("Verifica i permessi di accesso alla fotocamera per il terminale/IDE.")
        return
    print("-> Webcam attivata. Premere 'q' o ESC per chiudere la finestra.")

    # Stato del modello attivo: 'cnn' o 'svm'
    active_mode = "cnn"
    
    # Variabili per calcolo FPS
    prev_time = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERRORE: Impossibile leggere il frame della webcam.")
            break

        # Specchia il frame orizzontalmente per un'interazione naturale
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rilevamento dei volti nel frame grayscale
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),  # Impostiamo una dimensione minima ragionevole per evitare falsi positivi
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Estrazione del ROI (Region of Interest) del volto in scala di grigi
            face_roi_gray = gray[y:y+h, x:x+w]
            
            # Ridimensionamento a 48x48 (dimensione input FER-2013)
            face_resized = cv2.resize(face_roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Esecuzione dell'inferenza in base al modello attivo
            label_text = "N/A"
            confidence = 0.0
            
            if active_mode == "cnn":
                # Preprocessing ed esecuzione CNN
                img_pil = Image.fromarray(face_resized)
                tensor = cnn_transform(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = cnn_model(tensor)
                    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                    
                pred_class = np.argmax(probabilities)
                confidence = probabilities[pred_class]
                label_text = EMOTIONS[pred_class]
                
            elif active_mode == "svm":
                # Estrazione feature, scaling ed esecuzione SVM
                feats = extract_classical_features(face_resized).reshape(1, -1)
                feats_scaled = scaler.transform(feats)
                
                # Predizione classe
                pred_class = svm_model.predict(feats_scaled)[0]
                
                # Calcolo pseudo-probabilità (softmax sui punteggi della decision function)
                decision_scores = svm_model.decision_function(feats_scaled)[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))  # Stabilità numerica
                probabilities = exp_scores / np.sum(exp_scores)
                
                confidence = probabilities[pred_class]
                label_text = EMOTIONS[pred_class]

            # ---------------------------------------------------------
            # Disegno del Bounding Box e delle Informazioni
            # ---------------------------------------------------------
            # Colori in BGR:
            # - CNN: Arancione Brillante (40, 120, 250)
            # - SVM: Blu/Azzurro (230, 160, 40)
            color = (40, 120, 250) if active_mode == "cnn" else (230, 160, 40)
            
            # Disegna il box principale
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Disegna una barra di sfondo per l'etichetta sopra il volto
            label_bg_y = max(0, y - 25)
            cv2.rectangle(frame, (x, label_bg_y), (x + w, y), color, cv2.FILLED)
            
            # Testo etichetta + accuratezza
            display_text = f"{label_text.split()[0]} ({confidence*100:.1f}%)"
            
            # Posiziona il testo
            cv2.putText(
                frame, 
                display_text, 
                (x + 5, y - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )

        # -------------------------------------------------------------
        # Calcolo e Visualizzazione FPS
        # -------------------------------------------------------------
        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0

        # -------------------------------------------------------------
        # Pannello di Controllo e Overlay GUI (Top-Left)
        # -------------------------------------------------------------
        # Sfondo per il pannello di controllo
        cv2.rectangle(frame, (10, 10), (330, 110), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (10, 10), (330, 110), (100, 100, 100), 1)
        
        # Titolo del pannello
        cv2.putText(
            frame, 
            "RICONOSCIMENTO EMOZIONI", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        # Modello Attivo
        mode_color = (40, 120, 250) if active_mode == "cnn" else (230, 160, 40)
        mode_text = "CNN (PyTorch)" if active_mode == "cnn" else "SVM (HOG+LBP)"
        cv2.putText(
            frame, 
            f"Modello: {mode_text}", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.45, 
            mode_color, 
            1, 
            cv2.LINE_AA
        )
        
        # FPS
        cv2.putText(
            frame, 
            f"Frequenza: {fps:.1f} FPS", 
            (20, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.45, 
            (200, 200, 200), 
            1, 
            cv2.LINE_AA
        )
        
        # Istruzioni hotkey
        cv2.putText(
            frame, 
            "Cambia: [c] CNN | [s] SVM | [q] Esci", 
            (20, 95), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (150, 150, 150), 
            1, 
            cv2.LINE_AA
        )

        # Mostra il frame
        cv2.imshow("Riconoscimento Espressioni Facciali - Demo", frame)

        # Gestione input da tastiera
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o tasto ESC
            break
        elif key == ord('c'):
            active_mode = "cnn"
            print("-> Switch a modello: CNN (Deep Learning)")
        elif key == ord('s'):
            active_mode = "svm"
            print("-> Switch a modello: SVM (HOG+LBP)")

    # Rilascia le risorse al termine
    cap.release()
    cv2.destroyAllWindows()
    print("Demo chiusa correttamente.")

if __name__ == "__main__":
    main()
