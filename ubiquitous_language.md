# Ubiquitous Language (Linguaggio Ubiquitario)
Questo documento definisce il vocabolario comune e i termini specifici adottati in questo progetto per garantire coerenza terminologica tra codice, commenti, notebook, documentazione e discussioni.

---

## Termini di Dominio e Dati

### 1. Emozione (Emotion)
- **Definizione**: Lo stato affettivo interno che cerchiamo di classificare.
- **Classi standard (FER-2013)**: 
  - *Rabbia (Angry)*
  - *Disgusto (Disgust)*
  - *Paura (Fear)*
  - *Felicità (Happy)*
  - *Tristezza (Sad)*
  - *Sorpresa (Surprise)*
  - *Neutro (Neutral)*
- **Nel codice**: Rappresentata come un valore intero da `0` a `6` (etichetta o `label`) o come stringa con il nome dell'emozione in italiano o inglese.

### 2. Espressione Facciale (Facial Expression)
- **Definizione**: La manifestazione visiva dell'emozione sul volto. Rappresenta l'input visivo grezzo del nostro sistema.

### 3. Volto / Faccia (Face / ROI - Region of Interest)
- **Definizione**: La sotto-porzione rettangolare dell'immagine che contiene il viso rilevato. Viene individuata da un Face Detector, ritagliata, convertita in scala di grigi e ridimensionata a `48x48` pixel prima di essere passata ai modelli di classificazione.

### 4. Rilevatore di Volti (Face Detector)
- **Definizione**: L'algoritmo preliminare che individua le coordinate del volto nell'immagine (es. Haar Cascades di OpenCV).

---

## Termini Metodologici e Algoritmici

### 5. Approccio Classico (Classical Approach)
- **Definizione**: Il metodo basato su feature estratte manualmente (Hand-crafted features) inserite in un classificatore statistico tradizionale.
- **Componenti**: Estrazione di HOG e LBP + Classificatore SVM.

### 6. Approccio Deep Learning (Deep Learning Approach)
- **Definizione**: Il metodo in cui le feature e la classificazione vengono apprese congiuntamente da una rete neurale convoluzionale.
- **Componenti**: Rete Neurale Convoluzionale (CNN) in PyTorch.

### 7. HOG (Histogram of Oriented Gradients)
- **Definizione**: Feature che descrive la forma geometrica e i contorni del volto calcolando la distribuzione delle direzioni del gradiente dell'intensità luminosa.

### 8. LBP (Local Binary Patterns)
- **Definizione**: Feature che descrive la texture e i micro-pattern locali della pelle (es. rughe d'espressione), confrontando ogni pixel con i suoi vicini.

### 9. SVM (Support Vector Machine)
- **Definizione**: Algoritmo di classificazione supervisionato utilizzato per trovare l'iperpiano ottimale che separa le classi di emozioni nel canale delle feature classiche (HOG + LBP).

### 10. CNN (Convolutional Neural Network)
- **Definizione**: Rete neurale profonda contenente livelli convoluzionali progettati per estrarre automaticamente pattern spaziali e gerarchici dalle immagini dei volti.

---

## Termini di Valutazione e Prestazioni

### 11. accuratezza (Accuracy)
- **Definizione**: La frazione di predizioni corrette sul totale delle immagini valutate.

### 12. F1-Score (per Classe)
- **Definizione**: La media armonica tra precisione (Precision) e richiamo (Recall) per ogni singola emozione, fondamentale per valutare le prestazioni su classi sbilanciate (es. Disgusto, tipicamente poco rappresentata).

### 13. Matrice di Confusione (Confusion Matrix)
- **Definizione**: Tabella che mostra le predizioni del modello rispetto alle etichette reali, utile per individuare quali emozioni vengono maggiormente confuse tra loro (es. Tristezza e Neutro, o Paura e Sorpresa).

### 14. Latenza di Inferenza (Inference Latency)
- **Definizione**: Il tempo (espresso in millisecondi, ms) impiegato da un modello per elaborare un volto ed emettere la predizione dell'emozione.

### 15. FPS (Frames Per Second)
- **Definizione**: Frequenza dei fotogrammi elaborati al secondo nella demo in tempo reale. Rappresenta la fluidità dell'applicazione sul computer dell'utente.
