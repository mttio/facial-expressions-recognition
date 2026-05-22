# Facial Expression Recognition (Riconoscimento delle Espressioni Facciali)
Questo documento descrive gli obiettivi, le scelte metodologiche, l'architettura e l'organizzazione del progetto per il riconoscimento delle emozioni tramite espressioni facciali.

---

## 1. Obiettivo del Progetto
Realizzare un sistema in grado di classificare le emozioni umane espresse tramite espressioni facciali. Il progetto prevede una ricerca dello stato dell'arte e il confronto tra due approcci metodologici differenti:
1. **Approccio Classico di Computer Vision**: Estrazione di feature geometriche/tessiturali fatte a mano (Hand-crafted features) accoppiate con un classificatore supervisionato tradizionale.
2. **Approccio Deep Learning**: Progettazione e addestramento da zero di una Rete Neurale Convoluzionale (CNN) per apprendere direttamente le feature dalle immagini.

---

## 2. Dataset: FER-2013
Utilizzeremo il dataset standard di riferimento **FER-2013** (Facial Expression Recognition 2013), composto da:
- **Immagini**: ~35.887 immagini in scala di grigi di dimensione 48x48 pixel, già ritagliate sui volti.
- **7 Classi di Emozioni**: 
  - 0: Rabbia (Angry)
  - 1: Disgusto (Disgust)
  - 2: Paura (Fear)
  - 3: Felicità (Happy)
  - 4: Tristezza (Sad)
  - 5: Sorpresa (Surprise)
  - 6: Neutro (Neutral)
- **Suddivisione**: Training set (~28.7k), Validation set (PublicTest, ~3.5k) e Test set (PrivateTest, ~3.5k).

*Nota: Scaricheremo il dataset programmaticamente (tramite Hugging Face Datasets o link diretto) per semplificare l'installazione senza richiedere chiavi API di Kaggle.*

---

## 3. Descrizione dei Due Approcci

### Approccio A: Classico (HOG + LBP + SVM)
- **Preprocessing**: Normalizzazione del contrasto (es. equalizzazione dell'istogramma CLAHE).
- **Estrazione Feature**:
  - **HOG (Histogram of Oriented Gradients)**: Cattura i gradienti e i contorni, utile per la forma generale del volto, degli occhi, della bocca.
  - **LBP (Local Binary Patterns)**: Cattura le informazioni sulla texture della pelle e le micro-rughe d'espressione.
  - *Concatenazione*: Unione dei vettori HOG e LBP per formare una feature rappresentativa per ogni volto.
- **Classificazione**: **SVM (Support Vector Machine)** lineare o con kernel RBF, ottimizzata tramite Grid Search / cross-validation su Scikit-Learn.

### Approccio B: Deep Learning (Custom CNN in PyTorch)
- **Framework**: PyTorch.
- **Architettura**: Una CNN custom ottimizzata per input 48x48 (es. 3-4 blocchi convoluzionali con Batch Normalization, Max Pooling, Dropout per evitare l'overfitting, e strati Fully Connected finali).
- **Training**: Addestramento da zero con data augmentation (rotazioni casuali, flip orizzontali, zoom) per migliorare la generalizzazione.
- **Ottimizzazione**: Loss Cross-Entropy, ottimizzatore Adam/SGD, e scheduling del learning rate.

---

## 4. Struttura del Progetto
Il codice sarà organizzato in Jupyter Notebook per le fasi di analisi e addestramento, e in uno script Python per la demo real-time:

```text
riconoscimento_espressioni/
│
├── .venv/                         # Ambiente virtuale Python
├── shared_understanding.md         # Questo file
│
├── data/                          # Cartella per il dataset FER-2013 (scaricato automaticamente)
│
├── models/                        # Modelli salvati (.pkl per SVM, .pth per PyTorch CNN)
│
├── notebooks/                     # Jupyter Notebook per sviluppo e analisi
│   ├── 01_eda_and_preprocessing.ipynb    # Analisi esplorativa dei dati e preprocessing
│   ├── 02_classical_approach.ipynb       # Estrazione HOG+LBP e addestramento/valutazione SVM
│   ├── 03_deep_learning_approach.ipynb  # Definizione, training e valutazione della CNN PyTorch
│   └── 04_comparison_and_discussion.ipynb # Confronto dettagliato (accuratezza, matrici, latenza, dimensioni)
│
└── webcam_demo.py                 # Script per demo in tempo reale da webcam con OpenCV
```

---

## 5. Demo in Tempo Reale (`webcam_demo.py`)
Uno script Python interattivo che:
1. Avvia la webcam tramite OpenCV.
2. Rileva i volti in tempo reale utilizzando un detector leggero (Haar Cascades o MediaPipe).
3. Esegue il ritaglio, la conversione in scala di grigi e il ridimensionamento a 48x48 del volto rilevato.
4. Consente all'utente di selezionare il modello da usare (SVM Classico o CNN Deep Learning).
5. Mostra a schermo il bounding box del volto con l'emozione predetta e la relativa probabilità/confidenza.

---

## 6. Strumenti e Librerie
Il progetto utilizzerà le seguenti tecnologie:
- **Linguaggio**: Python 3.13
- **Image Processing / Feature Extraction**: `opencv-python`, `scikit-image`
- **Machine Learning**: `scikit-learn`
- **Deep Learning**: `torch`, `torchvision` (PyTorch)
- **Data Science / Plotting**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Ambiente**: `jupyter` / `notebook`
