# Riconoscimento delle Espressioni Facciali (Facial Expression Recognition)

Questo progetto ha come obiettivo la classificazione automatica delle emozioni umane espresse tramite il volto, effettuando un confronto sistematico tra l'**approccio classico di Computer Vision** (estrazione manuale di feature geometriche e tessiturali accoppiate a un classificatore supervisionato) e l'**approccio di Deep Learning** (progettazione e addestramento da zero di una Rete Neurale Convoluzionale).

Il progetto è stato sviluppato nell'ambito del corso di **Elaborazione delle Immagini (Image Processing)** presso il **Politecnico di Torino**.

---

## 🛠️ Tecnologie e Librerie
Il progetto è realizzato interamente in **Python 3.13** e utilizza le seguenti librerie principali:
* **Deep Learning**: `torch`, `torchvision` (PyTorch)
* **Machine & Classical Learning**: `scikit-learn`, `scikit-image`
* **Computer Vision & Demo in tempo reale**: `opencv-python`
* **Analisi dei Dati**: `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Dataset Management**: `datasets` (Hugging Face), `huggingface_hub`

---

## 📂 Struttura del Progetto

```text
riconoscimento_espressioni/
│
├── data/                          # Contiene il dataset FER-2013 (salvato in .npz)
│
├── models/                        # Modelli e scaler serializzati ad addestramento completato
│   ├── cnn_model.pth              # Pesi ottimali della CNN PyTorch
│   ├── svm_model.pkl              # Classificatore SVM (Support Vector Machine)
│   └── svm_scaler.pkl             # Standardizzazione delle feature HOG+LBP
│
├── notebooks/                     # Jupyter Notebook per le fasi di analisi, training e confronto
│   ├── 01_eda_and_preprocessing.ipynb     # EDA (Exploratory Data Analysis) e preprocessing
│   ├── 02_classical_approach.ipynb        # Estrazione feature HOG+LBP e train SVM
│   ├── 03_deep_learning_approach.ipynb    # Definizione e addestramento di EmotionCNN
│   └── 04_comparison_and_discussion.ipynb  # Report dettagliato e confronto quantitativo/qualitativo
│
├── src/                           # Script di supporto e automazione
│   └── download_dataset.py        # Scaricamento automatico del dataset FER-2013 da Hugging Face
│
├── project_documentation.md       # Documentazione tecnica approfondita del progetto
├── shared_understanding.md         # Obiettivi, requisiti e design di alto livello
├── ubiquitous_language.md         # Glossario terminologico del dominio applicativo
├── bozza_presentazione_gamma.md   # Bozza per la presentazione finale del progetto
├── webcam_demo.py                 # Script interattivo per inferenza in tempo reale da webcam
└── requirements.txt               # Dipendenze software necessarie
```

---

## 📊 Il Dataset: FER-2013
Il sistema è addestrato e validato sul dataset di riferimento standard **FER-2013** (Facial Expression Recognition 2013), composto da immagini in scala di grigi **48x48 pixel** pre-allineate ed estrapolate sui volti.

### Suddivisione del Dataset:
* **Training Set**: 28.709 immagini
* **Validation Set (PublicTest)**: 3.589 immagini
* **Test Set (PrivateTest)**: 3.589 immagini

### Mappatura delle Classi (Emozioni):
Il dataset contiene 7 emozioni distinte:
1. `0: Rabbia (Angry)`
2. `1: Disgusto (Disgust)` (classe fortemente sottorappresentata, ~1.5% del dataset)
3. `2: Paura (Fear)`
4. `3: Felicità (Happy)` (classe maggioritaria, ~25% del dataset)
5. `4: Tristezza (Sad)`
6. `5: Sorpresa (Surprise)`
7. `6: Neutro (Neutral)`

> [!WARNING]
> Lo sbilanciamento delle classi rende l'**F1-Score macro** la metrica di riferimento più affidabile per la valutazione globale dei modelli rispetto alla sola **Accuratezza (Accuracy)**.

---

## ⚙️ Installazione e Configurazione

### 1. Clonare il repository ed accedere alla cartella:
```bash
git clone <url-del-repository>
cd riconoscimento_espressioni
```

### 2. Creare e attivare l'ambiente virtuale:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

### 4. Scaricare il dataset:
Il dataset originale FER-2013 non è caricato su GitHub (la cartella `data/` è inserita nel file `.gitignore`). Per scaricarlo ed estrarlo automaticamente da Hugging Face, eseguire il seguente script:
```bash
python src/download_dataset.py
```
*Questo script effettuerà il download e genererà il file compresso `data/fer2013.npz` contenente i dati di training, validation e test, pronti per essere utilizzati all'interno dei notebook e della demo.*

---

## 🔍 Descrizione dei Due Approcci

### 📐 Approccio A: Classico (HOG + LBP + SVM)
1. **Preprocessing**: Normalizzazione locale dell'istogramma (CLAHE) per gestire le variazioni di illuminazione.
2. **Estrazione Feature**:
   * **HOG (Histogram of Oriented Gradients)**: Cattura i gradienti locali di luminosità per delineare forme e contorni (900 feature).
   * **LBP (Local Binary Patterns)**: Estrae micro-texture e rughe d'espressione per cella, mantenendo la correlazione spaziale (360 feature).
   * **Vettore Unito**: Concatenazione dei due vettori per un totale di **1260 feature**.
3. **Classificazione**: Modello `LinearSVC` (SVM lineare) addestrato previa standardizzazione via `StandardScaler`, con ottimizzazione dell'iperparametro $C$ tramite Grid Search.

### 🧠 Approccio B: Deep Learning (EmotionCNN in PyTorch)
1. **Architettura**: Rete convoluzionale custom (`EmotionCNN`) composta da 3 blocchi convoluzionali consecutivi (ciascuno con strati Conv2D, Batch Normalization, attivazione ReLU, Max Pooling ed eliminazione stocastica via Dropout) e un classificatore fully connected finale (con Dropout al 50% per contrastare l'overfitting).
2. **Data Augmentation**: Applicazione stocastica di rotazioni entro $\pm15^\circ$ e ribaltamenti orizzontali casuali sul training set.
3. **Ottimizzazione**: Minimizzazione della Loss Cross-Entropy tramite ottimizzatore `Adam` e scheduling dinamico del learning rate (`ReduceLROnPlateau`). Rilevamento ed abilitazione automatica dei dispositivi hardware disponibili (Apple Silicon `mps`, NVIDIA `cuda` o `cpu`).

---

## 🖥️ Demo in Tempo Reale (`webcam_demo.py`)
Lo script `webcam_demo.py` avvia la webcam locale ed effettua l'inferenza in tempo reale sul volto dell'utente, consentendo un confronto istantaneo tra le predizioni dei due modelli.

Per avviare la demo:
```bash
python webcam_demo.py
```

### ⌨️ Controlli da Tastiera nella Finestra della Demo:
* `c`: Cambia istantaneamente il modello attivo a **CNN (Deep Learning)**. (Il bounding box diventa **arancione**).
* `s`: Cambia istantaneamente il modello attivo a **SVM (Approccio Classico)**. (Il bounding box diventa **azzurro**).
* `q` oppure `ESC`: Chiude l'applicazione in modo pulito rilasciando la webcam.

---

## 📈 Risultati Sperimentali e Discussione

Di seguito sono riportate le prestazioni quantitative misurate sul Test Set (PrivateTest) di FER-2013 ed eseguite su CPU:

| Metrica di Confronto | Approccio Classico (HOG + LBP + SVM) | Approccio Deep Learning (EmotionCNN) |
| :--- | :---: | :---: |
| **Accuratezza Test (Accuracy)** | ~40% - 45% | **~60% - 65%** |
| **F1-Score Macro** | ~38% - 42% | **~58% - 62%** |
| **Latenza di Inferenza (CPU)** | ~1.5 - 2.5 ms | **~1.0 - 2.0 ms** |
| **Dimensione File su Disco** | **~6.5 MB** | ~18.5 MB |

### Considerazioni Chiave:
1. **Accuratezza**: La CNN supera l'approccio classico di circa 20 punti percentuali, dimostrando capacità superiori di astrazione non lineare rispetto ai filtri rigidi di HOG e LBP.
2. **Latenza ed Efficienza**: Sebbene la CNN possieda milioni di parametri in più rispetto all'SVM, la latenza di inferenza su CPU è paragonabile o persino inferiore. Questo accade perché il forward pass di PyTorch è altamente ottimizzato e parallelizzato in C++, mentre l'estrazione delle feature classiche in Python (tramite `scikit-image`) viene eseguita in modalità sequenziale su un singolo thread.
3. **Analisi degli Errori**: Entrambi i modelli faticano ad accoppiare emozioni simili a livello mimico (es. *Paura* vs. *Sorpresa*, *Tristezza* vs. *Neutro*) e soffrono la scarsità di campioni per la classe *Disgusto*. La classe *Felicità*, grazie al sorriso marcato e all'abbondanza di dati, risulta la più facile da classificare (F1-score $>80\%$).

---

## 👤 Autore
* **Matteo Berga**
* Politecnico di Torino - Corso di Image Processing (Elaborazione delle Immagini)
