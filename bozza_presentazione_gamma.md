# Presentazione del Progetto: Riconoscimento delle Espressioni Facciali
## Linee Guida per Gamma (o altro tool di generazione AI)
*Questo file è formattato in Markdown ed è ottimizzato per essere importato o incollato direttamente in Gamma. Ogni sezione contrassegnata da `---` rappresenta una nuova diapositiva. I blocchi in corsivo tra parentesi quadre come `[Suggerimento Visivo: ...]` contengono indicazioni per il layout grafico.*

---

# Diapositiva 1: Titolo e Benvenuto
## Riconoscimento delle Espressioni Facciali: Visione Classica vs Deep Learning
### Sviluppo, ottimizzazione e confronto in tempo reale di due approcci di Computer Vision

* **Autore:** Matteo Berga
* **Ambito:** Progetto di Elaborazione delle Immagini (Image Processing)
* **Obiettivo:** Classificazione automatica di 7 emozioni umane a partire da immagini di volti, con demo interattiva in tempo reale.

*[Suggerimento Visivo: Layout con sfondo scuro premium, tonalità bluastro-arancioni per rappresentare la dicotomia tra approccio classico e deep learning, e icona stilizzata di un volto con punti di tracciamento geometrici e reti neurali.]*

---

# Diapositiva 2: Introduzione e Obiettivi del Progetto
## La Sfida della Classificazione delle Emozioni
### Come tradurre la complessità delle espressioni umane in modelli computazionali?

Il progetto esplora e mette a confronto in modo sistematico due pilastri della Computer Vision:

* **1. Approccio Classico (Feature Engineering Manuale):**
  * Estrazione di caratteristiche geometriche e tessiturali fatte a mano (*hand-crafted*).
  * Combinazione di algoritmi **HOG** (forma) e **LBP** (texture).
  * Classificazione supervisionata robusta tramite **SVM** (Support Vector Machine).
* **2. Approccio Deep Learning (Feature Learning Automatico):**
  * Progettazione da zero di una Rete Neurale Convoluzionale (**CNN**).
  * Apprendimento congiunto di filtri spaziali e regole decisionali tramite PyTorch.
* **Obiettivo Finale:** Implementare un'applicazione desktop in tempo reale in grado di confrontare istantaneamente le prestazioni dei due modelli tramite webcam.

*[Suggerimento Visivo: Diagramma a due rami (Split-screen) che mostra a sinistra la pipeline classica (Immagine -> HOG+LBP -> SVM -> Emozione) e a destra la pipeline di Deep Learning (Immagine -> CNN -> Emozione).]*

---

# Diapositiva 3: Il Dataset di Riferimento (FER-2013)
## I Dati Dietro ai Modelli
### Analisi quantitativa e strutturale del dataset standard Facial Expression Recognition 2013

* **Caratteristiche delle Immagini:**
  * Immagini grezze in scala di grigi ad alta risoluzione locale (**48x48 pixel**).
  * Volti pre-allineati, centrati e ritagliati per eliminare rumore di sfondo.
* **Le 7 Classi da Classificare (Emozioni):**
  * `0: Rabbia (Angry)` | `1: Disgusto (Disgust)` | `2: Paura (Fear)`
  * `3: Felicità (Happy)` | `4: Tristezza (Sad)` | `5: Sorpresa (Surprise)`
  * `6: Neutro (Neutral)`
* **Suddivisione Rigorosa del Dataset:**
  * **Training Set:** 28.709 immagini (per addestrare i modelli).
  * **Validation Set (PublicTest):** 3.589 immagini (per ottimizzare gli iperparametri).
  * **Test Set (PrivateTest):** 3.589 immagini (per la valutazione finale "blind").

*[Suggerimento Visivo: Griglia di esempio con 7 immagini in scala di grigi del dataset FER-2013, una per ciascuna delle emozioni, con le relative etichette in evidenza.]*

---

# Diapositiva 4: Analisi Esplorativa e Sbilanciamento delle Classi
## La Realtà dei Dati Reali: Lo Sbilanciamento
### Perché l'accuratezza globale può essere una metrica ingannevole?

L'analisi esplorativa dei dati (EDA) evidenzia un forte sbilanciamento tra le classi, conservato in tutti gli split:

* **La Classe Maggioritaria:** 
  * **Felicità (Happy)** rappresenta circa il **25%** del dataset (oltre 7.200 immagini nel train).
* **La Classe Minoritaria:** 
  * **Disgusto (Disgust)** è fortemente sottorappresentata con appena l'**1.5%** del dataset (436 immagini nel train).
* **Le Classi Intermedie:** 
  * Rabbia, Paura, Tristezza, Neutro e Sorpresa oscillano stabilmente tra il 10% e il 15%.

> **Nota Metodologica:** A causa di questo sbilanciamento, l'**F1-Score macro** è stato scelto come metrica principale per valutare le prestazioni complessive, evitando che l'ottima performance sui volti felici mascherasse scarsi risultati sulle emozioni più rare.

*[Suggerimento Visivo: Grafico a barre orizzontali che mostra la distribuzione percentuale delle 7 emozioni nel dataset, evidenziando in verde la barra "Happy" (molto lunga) e in rosso la barra "Disgust" (estremamente corta).]*

---

# Diapositiva 5: Approccio Classico (1/2) - Estrazione delle Feature
## Catturare Forma e Texture: HOG & LBP
### Come descrivere geometricamente le caratteristiche di un volto senza reti neurali

L'approccio classico richiede la conversione dell'immagine in un vettore numerico di caratteristiche salienti:

```
                  ┌──► Istogramma dei Gradienti (HOG) ──► 900 feature  ┐
Input (48x48) ────┤                                                    ├─► Vettore Unito (1260 feature)
                  └──► Local Binary Patterns (LBP)     ──► 360 feature  ┘
```

* **1. HOG (Histogram of Oriented Gradients) - Forma e Contorni:**
  * Analizza la direzione dei gradienti di luminosità per identificare contorni e forme (occhi, naso, bocca).
  * **Configurazione:** Celle $8 \times 8$ pixel, blocchi $2 \times 2$ celle sovrapposti, istogramma a 9 bin.
  * **Risultato:** Vettore di **900 feature** geometriche.
* **2. LBP (Local Binary Patterns) - Micro-texture e Rughe:**
  * Ideale per catturare la texture della pelle e le micro-rughe d'espressione.
  * **Configurazione:** Vicinato $P=8$ su raggio $R=1$, pattern uniformi per invarianza rotazionale.
  * **Mantenimento Spaziale:** Suddivisione dell'immagine in celle $8 \times 8$ pixel, calcolando un istogramma a 10 bin per cella per non perdere l'informazione di *posizione* spaziale.
  * **Risultato:** Vettore di **360 feature** tessiturali.

*[Suggerimento Visivo: Due immagini esplicative: a sinistra l'immagine HOG del volto che mostra i vettori dei gradienti (sembra un disegno stilizzato a trattini), a destra l'immagine LBP con la mappa delle micro-texture evidenziate.]*

---

# Diapositiva 6: Approccio Classico (2/2) - Classificazione con SVM
## Ottimizzazione e Classificazione del Vettore Feature
### Standardizzazione dei dati e addestramento del classificatore lineare

Una volta estratto il vettore unito di **1260 feature**, la pipeline procede con la fase di classificazione:

* **1. Standardizzazione delle Feature (`StandardScaler`):**
  * Le SVM sono sensibili alla scala delle feature.
  * Media centrata a 0 e varianza riscalata a 1 per ogni caratteristica, calcolate esclusivamente sul Training Set per evitare *data leakage*.
* **2. Classificazione via SVM (`LinearSVC`):**
  * Scelta del classificatore lineare (Linear Support Vector Classifier) per l'ottimo compromesso tra velocità di inferenza e capacità di separazione in spazi a dimensionalità medio-alta.
* **3. Ottimizzazione degli Iperparametri (Grid Search):**
  * Ricerca sistematica del parametro di regolarizzazione $C \in [0.01, 0.1, 1.0]$.
  * Il miglior modello ha mostrato prestazioni ottimali con $C=0.01$ o $C=0.1$, bilanciando la complessità del modello e mitigando l'overfitting.

*[Suggerimento Visivo: Schema della pipeline classica con tre passaggi sequenziali: 1. Concatenazione Feature (1260), 2. Standardizzazione Z-score, 3. Iperpiano di separazione SVM.]*

---

# Diapositiva 7: Approccio Deep Learning (1/2) - Architettura CNN
## EmotionCNN: Rete Neurale Convoluzionale Custom
### Progettazione ed evoluzione di una rete convoluzionale profonda ottimizzata per immagini 48x48

La rete convoluzionale apprende in autonomia ed in modo integrato sia le feature che la classificazione. La struttura finale è organizzata in **3 blocchi convoluzionali sequenziali** seguiti da un **classificatore fully connected**:

* **Blocco Convoluzionale 1:**
  * 2x Conv2d ($1 \to 64$ filtri, kernel $3 \times 3$, padding 1) + Batch Normalization + ReLU.
  * MaxPool2d ($2 \times 2$, stride 2) $\to$ Riduzione spaziale a $24 \times 24$.
  * Dropout (25%) per la regolarizzazione.
* **Blocco Convoluzionale 2:**
  * 2x Conv2d ($64 \to 128$ filtri, kernel $3 \times 3$, padding 1) + Batch Normalization + ReLU.
  * MaxPool2d ($2 \times 2$, stride 2) $\to$ Riduzione spaziale a $12 \times 12$.
  * Dropout (25%).
* **Blocco Convoluzionale 3:**
  * 2x Conv2d ($128 \to 256$ filtri, kernel $3 \times 3$, padding 1) + Batch Normalization + ReLU.
  * MaxPool2d ($2 \times 2$, stride 2) $\to$ Riduzione spaziale a $6 \times 6$.
  * Dropout (25%).
* **Classificatore Fully Connected (FC):**
  * Flattening a un vettore di **9216** feature.
  * Linear Layer ($9216 \to 512$ neuroni) + Batch Normalization + ReLU + Dropout (50%).
  * Linear Output Layer ($512 \to 7$ logits per ciascuna emozione).

*[Suggerimento Visivo: Rappresentazione 3D schematica della CNN (EmotionCNN) che mostra i volumi dei tensori che si rimpiccioliscono in larghezza/altezza e aumentano in profondità (canali): 48x48x1 -> 24x24x64 -> 12x12x128 -> 6x6x256 -> Vettore di 9216 -> Output di 7.]*

---

# Diapositiva 8: Approccio Deep Learning (2/2) - Training & Regolarizzazione
## Strategia di Addestramento in PyTorch
### Ottimizzazione, scheduling e tecniche di prevenzione dell'overfitting

* **Data Augmentation (Training Set):**
  * Per simulare variazioni reali e aumentare la robustezza: ribaltamento orizzontale casuale (`RandomHorizontalFlip`) e rotazioni casuali fino a $\pm15^\circ$ (`RandomRotation`).
* **Loss Function e Ottimizzatore:**
  * `CrossEntropyLoss` (calcolo integrato della Softmax).
  * Ottimizzatore `Adam` (learning rate iniziale a `0.001`, `weight_decay=1e-4` per regolarizzazione dei pesi).
* **Ottimizzazione Dinamica del Learning Rate:**
  * Utilizzo di `ReduceLROnPlateau` monitorando l'accuratezza di validation.
  * Dimezzamento del learning rate (fattore `0.5`) in caso di assenza di miglioramenti per 3 epoche consecutive.
* **Checkpoint e Scalabilità Hardware:**
  * Salvataggio automatico dello stato del modello solo al raggiungimento del picco di accuratezza sul validation set (`cnn_model.pth`).
  * Supporto cross-platform integrato: accelerazione **MPS** su Apple Silicon, **CUDA** su NVIDIA GPU e fallback su **CPU**.

*[Suggerimento Visivo: Grafico delle curve di addestramento (Loss e Accuracy per epoche su Train e Val) che mostra la convergenza stabile e la riduzione graduale della loss in 30 epoche.]*

---

# Diapositiva 9: Esecuzione in Tempo Reale: webcam_demo.py
## La Demo Interattiva
### Rilevamento dei volti e inferenza in tempo reale a fotogrammi elevati

Lo script `webcam_demo.py` offre un banco di prova pratico ed immediato per validare l'esperienza utente:

1. **Acquisizione e Specchiatura:** Frame catturati da webcam tramite OpenCV (`cv2.VideoCapture(0)`) e specchiati per un'interazione naturale.
2. **Face Detection:** Utilizzo del rilevatore a cascata di Haar (`haarcascade_frontalface_default.xml`) con dimensione minima impostata a $60 \times 60$ pixel per evitare falsi positivi sullo sfondo.
3. **Preprocessing del ROI:** Il volto viene isolato, convertito in scala di grigi, ridimensionato a $48 \times 48$ pixel usando l'interpolazione ad area.
4. **Inferenza Istantanea:**
   * **In modalità CNN:** Conversione in tensore, riscalamento in $[-1.0, 1.0]$ e forward pass. Probabilità ottenuta con Softmax.
   * **In modalità SVM:** Estrazione delle feature HOG+LBP, scaling ed esecuzione. Pseudo-probabilità calcolata con Softmax sui punteggi della `decision_function`.
5. **Overlay Grafico Dinamico:**
   * Disegno del Bounding Box: **Arancione** per la CNN, **Blu/Azzurro** per la SVM.
   * Visualizzazione in tempo reale di: FPS correnti, nome dell'emozione rilevata e percentuale di confidenza.
   * Switch istantaneo del modello premendo i tasti `'c'` (CNN) o `'s'` (SVM).

*[Suggerimento Visivo: Mockup di uno schermo di computer che mostra la webcam attiva con un volto contornato da un bounding box arancione della CNN con etichetta "Felicita (98.4%)" ed un pannello informativo in alto a sinistra che illustra FPS e tasti rapidi.]*

---

# Diapositiva 10: Risultati Sperimentali - Confronto Quantitativo
## I Numeri a Confronto
### Analisi delle performance misurate sul Test Set (PrivateTest) di FER-2013

| Metrica di Confronto | Approccio Classico (HOG + LBP + SVM) | Approccio Deep Learning (EmotionCNN) |
|---|---|---|
| **Accuratezza Test (Accuracy)** | ~40% - 45% | **~60% - 65%** |
| **F1-Score Macro** | ~38% - 42% | **~58% - 62%** |
| **Latenza di Inferenza (CPU)** | ~1.5 - 2.5 millisecondi | **~1.0 - 2.0 millisecondi** |
| **Dimensione File su Disco** | **~6.5 Megabytes** (modello + scaler) | ~18.5 Megabytes (pesi `.pth`) |

### Conclusioni Quantitative:
* La **CNN** supera nettamente l'approccio classico con un incremento assoluto di circa il **20%** sia in accuratezza che in F1-score macro.
* La dimensione del file su disco della CNN (18.5 MB) rimane estremamente contenuta, rendendo entrambi i modelli compatibili con sistemi embedded o dispositivi a basse risorse.

*[Suggerimento Visivo: Tabella riassuntiva pulita con i valori chiave in evidenza; le righe vincenti della CNN evidenziate con un badge o colore verde chiaro/arancione.]*

---

# Diapositiva 11: Analisi Qualitativa e Comportamento sui Dati
## Dove Sbagliano i Modelli?
### Studio delle matrici di confusione e del comportamento sui dati sbilanciati

L'analisi qualitativa sul Test Set rivela sfide condivise da entrambe le metodologie:

* **1. Confusione tra Emozioni Simili:**
  * **Paura (Fear)** e **Sorpresa (Surprise)** vengono spesso scambiate a causa di configurazioni facciali estremamente simili (occhi sbarrati, bocca spalancata).
  * **Tristezza (Sad)** e **Neutro (Neutral)** si sovrappongono frequentemente in presenza di micro-espressioni o volti a riposo.
* **2. Impatto del Dataset Sbilanciato:**
  * **Felicità (Happy)** è la classe più performante (F1-score $>80\%$) grazie alla netta visibilità geometrica del sorriso (facilmente catturabile da HOG/LBP e CNN) e al volume di dati elevato.
  * **Disgusto (Disgust)** è l'emozione con le performance più basse, risentendo del ridotto numero di campioni di addestramento.
* **3. Limiti delle Feature Hand-Crafted:**
  * HOG e LBP catturano schemi rigidi e lineari. Non riescono a modellare le variazioni individuali del viso (forma del naso, pieghe soggettive, lievi rotazioni) che la CNN riesce invece a interpretare grazie all'astrazione gerarchica dei suoi strati profondi.

*[Suggerimento Visivo: Rappresentazione schematica di due matrici di confusione semplificate che evidenziano visivamente i punti di accumulo degli errori (es. l'incrocio tra Paura e Sorpresa).]*

---

# Diapositiva 12: Il Paradosso della Latenza su CPU
## Perché la Rete Neurale è Più Veloce delle Feature Classiche?
### Un'analisi approfondita sull'efficienza computazionale dei due metodi

Un risultato controintuitivo dell'esperimento è che il forward pass della CNN (modello più complesso con milioni di parametri) è spesso **più rapido** dell'estrazione di feature HOG+LBP su CPU.

* **La Lentezza del Machine Learning Classico:**
  * L'estrazione di HOG e LBP è implementata in Python tramite la libreria `scikit-image`.
  * Questo calcolo avviene in modo sequenziale (single-thread) su CPU ad ogni fotogramma, introducendo un collo di bottiglia computazionale fisso di circa 2 ms.
* **L'Efficienza del Deep Learning:**
  * Il forward pass in PyTorch, sebbene contenga un numero elevato di operazioni algebriche, è eseguito tramite codice C++ altamente ottimizzato e parallelizzato.
  * Sfrutta le librerie di algebra lineare avanzate (come Intel MKL o OpenMP) e l'accelerazione hardware (MPS su Mac Apple Silicon) per elaborare i tensori in frazioni di millisecondo.
* **Impatto Pratico:**
  * Entrambi i modelli garantiscono una demo real-time fluida (oltre 30-60 FPS), ma la CNN dimostra un'efficienza algoritmica superiore in fase di produzione.

*[Suggerimento Visivo: Grafico comparativo della latenza (in millisecondi) dove si mostra che il tempo della CNN è inferiore a quello di HOG+LBP + Classificazione SVM, evidenziando il tempo di calcolo delle feature classico come collo di bottiglia.]*

---

# Diapositiva 13: Conclusioni e Sviluppi Futuri
## Sintesi dei Risultati e Passi Successivi
### Cosa abbiamo imparato e come possiamo estendere il progetto?

* **Punti Chiave:**
  * **Il Deep Learning vince** in accuratezza (+20%) e stabilità qualitativa senza sacrificare la latenza.
  * **L'approccio classico è utile** come solido benchmark iniziale e richiede pochissimo addestramento (pochi minuti rispetto a epoche di rete neurale).
  * **La demo real-time** dimostra la fattibilità tecnica di entrambi i modelli su dispositivi consumer comuni (senza necessità di GPU dedicate).
* **Sviluppi Futuri Consigliati:**
  * **Risoluzione dello sbilanciamento:** Applicazione di tecniche di *over-sampling*, loss pesate (Focal Loss) o sintesi di volti con GAN per la classe Disgusto.
  * **Face Detection Avanzato:** Sostituzione di Haar Cascades con rilevatori basati su Deep Learning (es. MediaPipe o SSD) per una maggiore stabilità alle variazioni di inclinazione e illuminazione.
  * **Modelli più leggeri e moderni:** Esplorazione del Transfer Learning con MobileNetV3 o ShuffleNet per massimizzare l'efficienza su dispositivi mobile ed edge.

*[Suggerimento Visivo: Icona di un razzo o di una freccia puntata verso l'alto con un layout pulito a tre colonne per i passi successivi (Bilanciamento dati, MediaPipe, MobileNetV3).]*
