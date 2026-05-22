import nbformat as nbf
import os

def create_classical_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 02. Approccio Classico: Feature HOG + LBP + Classificatore SVM\n"
        "Questo notebook descrive l'implementazione dell'approccio classico per il riconoscimento delle espressioni facciali. "
        "Il processo prevede:\n"
        "1. Estrazione di feature HOG (contorni e forme geometriche).\n"
        "2. Estrazione di feature LBP (pattern locali di texture e micro-variazioni) calcolate cella per cella.\n"
        "3. Concatenazione dei due vettori di caratteristiche.\n"
        "4. Standardizzazione delle feature.\n"
        "5. Addestramento di un classificatore SVM (Support Vector Machine) ottimizzato.\n"
        "6. Valutazione e salvataggio del modello finalizzato."
    ))
    
    # Cell 2: Code Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from tqdm import tqdm\n"
        "from skimage.feature import hog, local_binary_pattern\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        "import joblib\n\n"
        "sns.set_theme(style=\"white\")\n"
        "plt.rcParams[\"figure.figsize\"] = (8, 6)"
    ))
    
    # Cell 3: Code Loading Data
    cells.append(nbf.v4.new_code_cell(
        "# Carichiamo il dataset pre-salvato\n"
        "data_path = \"../data/fer2013.npz\"\n"
        "if not os.path.exists(data_path):\n"
        "    data_path = \"data/fer2013.npz\"\n\n"
        "data = np.load(data_path)\n"
        "X_train_raw, y_train = data[\"train_images\"], data[\"train_labels\"]\n"
        "X_val_raw, y_val = data[\"val_images\"], data[\"val_labels\"]\n"
        "X_test_raw, y_test = data[\"test_images\"], data[\"test_labels\"]\n\n"
        "print(f\"Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}\")"
    ))
    
    # Cell 4: Markdown Feature Extraction Explanation
    cells.append(nbf.v4.new_markdown_cell(
        "## Estrazione delle Feature HOG e LBP\n"
        "Implementiamo una funzione che combina:\n"
        "- **HOG** con `orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`, che produce 900 feature.\n"
        "- **LBP** con `P=8, R=1` (metodo uniform). Calcoliamo poi l'istogramma LBP a 10 bin per ogni cella 8x8 (36 celle totali), "
        "ottenendo 360 feature.\n"
        "Il vettore risultante avrà dimensione **1260**."
    ))
    
    # Cell 5: Code Feature Extraction Implementation
    cells.append(nbf.v4.new_code_cell(
        "def extract_hog_lbp_features(img):\n"
        "    # 1. Estrazione HOG (Gradienti orientati)\n"
        "    hog_feats = hog(\n"
        "        img, \n"
        "        orientations=9, \n"
        "        pixels_per_cell=(8, 8), \n"
        "        cells_per_block=(2, 2), \n"
        "        visualize=False\n"
        "    )\n"
        "    \n"
        "    # 2. Estrazione LBP (Texture locale)\n"
        "    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')\n"
        "    \n"
        "    # Calcolo dell'istogramma LBP per ogni cella 8x8 per mantenere l'informazione spaziale\n"
        "    cell_size = 8\n"
        "    lbp_feats = []\n"
        "    for i in range(0, 48, cell_size):\n"
        "        for j in range(0, 48, cell_size):\n"
        "            cell = lbp[i:i+cell_size, j:j+cell_size]\n"
        "            # Uniform LBP produce valori da 0 a 9\n"
        "            hist, _ = np.histogram(cell, bins=10, range=(0, 10))\n"
        "            hist = hist.astype(np.float32)\n"
        "            # Normalizzazione dell'istogramma\n"
        "            norm = np.linalg.norm(hist)\n"
        "            if norm > 0:\n"
        "                hist /= norm\n"
        "            lbp_feats.extend(hist)\n"
        "            \n"
        "    lbp_feats = np.array(lbp_feats, dtype=np.float32)\n"
        "    \n"
        "    # Concatenazione HOG + LBP\n"
        "    return np.concatenate([hog_feats, lbp_feats])\n\n"
        "def extract_features_batch(images, desc):\n"
        "    features = []\n"
        "    for img in tqdm(images, desc=desc):\n"
        "        features.append(extract_hog_lbp_features(img))\n"
        "    return np.array(features, dtype=np.float32)"
    ))
    
    # Cell 6: Code Running Feature Extraction
    cells.append(nbf.v4.new_code_cell(
        "# Estrazione feature sui tre split\n"
        "print(\"Avvio estrazione delle feature... (può richiedere circa 1-2 minuti)\")\n"
        "X_train_feats = extract_features_batch(X_train_raw, \"Train Features\")\n"
        "X_val_feats = extract_features_batch(X_val_raw, \"Val Features\")\n"
        "X_test_feats = extract_features_batch(X_test_raw, \"Test Features\")\n\n"
        "print(f\"Dimensioni feature estratte:\")\n"
        "print(f\"Train: {X_train_feats.shape}, Val: {X_val_feats.shape}, Test: {X_test_feats.shape}\")"
    ))
    
    # Cell 7: Markdown Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "## Standardizzazione dei Dati\n"
        "Le SVM sono sensibili alla scala delle feature. Utilizziamo `StandardScaler` per normalizzare le caratteristiche "
        "con media 0 e varianza 1."
    ))
    
    # Cell 8: Code Scaling
    cells.append(nbf.v4.new_code_cell(
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train_feats)\n"
        "X_val_scaled = scaler.transform(X_val_feats)\n"
        "X_test_scaled = scaler.transform(X_test_feats)\n\n"
        "print(\"Standardizzazione completata.\")"
    ))
    
    # Cell 9: Markdown Grid Search
    cells.append(nbf.v4.new_markdown_cell(
        "## Ottimizzazione della SVM (Grid Search / Validation)\n"
        "Addestriamo diversi modelli SVM lineari provando diversi valori del parametro di regolarizzazione $C$ "
        "sul validation set per trovare il miglior compromesso."
    ))
    
    # Cell 10: Code Grid Search Run
    cells.append(nbf.v4.new_code_cell(
        "c_values = [0.01, 0.1, 1.0]\n"
        "best_val_acc = 0.0\n"
        "best_model = None\n"
        "best_c = None\n\n"
        "for c in c_values:\n"
        "    print(f\"Addestramento SVM con C={c}...\")\n"
        "    # Usiamo dual=False perché n_samples > n_features\n"
        "    clf = LinearSVC(C=c, dual=False, max_iter=2000, random_state=42)\n"
        "    clf.fit(X_train_scaled, y_train)\n"
        "    \n"
        "    # Valutiamo sul validation set\n"
        "    val_preds = clf.predict(X_val_scaled)\n"
        "    val_acc = accuracy_score(y_val, val_preds)\n"
        "    print(f\"  -> Accuratezza Validation: {val_acc:.4f}\")\n"
        "    \n"
        "    if val_acc > best_val_acc:\n"
        "        best_val_acc = val_acc\n"
        "        best_model = clf\n"
        "        best_c = c\n\n"
        "print(f\"\\nModello migliore trovato con C={best_c} (Acc: {best_val_acc:.4f})\")"
    ))
    
    # Cell 11: Markdown Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## Valutazione sul Test Set\n"
        "Valutiamo il miglior modello sul Test Set (PrivateTest) calcolando l'accuratezza totale, le metriche per classe "
        "e mostrando la matrice di confusione."
    ))
    
    # Cell 12: Code Evaluation
    cells.append(nbf.v4.new_code_cell(
        "y_pred = best_model.predict(X_test_scaled)\n"
        "test_acc = accuracy_score(y_test, y_pred)\n"
        "print(f\"Accuratezza complessiva sul Test Set: {test_acc:.4f}\\n\")\n\n"
        "EMOTIONS = {\n"
        "    0: \"Rabbia (Angry)\",\n"
        "    1: \"Disgusto (Disgust)\",\n"
        "    2: \"Paura (Fear)\",\n"
        "    3: \"Felicità (Happy)\",\n"
        "    4: \"Tristezza (Sad)\",\n"
        "    5: \"Sorpresa (Surprise)\",\n"
        "    6: \"Neutro (Neutral)\"\n"
        "}\n"
        "target_names = [EMOTIONS[i] for i in range(7)]\n\n"
        "print(\"Report di classificazione:\")\n"
        "print(classification_report(y_test, y_pred, target_names=target_names))"
    ))
    
    # Cell 13: Code Confusion Matrix
    cells.append(nbf.v4.new_code_cell(
        "# Calcolo della matrice di confusione\n"
        "cm = confusion_matrix(y_test, y_pred)\n"
        "cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalizzata per riga\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    cm_normalized, \n"
        "    annot=True, \n"
        "    fmt=\".2f\", \n"
        "    cmap=\"Blues\", \n"
        "    xticklabels=target_names, \n"
        "    yticklabels=target_names\n"
        ")\n"
        "plt.title(\"Matrice di Confusione Normalizzata (SVM HOG+LBP)\", fontsize=14, fontweight='bold')\n"
        "plt.ylabel(\"Etichetta Reale\", fontsize=12)\n"
        "plt.xlabel(\"Etichetta Predetta\", fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 14: Markdown Saving Model
    cells.append(nbf.v4.new_markdown_cell(
        "## Salvataggio del Modello e dello Scaler\n"
        "Salviamo lo scaler e il modello SVM addestrato in formato `.joblib`/`.pkl` nella cartella `models/` "
        "in modo da poterli ricaricare nella demo in tempo reale."
    ))
    
    # Cell 15: Code Saving
    cells.append(nbf.v4.new_code_cell(
        "os.makedirs(\"../models\", exist_ok=True)\n"
        "os.makedirs(\"models\", exist_ok=True)\n\n"
        "# Salvataggio su file\n"
        "models_dir = \"../models\" if os.path.exists(\"../models\") else \"models\"\n"
        "model_file = os.path.join(models_dir, \"svm_model.pkl\")\n"
        "scaler_file = os.path.join(models_dir, \"svm_scaler.pkl\")\n\n"
        "joblib.dump(best_model, model_file)\n"
        "joblib.dump(scaler, scaler_file)\n\n"
        "print(f\"Modello salvato in: {model_file}\")\n"
        "print(f\"Scaler salvato in: {scaler_file}\")"
    ))
    
    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/02_classical_approach.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 02_classical_approach.ipynb generato con successo!")

if __name__ == "__main__":
    create_classical_notebook()
