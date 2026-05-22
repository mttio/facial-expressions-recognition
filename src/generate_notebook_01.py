import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 01. Analisi Esplorativa dei Dati (EDA) e Preprocessing\n"
        "Questo notebook carica il dataset FER-2013 pre-salvato come archivio NumPy, analizza le statistiche di base, "
        "mostra la distribuzione delle classi per ciascuno split (Train, Val, Test) e visualizza alcune immagini di "
        "esempio per ciascuna emozione."
    ))
    
    # Cell 2: Code Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Impostiamo lo stile dei grafici\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 6)"
    ))
    
    # Cell 3: Code Loading Data
    cells.append(nbf.v4.new_code_cell(
        "# Carichiamo il dataset\n"
        "data_path = \"../data/fer2013.npz\"\n"
        "if not os.path.exists(data_path):\n"
        "    data_path = \"data/fer2013.npz\"\n\n"
        "data = np.load(data_path)\n\n"
        "X_train, y_train = data[\"train_images\"], data[\"train_labels\"]\n"
        "X_val, y_val = data[\"val_images\"], data[\"val_labels\"]\n"
        "X_test, y_test = data[\"test_images\"], data[\"test_labels\"]\n\n"
        "print(\"Dimensioni dei dati caricati:\")\n"
        "print(f\"Train - Immagini: {X_train.shape}, Etichette: {y_train.shape}\")\n"
        "print(f\"Val   - Immagini: {X_val.shape}, Etichette: {y_val.shape}\")\n"
        "print(f\"Test  - Immagini: {X_test.shape}, Etichette: {y_test.shape}\")"
    ))
    
    # Cell 4: Code Mapping Emotions
    cells.append(nbf.v4.new_code_cell(
        "# Definiamo la mappatura delle emozioni in italiano e inglese\n"
        "EMOTIONS = {\n"
        "    0: \"Rabbia (Angry)\",\n"
        "    1: \"Disgusto (Disgust)\",\n"
        "    2: \"Paura (Fear)\",\n"
        "    3: \"Felicità (Happy)\",\n"
        "    4: \"Tristezza (Sad)\",\n"
        "    5: \"Sorpresa (Surprise)\",\n"
        "    6: \"Neutro (Neutral)\"\n"
        "}\n\n"
        "emotion_names = [EMOTIONS[i] for i in range(7)]"
    ))
    
    # Cell 5: Markdown Class Distribution
    cells.append(nbf.v4.new_markdown_cell(
        "## Distribuzione delle Classi (Emozioni)\n"
        "Analizziamo lo sbilanciamento delle classi nei tre diversi split per verificare che la distribuzione sia consistente."
    ))
    
    # Cell 6: Code Class Counts & Percentages
    cells.append(nbf.v4.new_code_cell(
        "# Creiamo un DataFrame per la visualizzazione delle frequenze\n"
        "df_train = pd.DataFrame({\"Emozione\": [EMOTIONS[y] for y in y_train], \"Split\": \"Train\"})\n"
        "df_val = pd.DataFrame({\"Emozione\": [EMOTIONS[y] for y in y_val], \"Split\": \"Val\"})\n"
        "df_test = pd.DataFrame({\"Emozione\": [EMOTIONS[y] for y in y_test], \"Split\": \"Test\"})\n\n"
        "df_all = pd.concat([df_train, df_val, df_test])\n\n"
        "# Conteggio delle classi\n"
        "counts = df_all.groupby([\"Split\", \"Emozione\"]).size().unstack(level=0)\n"
        "counts = counts.reindex(emotion_names)\n"
        "print(\"Numero di campioni per classe e split:\")\n"
        "print(counts)\n\n"
        "# Calcoliamo le percentuali\n"
        "percentages = counts.div(counts.sum(axis=0), axis=1) * 100\n"
        "print(\"\\nDistribuzione percentuale (%) per split:\")\n"
        "print(percentages.round(2))"
    ))
    
    # Cell 7: Code Class Distribution Plot
    cells.append(nbf.v4.new_code_cell(
        "# Grafico a barre della distribuzione delle classi\n"
        "plt.figure(figsize=(12, 6))\n"
        "ax = sns.countplot(\n"
        "    data=df_all, \n"
        "    x=\"Emozione\", \n"
        "    hue=\"Split\", \n"
        "    order=emotion_names,\n"
        "    palette=\"viridis\"\n"
        ")\n"
        "plt.title(\"Distribuzione delle Emozioni per ciascun Split nel Dataset FER-2013\", fontsize=14)\n"
        "plt.xlabel(\"Emozione\", fontsize=12)\n"
        "plt.ylabel(\"Numero di Campioni\", fontsize=12)\n"
        "plt.xticks(rotation=15)\n"
        "plt.legend(title=\"Split\")\n\n"
        "# Aggiunge i valori sopra le barre\n"
        "for p in ax.patches:\n"
        "    height = p.get_height()\n"
        "    if height > 0:\n"
        "        ax.annotate(\n"
        "            f'{int(height)}',\n"
        "            (p.get_x() + p.get_width() / 2., height),\n"
        "            ha='center', va='bottom',\n"
        "            fontsize=9, color='black',\n"
        "            xytext=(0, 3),\n"
        "            textcoords='offset points'\n"
        "        )\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 8: Markdown Visualization Intro
    cells.append(nbf.v4.new_markdown_cell(
        "## Visualizzazione delle Immagini del Volto\n"
        "Mostriamo una griglia di immagini per ciascuna delle 7 emozioni per ispezionare visivamente la qualità dei volti ed "
        "eventuali rumori/ambiguità nei dati."
    ))
    
    # Cell 9: Code Visualization Grid
    cells.append(nbf.v4.new_code_cell(
        "# Grid di immagini di esempio\n"
        "n_examples = 5\n"
        "fig, axes = plt.subplots(7, n_examples, figsize=(15, 18))\n\n"
        "for emotion_idx, emotion_name in EMOTIONS.items():\n"
        "    # Troviamo gli indici corrispondenti a questa emozione nel training set\n"
        "    indices = np.where(y_train == emotion_idx)[0]\n"
        "    # Selezioniamo casualmente n_examples\n"
        "    np.random.seed(42) # per riproducibilità\n"
        "    selected_indices = np.random.choice(indices, n_examples, replace=False)\n"
        "    \n"
        "    for col_idx, img_idx in enumerate(selected_indices):\n"
        "        ax = axes[emotion_idx, col_idx]\n"
        "        img = X_train[img_idx]\n"
        "        ax.imshow(img, cmap='gray')\n"
        "        ax.axis('off')\n"
        "        \n"
        "        if col_idx == 0:\n"
        "            ax.set_title(emotion_name, loc='left', fontweight='bold', fontsize=12)\n\n"
        "plt.suptitle(\"Esempi di Espressioni Facciali per Emozione (FER-2013)\", fontsize=16, fontweight='bold', y=0.99)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 10: Markdown Conclusion
    cells.append(nbf.v4.new_markdown_cell(
        "## Conclusioni dell'EDA\n"
        "1. **Risoluzione e Colore**: Le immagini sono 48x48 pixel in scala di grigi.\n"
        "2. **Sbilanciamento delle Classi**: La classe *Felicità (Happy)* è la più rappresentata (~25% in tutti gli split), "
        "mentre la classe *Disgusto (Disgust)* è fortemente sottorappresentata (~1.5%). Questo sbilanciamento dovrà essere tenuto "
        "in considerazione nell'addestramento dei modelli (ad es. usando pesi di classe o metriche di valutazione pesate "
        "come l'F1-score invece della sola accuratezza).\n"
        "3. **Distribuzione degli Split**: Gli split di Train, Validation e Test hanno percentuali di classi molto simili, "
        "garantendo una valutazione coerente ed equa dei modelli.\n"
        "4. **Rumorosità dei Dati**: Alcune immagini mostrano occlusioni parziali (mani sul volto, occhiali da sole), scritte "
        "o disegni sovrapposti, o angolazioni estreme. Questo rende FER-2013 un dataset \"in the wild\" realistico e stimolante "
        "per gli algoritmi di computer vision."
    ))
    
    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/01_eda_and_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 01_eda_and_preprocessing.ipynb generato con successo!")

if __name__ == "__main__":
    create_eda_notebook()
