import nbformat as nbf
import os

def create_dl_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 03. Approccio Deep Learning: Rete Neurale Convoluzionale (CNN) in PyTorch\n"
        "Questo notebook descrive l'implementazione del secondo approccio basato sul Deep Learning. Il processo prevede:\n"
        "1. Creazione di un PyTorch Custom Dataset e DataLoaders per lo split di Train, Validation e Test.\n"
        "2. Applicazione di Data Augmentation (rotazione e flip orizzontale) per il set di training per contrastare l'overfitting.\n"
        "3. Definizione di un'architettura CNN profonda e performante (blocchi convoluzionali con Batch Normalization, Max Pooling, Dropout e livelli Fully Connected).\n"
        "4. Definizione del ciclo di addestramento ottimizzato per l'hardware Apple Silicon (MPS - Metal Performance Shaders) o GPU CUDA.\n"
        "5. Valutazione e salvataggio del modello."
    ))
    
    # Cell 2: Code Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from tqdm import tqdm\n"
        "from PIL import Image\n\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import torch.optim as optim\n"
        "from torch.utils.data import Dataset, DataLoader\n"
        "from torchvision import transforms\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
        "sns.set_theme(style=\"white\")\n"
        "plt.rcParams[\"figure.figsize\"] = (8, 6)"
    ))
    
    # Cell 3: Code Device Config
    cells.append(nbf.v4.new_code_cell(
        "# Configurazione del dispositivo per l'accelerazione hardware\n"
        "if torch.backends.mps.is_available():\n"
        "    device = torch.device(\"mps\")\n"
        "elif torch.cuda.is_available():\n"
        "    device = torch.device(\"cuda\")\n"
        "else:\n"
        "    device = torch.device(\"cpu\")\n\n"
        "print(f\"Dispositivo di calcolo selezionato: {device}\")"
    ))
    
    # Cell 4: Code Loading Data
    cells.append(nbf.v4.new_code_cell(
        "# Carichiamo il dataset pre-salvato\n"
        "data_path = \"../data/fer2013.npz\"\n"
        "if not os.path.exists(data_path):\n"
        "    data_path = \"data/fer2013.npz\"\n\n"
        "data = np.load(data_path)\n"
        "X_train, y_train = data[\"train_images\"], data[\"train_labels\"]\n"
        "X_val, y_val = data[\"val_images\"], data[\"val_labels\"]\n"
        "X_test, y_test = data[\"test_images\"], data[\"test_labels\"]\n\n"
        "print(f\"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\")"
    ))
    
    # Cell 5: Markdown Dataset and Transforms
    cells.append(nbf.v4.new_markdown_cell(
        "## Definizione del Dataset e Data Augmentation\n"
        "Definiamo una classe custom `FERDataset` che converte gli array numpy in immagini PIL al volo per consentire l'uso delle "
        "funzioni di Data Augmentation offerte da `torchvision.transforms` (flip orizzontali, rotazioni casuali)."
    ))
    
    # Cell 6: Code Dataset Definition
    cells.append(nbf.v4.new_code_cell(
        "class FERDataset(Dataset):\n"
        "    def __init__(self, images, labels, transform=None):\n"
        "        self.images = images\n"
        "        self.labels = labels\n"
        "        self.transform = transform\n"
        "        \n"
        "    def __len__(self):\n"
        "        return len(self.images)\n"
        "        \n"
        "    def __getitem__(self, idx):\n"
        "        img = Image.fromarray(self.images[idx])\n"
        "        label = self.labels[idx]\n"
        "        \n"
        "        if self.transform:\n"
        "            img = self.transform(img)\n"
        "            \n"
        "        return img, label\n\n"
        "# Definiamo le trasformazioni\n"
        "train_transform = transforms.Compose([\n"
        "    transforms.RandomHorizontalFlip(),\n"
        "    transforms.RandomRotation(15),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize((0.5,), (0.5,))\n"
        "])\n\n"
        "val_test_transform = transforms.Compose([\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize((0.5,), (0.5,))\n"
        "])\n\n"
        "# Creazione dei dataset\n"
        "train_dataset = FERDataset(X_train, y_train, transform=train_transform)\n"
        "val_dataset = FERDataset(X_val, y_val, transform=val_test_transform)\n"
        "test_dataset = FERDataset(X_test, y_test, transform=val_test_transform)\n\n"
        "# Creazione dei dataloader\n"
        "BATCH_SIZE = 128\n"
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n"
        "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n"
        "test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)\n\n"
        "print(\"Dataloaders pronti.\")"
    ))
    
    # Cell 7: Markdown Model Architecture
    cells.append(nbf.v4.new_markdown_cell(
        "## Architettura della CNN\n"
        "Definiamo un'architettura convoluzionale con 3 macro-blocchi (ciascuno formato da due convoluzioni a filtro $3\\times3$, "
        "Batch Normalization, Max Pooling per ridurre le dimensioni e Dropout per regolarizzare) e 2 strati Fully Connected finali."
    ))
    
    # Cell 8: Code CNN Definition
    cells.append(nbf.v4.new_code_cell(
        "class EmotionCNN(nn.Module):\n"
        "    def __init__(self):\n"
        "        super(EmotionCNN, self).__init__()\n"
        "        \n"
        "        # Blocco 1: Input 48x48x1 -> Conv 64 -> MaxPool -> 24x24x64\n"
        "        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)\n"
        "        self.bn1 = nn.BatchNorm2d(64)\n"
        "        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)\n"
        "        self.bn2 = nn.BatchNorm2d(64)\n"
        "        self.pool1 = nn.MaxPool2d(2, 2)\n"
        "        self.drop1 = nn.Dropout(0.25)\n"
        "        \n"
        "        # Blocco 2: Input 24x24x64 -> Conv 128 -> MaxPool -> 12x12x128\n"
        "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)\n"
        "        self.bn3 = nn.BatchNorm2d(128)\n"
        "        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)\n"
        "        self.bn4 = nn.BatchNorm2d(128)\n"
        "        self.pool2 = nn.MaxPool2d(2, 2)\n"
        "        self.drop2 = nn.Dropout(0.25)\n"
        "        \n"
        "        # Blocco 3: Input 12x12x128 -> Conv 256 -> MaxPool -> 6x6x256\n"
        "        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)\n"
        "        self.bn5 = nn.BatchNorm2d(256)\n"
        "        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)\n"
        "        self.bn6 = nn.BatchNorm2d(256)\n"
        "        self.pool3 = nn.MaxPool2d(2, 2)\n"
        "        self.drop3 = nn.Dropout(0.25)\n"
        "        \n"
        "        # Fully Connected: Input 256 * 6 * 6 = 9216 -> Dense 512 -> Dense 7\n"
        "        self.fc1 = nn.Linear(256 * 6 * 6, 512)\n"
        "        self.bn_fc1 = nn.BatchNorm1d(512)\n"
        "        self.drop_fc1 = nn.Dropout(0.5)\n"
        "        self.fc2 = nn.Linear(512, 7)\n"
        "        \n"
        "    def forward(self, x):\n"
        "        x = F.relu(self.bn1(self.conv1(x)))\n"
        "        x = F.relu(self.bn2(self.conv2(x)))\n"
        "        x = self.pool1(x)\n"
        "        x = self.drop1(x)\n"
        "        \n"
        "        x = F.relu(self.bn3(self.conv3(x)))\n"
        "        x = F.relu(self.bn4(self.conv4(x)))\n"
        "        x = self.pool2(x)\n"
        "        x = self.drop2(x)\n"
        "        \n"
        "        x = F.relu(self.bn5(self.conv5(x)))\n"
        "        x = F.relu(self.bn6(self.conv6(x)))\n"
        "        x = self.pool3(x)\n"
        "        x = self.drop3(x)\n"
        "        \n"
        "        # Flattening\n"
        "        x = x.view(-1, 256 * 6 * 6)\n"
        "        \n"
        "        x = F.relu(self.bn_fc1(self.fc1(x)))\n"
        "        x = self.drop_fc1(x)\n"
        "        x = self.fc2(x)\n"
        "        return x\n\n"
        "model = EmotionCNN().to(device)\n"
        "print(model)"
    ))
    
    # Cell 9: Markdown Training Loops Setup
    cells.append(nbf.v4.new_markdown_cell(
        "## Addestramento del Modello\n"
        "Definiamo i parametri di addestramento:\n"
        "- **Loss**: CrossEntropyLoss.\n"
        "- **Optimizer**: Adam con learning rate iniziale pari a `0.001` e regolarizzazione $L_2$ (`weight_decay=1e-4`).\n"
        "- **Scheduler**: `ReduceLROnPlateau` che dimezza il learning rate se l'accuratezza sul validation set non migliora "
        "per 3 epoche.\n"
        "Addestreremo per **30 epoche**, salvando costantemente i pesi del modello che ottiene la miglior accuratezza "
        "di validazione."
    ))
    
    # Cell 10: Code Training Implementation
    cells.append(nbf.v4.new_code_cell(
        "criterion = nn.CrossEntropyLoss()\n"
        "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)\n"
        "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)\n\n"
        "EPOCHS = 30\n"
        "best_val_acc = 0.0\n"
        "history = {\"train_loss\": [], \"train_acc\": [], \"val_loss\": [], \"val_acc\": []}\n\n"
        "os.makedirs(\"../models\", exist_ok=True)\n"
        "os.makedirs(\"models\", exist_ok=True)\n"
        "models_dir = \"../models\" if os.path.exists(\"../models\") else \"models\"\n"
        "best_model_path = os.path.join(models_dir, \"cnn_model.pth\")\n\n"
        "for epoch in range(EPOCHS):\n"
        "    # Training Phase\n"
        "    model.train()\n"
        "    train_loss, train_correct, train_total = 0.0, 0, 0\n"
        "    \n"
        "    for inputs, targets in train_loader:\n"
        "        inputs, targets = inputs.to(device), targets.to(device)\n"
        "        \n"
        "        optimizer.zero_grad()\n"
        "        outputs = model(inputs)\n"
        "        loss = criterion(outputs, targets)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        \n"
        "        train_loss += loss.item() * inputs.size(0)\n"
        "        _, predicted = outputs.max(1)\n"
        "        train_total += targets.size(0)\n"
        "        train_correct += predicted.eq(targets).sum().item()\n"
        "        \n"
        "    train_epoch_loss = train_loss / train_total\n"
        "    train_epoch_acc = train_correct / train_total\n"
        "    \n"
        "    # Validation Phase\n"
        "    model.eval()\n"
        "    val_loss, val_correct, val_total = 0.0, 0, 0\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        for inputs, targets in val_loader:\n"
        "            inputs, targets = inputs.to(device), targets.to(device)\n"
            "            outputs = model(inputs)\n"
        "            loss = criterion(outputs, targets)\n"
        "            \n"
        "            val_loss += loss.item() * inputs.size(0)\n"
        "            _, predicted = outputs.max(1)\n"
        "            val_total += targets.size(0)\n"
        "            val_correct += predicted.eq(targets).sum().item()\n"
        "            \n"
        "    val_epoch_loss = val_loss / val_total\n"
        "    val_epoch_acc = val_correct / val_total\n"
        "    \n"
        "    # Aggiorna lo scheduler\n"
        "    scheduler.step(val_epoch_acc)\n"
        "    \n"
        "    # Salva nello storico\n"
        "    history[\"train_loss\"].append(train_epoch_loss)\n"
        "    history[\"train_acc\"].append(train_epoch_acc)\n"
        "    history[\"val_loss\"].append(val_epoch_loss)\n"
        "    history[\"val_acc\"].append(val_epoch_acc)\n"
        "    \n"
        "    print(f\"Epoca [{epoch+1:02d}/{EPOCHS}]: \"\n"
        "          f\"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f} || \"\n"
        "          f\"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}\")\n"
        "          \n"
        "    # Salvataggio del modello migliore\n"
        "    if val_epoch_acc > best_val_acc:\n"
        "        best_val_acc = val_epoch_acc\n"
        "        torch.save(model.state_dict(), best_model_path)\n"
        "        print(f\"  --> Nuovo miglior modello salvato con accuratezza validazione: {best_val_acc:.4f}\")\n\n"
        "print(\"Training completato!\")"
    ))
    
    # Cell 11: Markdown Plot Curves
    cells.append(nbf.v4.new_markdown_cell(
        "## Grafici delle Curve di Apprendimento\n"
        "Visualizziamo l'andamento della Loss e dell'Accuratezza durante l'addestramento."
    ))
    
    # Cell 12: Code Plot Curves
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(14, 5))\n\n"
        "# Grafico della Loss\n"
        "plt.subplot(1, 2, 1)\n"
        "plt.plot(range(1, EPOCHS + 1), history[\"train_loss\"], label=\"Train Loss\", color=\"royalblue\")\n"
        "plt.plot(range(1, EPOCHS + 1), history[\"val_loss\"], label=\"Val Loss\", color=\"tomato\")\n"
        "plt.title(\"Andamento della Loss durante il Training\")\n"
        "plt.xlabel(\"Epoca\")\n"
        "plt.ylabel(\"Loss\")\n"
        "plt.legend()\n\n"
        "# Grafico dell'Accuratezza\n"
        "plt.subplot(1, 2, 2)\n"
        "plt.plot(range(1, EPOCHS + 1), history[\"train_acc\"], label=\"Train Accuracy\", color=\"royalblue\")\n"
        "plt.plot(range(1, EPOCHS + 1), history[\"val_acc\"], label=\"Val Accuracy\", color=\"tomato\")\n"
        "plt.title(\"Andamento dell'Accuratezza durante il Training\")\n"
        "plt.xlabel(\"Epoca\")\n"
        "plt.ylabel(\"Accuracy\")\n"
        "plt.legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 13: Markdown Test Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## Valutazione sul Test Set\n"
        "Carichiamo i migliori pesi salvati ed eseguiamo la valutazione finale sul Test Set (PrivateTest)."
    ))
    
    # Cell 14: Code Test Evaluation
    cells.append(nbf.v4.new_code_cell(
        "# Carichiamo il modello migliore salvato\n"
        "model.load_state_dict(torch.load(best_model_path, map_location=device))\n"
        "model.eval()\n\n"
        "test_preds = []\n"
        "test_targets = []\n\n"
        "with torch.no_grad():\n"
        "    for inputs, targets in test_loader:\n"
        "        inputs = inputs.to(device)\n"
        "        outputs = model(inputs)\n"
        "        _, predicted = outputs.max(1)\n"
        "        test_preds.extend(predicted.cpu().numpy())\n"
        "        test_targets.extend(targets.numpy())\n\n"
        "test_acc = accuracy_score(test_targets, test_preds)\n"
        "print(f\"Accuratezza complessiva della CNN sul Test Set: {test_acc:.4f}\\n\")\n\n"
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
        "print(classification_report(test_targets, test_preds, target_names=target_names))"
    ))
    
    # Cell 15: Code Confusion Matrix
    cells.append(nbf.v4.new_code_cell(
        "cm = confusion_matrix(test_targets, test_preds)\n"
        "cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    cm_normalized, \n"
        "    annot=True, \n"
        "    fmt=\".2f\", \n"
        "    cmap=\"Oranges\", \n"
        "    xticklabels=target_names, \n"
        "    yticklabels=target_names\n"
        ")\n"
        "plt.title(\"Matrice di Confusione Normalizzata (CNN PyTorch)\", fontsize=14, fontweight='bold')\n"
        "plt.ylabel(\"Etichetta Reale\", fontsize=12)\n"
        "plt.xlabel(\"Etichetta Predetta\", fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/03_deep_learning_approach.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 03_deep_learning_approach.ipynb generato con successo!")

if __name__ == "__main__":
    create_dl_notebook()
