import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Inizio scaricamento del dataset FER-2013 da Hugging Face...")
    # Carica il dataset
    dataset = load_dataset("abhilash88/fer2013-enhanced")
    
    print("\nDataset caricato. Struttura:")
    print(dataset)
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Per ogni split, estraiamo le immagini e le etichette
    splits = ["train", "validation", "test"]
    data_dict = {}
    
    for split in splits:
        print(f"\nElaborazione dello split: {split}...")
        hf_split = dataset[split]
        
        images = []
        labels = []
        
        # Iteriamo con tqdm per mostrare il progresso
        for item in tqdm(hf_split, desc=f"Processando {split}"):
            img_arr = np.array(item["image"], dtype=np.uint8)
            images.append(img_arr)
            labels.append(item["emotion"])
            
        data_dict[f"{split}_images"] = np.array(images, dtype=np.uint8)
        data_dict[f"{split}_labels"] = np.array(labels, dtype=np.int64)
        
        print(f"Dimensioni per {split}:")
        print(f"  Immagini: {data_dict[f'{split}_images'].shape}")
        print(f"  Etichette: {data_dict[f'{split}_labels'].shape}")

    # Salva in un unico archivio compresso NPZ
    output_path = os.path.join(data_dir, "fer2013.npz")
    print(f"\nSalvataggio del dataset in corso in: {output_path}...")
    np.savez_compressed(
        output_path,
        train_images=data_dict["train_images"],
        train_labels=data_dict["train_labels"],
        val_images=data_dict["validation_images"],
        val_labels=data_dict["validation_labels"],
        test_images=data_dict["test_images"],
        test_labels=data_dict["test_labels"]
    )
    print("Salvataggio completato con successo!")

if __name__ == "__main__":
    main()
