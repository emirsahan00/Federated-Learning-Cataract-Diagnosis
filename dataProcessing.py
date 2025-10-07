import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
from PIL import Image

class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Ana dizin yolu (cataract ve normal klasörlerini içeren)
            transform: Görüntülere uygulanacak dönüşümler
            is_train: Eğitim seti için True, test seti için False
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.classes = ['normal', 'cataract']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Her sınıf için görüntüleri yükle
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Desteklenen görüntü formatları
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(valid_extensions):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Veri seti yüklendi:")
        for i, cls in enumerate(self.classes):
            count = sum(1 for label in self.labels if label == i)
            print(f"- {cls}: {count} örnek")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_cataract_dataset(data_path):
    """
    Veri setini yükle ve dönüşümleri uygula
    """
    # Eğitim için dönüşümler
    train_transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test için dönüşümler
    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tüm veriyi tek bir dataset olarak yükle
    full_dataset = CataractDataset(data_path, transform=train_transform, is_train=True)
    
    # Veriyi train ve test olarak böl (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(2023)
    )
    
    # Test seti için dönüşümleri güncelle
    testset.dataset.transform = test_transform
    testset.dataset.is_train = False

    return trainset, testset

def prepare_dataset(num_clients: int, batch_size: int):
    """Veri setini hazırla ve istemciler arasında böl."""
    # Veri setini yükle
    transform = Compose([
        Resize((96, 96)),  # Daha küçük görüntü boyutu
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Veri setini yükle
    dataset = CataractDataset(root_dir='LastCataractDataset', transform=transform, is_train=True)
    
    # Veri setini istemciler arasında böl
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(dataset, lengths)
    
    # Her istemci için veri yükleyicileri oluştur
    trainloaders = []
    validationloaders = []
    
    for ds in datasets:
        # Eğitim ve doğrulama setlerini ayır
        train_size = int(0.8 * len(ds))
        val_size = len(ds) - train_size
        train_dataset, val_dataset = random_split(ds, [train_size, val_size])
        
        # Veri yükleyicileri oluştur
        trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True  # Son eksik batch'i atla
        )
        
        validationloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True  # Son eksik batch'i atla
        )
        
        trainloaders.append(trainloader)
        validationloaders.append(validationloader)
    
    # Test seti için veri yükleyici
    testloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True  # Son eksik batch'i atla
    )
    
    return trainloaders, validationloaders, testloader