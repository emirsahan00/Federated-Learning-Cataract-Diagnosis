import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.parameter import ndarrays_to_parameters

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        
        # Giriş: 224x224x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 224x224x16
        self.pool1 = nn.MaxPool2d(2, 2)  # 112x112x16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 112x112x32
        self.pool2 = nn.MaxPool2d(2, 2)  # 56x56x32
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 56x56x64
        self.pool3 = nn.MaxPool2d(2, 2)  # 28x28x64
        
        # Global average pooling ile boyutu sabitle
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1x64
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.5)  # Dropout oranını 0.5'e çıkardım
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Evrişim ve havuzlama katmanları
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def predict(self, x, device="cpu", temperature=1.0):
        """
        Tek bir görüntü için tahmin yapar ve güven değerlerini döndürür.
        
        Args:
            x (torch.Tensor): Giriş görüntüsü (1, 3, 224, 224)
            device (str): Kullanılacak cihaz
            temperature (float): Softmax sıcaklık parametresi
            
        Returns:
            tuple: (sınıf, güven değeri)
        """
        self.eval()
        self.to(device)
        x = x.to(device)
        
        with torch.no_grad():
            outputs = self(x)
            # Sıcaklık parametresi ile softmax uygula
            scaled_outputs = outputs / temperature
            probabilities = F.softmax(scaled_outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Sınıf isimlerini belirle
            class_names = {0: "Normal", 1: "Katarakt"}
            
            return class_names[predicted.item()], confidence.item()

def train(net, trainloader, optimizer, epochs, device: str):
    """Verilen modeli eğitim veri kümesi üzerinde eğit."""
    # Sınıf ağırlıklarını hesapla
    class_counts = torch.zeros(2)
    for _, labels in trainloader:
        for label in labels:
            class_counts[label] += 1
    
    # Sınıf ağırlıklarını hesapla (ters orantılı)
    class_weights = 1.0 / (class_counts + 1e-6)  # Sıfıra bölünmeyi önle
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    net.train()
    net.to(device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    total_loss = 0.0
    num_batches = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_epoch_loss = epoch_loss / epoch_batches
        scheduler.step(avg_epoch_loss)
        
        # En iyi modeli kaydet
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
        
        total_loss += epoch_loss
        num_batches += epoch_batches
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return total_loss / num_batches

def test(net, testloader, device: str):
    """Verilen modeli test veri kümesi üzerinde doğrula."""
    # Sınıf ağırlıklarını hesapla
    class_counts = torch.zeros(2)
    for _, labels in testloader:
        for label in labels:
            class_counts[label] += 1
    
    # Sınıf ağırlıklarını hesapla (ters orantılı)
    class_weights = 1.0 / (class_counts + 1e-6)  # Sıfıra bölünmeyi önle
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    correct, total = 0, 0
    class_correct = [0, 0]  # Her sınıf için doğru tahminler
    class_total = [0, 0]    # Her sınıf için toplam örnekler
    total_loss = 0.0
    num_batches = 0
    
    net.to(device)
    net.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Sınıf bazlı doğruluk
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    accuracy = correct / total
    class_accuracies = [class_correct[i] / (class_total[i] + 1e-6) for i in range(2)]
    avg_loss = total_loss / num_batches
    
    return avg_loss, accuracy, class_accuracies

def model_to_parameters(model):
    """Model parametrelerini numpy dizilerine dönüştür."""
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Model parametreleri çıkarıldı!")
    return parameters

def get_initial_parameters():
    """Modeli başlat ve başlangıç parametrelerini döndür."""
    model = Net(num_classes=2)
    return model_to_parameters(model)

def load_model_and_predict(image_path, model_weights_path, device="cpu"):
    """
    Kaydedilmiş model ağırlıklarını yükler ve verilen görüntü için tahmin yapar.
    
    Args:
        image_path (str): Tahmin yapılacak görüntünün yolu
        model_weights_path (str): Model ağırlıklarının kaydedildiği dosya yolu
        device (str): Kullanılacak cihaz
        
    Returns:
        tuple: (sınıf, güven değeri)
    """
    # Model oluştur
    model = Net(num_classes=2)
    
    # Ağırlıkları yükle
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Görüntüyü yükle ve ön işle
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Batch boyutu ekle
    
    # Tahmin yap
    return model.predict(image_tensor, device)