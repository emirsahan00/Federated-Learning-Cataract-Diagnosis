import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Net
import os

def load_model_weights(model_path):
    try:
        # Model ağırlıklarını yükle
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = Net(num_classes=2)
        
        # Ağırlıkları modele yükle
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model yüklendi. Kaydedilen accuracy: {checkpoint.get('accuracy', 'N/A')}")
            print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}%")
            print(f"Sınıf bazlı doğruluklar:")
            for cls, acc in checkpoint.get('class_accuracies', {}).items():
                print(f"- {cls}: {acc:.2f}%")
            print(f"Sınıf eşleştirmesi: {checkpoint.get('class_mapping', {0: 'Normal', 1: 'Katarakt'})}")
        else:
            print("Hata: Model ağırlıkları beklenen formatta değil!")
            return None
        
        model.eval()
        return model
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        print("\nModel çıktıları (logits):", outputs[0].tolist())
        
        # Sınıf ağırlıklarını hesapla (örnek olarak eşit ağırlıklar kullanıyoruz)
        class_weights = torch.tensor([0.5, 0.5])
        
        # Softmax ve sınıf ağırlıklarını uygula
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        weighted_probs = probabilities * class_weights.view(1, -1)
        weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)
        
        print("Ağırlıklı olasılıklar:", weighted_probs[0].tolist())
        
        _, predicted = torch.max(weighted_probs, 1)
        
        # Her sınıf için ağırlıklı olasılıkları al
        normal_prob = weighted_probs[0][0].item()
        cataract_prob = weighted_probs[0][1].item()
        
        # En yüksek olasılıklı sınıfı ve değerini döndür
        if normal_prob > cataract_prob:
            return 0, normal_prob, cataract_prob
        else:
            return 1, cataract_prob, normal_prob

def main():
    # Model ağırlıklarının yolu - en son eğitilen modelin yolunu kullan
    model_path = os.path.join('outputs', '2025-05-22', '19-07-11', 'C:/Courses/EmirFederatedLearning/federated/best_model.pth')
    
    # Test edilecek resmin yolu
    image_path = 'image.png'  # Test edeceğiniz resmin yolunu buraya yazın

    # Model ağırlıklarını yükle
    model = load_model_weights(model_path)
    if model is None:
        print("Model yüklenemedi, program sonlandırılıyor.")
        return

    # Resmi yükle ve tahmin yap
    try:
        image = preprocess_image(image_path)
        prediction, pred_prob, other_prob = predict(model, image)
        
        # Sonucu göster
        plt.figure(figsize=(10, 6))
        plt.imshow(Image.open(image_path))
        plt.title(f'Tahmin: {"Katarakt" if prediction == 1 else "Normal"}\n'
                 f'Güven: {pred_prob:.2%}\n'
                 f'Diğer sınıf olasılığı: {other_prob:.2%}')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Tahmin yapılırken hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()