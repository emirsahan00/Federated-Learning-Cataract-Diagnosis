from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
import io
import os
from model import Net

app = Flask(__name__)

# Global model değişkeni
model = None

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

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        
        # Sınıf ağırlıklarını hesapla (örnek olarak eşit ağırlıklar kullanıyoruz)
        class_weights = torch.tensor([0.5, 0.5])
        
        # Softmax ve sınıf ağırlıklarını uygula
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        weighted_probs = probabilities * class_weights.view(1, -1)
        weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)
        
        # Her sınıf için ağırlıklı olasılıkları al
        normal_prob = weighted_probs[0][0].item()
        cataract_prob = weighted_probs[0][1].item()
        
        # En yüksek olasılıklı sınıfı ve değerini döndür
        if normal_prob > cataract_prob:
            return 0, normal_prob, cataract_prob
        else:
            return 1, cataract_prob, normal_prob

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Global model kontrolü
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model yüklenmemiş. Lütfen önce modeli yükleyin.'
            })
        
        # Base64 kodlanmış resim verisini al
        if 'image' not in request.json:
            return jsonify({
                'success': False,
                'error': 'Request içinde "image" anahtarı bulunamadı'
            })
        
        image_data = request.json['image']
        
        # Base64'ten resim verisini çöz
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resmi ön işleme
        processed_image = preprocess_image(image)
        
        # Tahmin yap
        prediction, pred_prob, other_prob = predict_image(model, processed_image)
        
        # Sınıf eşleştirmesi
        class_names = {0: 'Normal', 1: 'Katarakt'}
        predicted_class = class_names[prediction]
        
        # Tüm sınıfların olasılıklarını hazırla
        all_predictions = {
            'Normal': float(other_prob if prediction == 1 else pred_prob),
            'Katarakt': float(pred_prob if prediction == 1 else other_prob)
        }
        
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': float(pred_prob),
            'all_predictions': all_predictions,
            'raw_outputs': {
                'normal_probability': float(other_prob if prediction == 1 else pred_prob),
                'cataract_probability': float(pred_prob if prediction == 1 else other_prob)
            }
        }
        
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """API'nin çalışıp çalışmadığını kontrol eden endpoint"""
    return jsonify({
        'success': True,
        'message': 'API çalışıyor',
        'model_loaded': model is not None
    })

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Model yükleme endpoint'i"""
    global model
    
    try:
        # Model yolunu al
        if 'model_path' in request.json:
            model_path = request.json['model_path']
        else:
            # Varsayılan model yolu
            model_path = os.path.join('outputs', '2025-05-22', '19-07-11', 'C:/Courses/EmirFederatedLearning/federated/best_model.pth')
        
        # Modeli yükle
        loaded_model = load_model_weights(model_path)
        
        if loaded_model is not None:
            model = loaded_model
            return jsonify({
                'success': True,
                'message': f'Model başarıyla yüklendi: {model_path}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model yüklenemedi'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Model yüklenirken hata: {str(e)}'
        })

if __name__ == '__main__':
    # Uygulama başlatılırken modeli yükle
    print("Modeli yükleniyor...")
    model_path = os.path.join('outputs', '2025-05-22', '19-07-11', 'C:/Courses/EmirFederatedLearning/federatedCursor/best_model.pth')
    model = load_model_weights(model_path)
    
    if model is None:
        print("Uyarı: Model yüklenemedi. /load_model endpoint'ini kullanarak modeli yükleyebilirsiniz.")
    else:
        print("Model başarıyla yüklendi!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)