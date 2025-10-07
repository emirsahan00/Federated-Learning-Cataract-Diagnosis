# OrderedDict: Sıralı bir sözlük (dictionary) veri yapısı sağlar.
from collections import OrderedDict

# PyTorch kütüphanesi: Derin öğrenme modelleri ve işlemleri için kullanılır.
import torch

# Hydra: Yapılandırmaları kolayca yönetmek ve dinamik olarak örnekleme yapmak için kullanılan bir araç.
from hydra.utils import instantiate

# omegaconf: Konfigürasyon yönetimi için kullanılan bir kütüphane. DictConfig, yapılandırmaları tutar.
from omegaconf import DictConfig

# model.py dosyasından Net sınıfı (sinir ağı modeli) ve test fonksiyonu ithal edilir.
from model import Net, test

# Global değişken olarak model ağırlıklarını saklayacağız
global_model_weights = None

def get_on_fit_config(config: DictConfig):
    """
    Sunucunun fit (eğitim) işlemi için gerekli konfigürasyonu döndüren bir fonksiyon oluşturur.

    Args:
        config (DictConfig): Yapılandırma dosyasından gelen parametreler.

    Returns:
        fit_config_fn (function): Eğitim sırasında kullanılacak parametreleri döndüren bir fonksiyon.
    """
    def fit_config_fn(server_round: int):
        """
        Her tur için eğitim parametrelerini döndürür.
        
        Args:
            server_round (int): Eğitim turunun (round) sayısı.

        Returns:
            dict: Öğrenme oranı (lr), momentum ve yerel epoch sayısı.
        """
        return {
            "lr": config.lr,  # Öğrenme oranını yapılandırmadan alır.
            "momentum": config.momentum,  # Momentum parametresini yapılandırmadan alır.
            "local_epochs": config.local_epochs,  # Yerel epoch sayısını yapılandırmadan alır.
        }

    return fit_config_fn  # fit_config_fn fonksiyonunu döndür.


def get_evalulate_fn(model_cfg: DictConfig, testloader):
    """
    Küresel modeli değerlendirmek için bir fonksiyon döndüren yardımcı bir fonksiyon.

    Args:
        model_cfg (DictConfig): Modelin yapılandırması.
        testloader (DataLoader): Test veri yükleyicisi.

    Returns:
        evaluate_fn (function): Değerlendirme işlemini gerçekleştiren bir fonksiyon.
    """
    def evaluate_fn(server_round: int, parameters, config):
        """
        Modeli değerlendirir ve kayıp (loss) ile doğruluk (accuracy) değerlerini döndürür.
        
        Args:
            server_round (int): Sunucudaki eğitim turunun sayısı.
            parameters (list): Küresel modelin parametreleri.
            config (dict): Değerlendirme sırasında kullanılabilecek ek parametreler.

        Returns:
            tuple: Kayıp değeri ve doğruluk metriğini içeren sözlük.
        """
        global global_model_weights
        
        # Modelin yapılandırmasını kullanarak yeni bir model örneği oluştur.
        model = instantiate(model_cfg)

        # Cihazı belirle: CUDA (GPU) varsa onu kullan, yoksa CPU'yu kullan.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Modelin mevcut parametrelerini güncellemek için, verilen parametreleri modelin state_dict anahtarlarıyla eşleştir.
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        
        # Güncellenmiş parametreleri modele yükle.
        model.load_state_dict(state_dict, strict=True)

        # Modeli test et ve kayıp ile doğruluk değerlerini al.
        loss, accuracy, class_accuracies = test(model, testloader, device)

        # Global model ağırlıklarını sakla
        global_model_weights = parameters
        print(f"Global model ağırlıkları güncellendi - Round {server_round}")
        print(f"Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Class accuracies - Normal: {class_accuracies[0]:.4f}, Cataract: {class_accuracies[1]:.4f}")

        # Kayıp ve doğruluk değerlerini döndür.
        return {"loss": loss}, {"accuracy": accuracy}

    return evaluate_fn  # evaluate_fn fonksiyonunu döndür.

def get_global_model_weights():
    return global_model_weights
