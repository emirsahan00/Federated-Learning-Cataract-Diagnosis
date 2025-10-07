# Gerekli kütüphanelerin import edilmesi
import pickle  # Veri serileştirme ve dosyaya kaydetme işlemleri için kullanılır
from pathlib import Path  # Dosya ve dizin yollarını yönetmek için kullanılır
import matplotlib
matplotlib.use('Agg')  # Qt backend yerine Agg backend'i kullan
import matplotlib.pyplot as plt  # Grafik çizimi için gerekli kütüphane
import torch
import os
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"  # Bellek monitörünü devre dışı bırak
import ray  # Ray için import

# Ray'i başlat
try:
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,
        num_cpus=1,
        logging_level="ERROR"
    )
except Exception as e:
    print(f"Ray başlatılırken hata oluştu: {e}")
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,
        num_cpus=1,
        logging_level="ERROR"
    )

# Federated Learning framework'ü olan Flower'ın modüllerinin import edilmesi
import flwr as fl  

# Hydra, yapılandırma yönetimini kolaylaştırmak için kullanılan bir kütüphane
import hydra
from hydra.core.hydra_config import HydraConfig  # Hydra'nın çalışma zamanındaki yapılandırma verilerini almak için
from hydra.utils import call, instantiate  # Hydra yapılandırmalarını işlemek için araçlar

# Omegaconf, yapılandırma işlemleri için kullanılır
from omegaconf import DictConfig, OmegaConf

# Yerel modüllerin import edilmesi
from client import generate_client_fn  # Müşteri oluşturma fonksiyonları
from dataProcessing import prepare_dataset  # Veri setini hazırlama işlemleri
from server import get_evalulate_fn, get_on_fit_config, get_global_model_weights  # Sunucu için yapılandırma fonksiyonları

# Hydra ile yapılandırmayı başlatan ana fonksiyon
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Ana çalışma fonksiyonu. Federated Learning simülasyonu çalıştırır.

    Args:
        cfg (DictConfig): Hydra tarafından sağlanan yapılandırma nesnesi.
    """
    try:
        print(OmegaConf.to_yaml(cfg))  # Yapılandırmayı YAML formatında yazdır
        save_path = HydraConfig.get().runtime.output_dir  # Çıkış dosyalarının kaydedileceği yol

        ## 2. Veri setini hazırlama
        # Eğitim, doğrulama ve test veri yükleyicilerini oluştur
        trainloaders, validationloaders, testloader = prepare_dataset(
            cfg.num_clients, cfg.batch_size
        )

        ## 3. Müşterileri tanımlama
        # Müşteri oluşturma fonksiyonunu ayarla
        client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)

        ## 4. Stratejiyi tanımlama
        # Federated Learning stratejisini instantiate ederek oluştur
        strategy = instantiate(
            cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, testloader)
        )

        ## 5. Simülasyonu başlatma
        # Federated Learning simülasyonunu başlat
        history = fl.simulation.start_simulation(
            client_fn=client_fn,  # Müşteri fonksiyonu
            num_clients=cfg.num_clients,  # Toplam müşteri sayısı
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),  # Sunucu yapılandırması
            strategy=strategy,  # Federated Learning stratejisi
            client_resources={"num_cpus": 1, "num_gpus": 0.0},  # Müşteri başına kullanılacak kaynaklar
        )

        # Sonuçları kaydet
        results = {"history": history}
        print("Eğitim tamamlandı!")
        print("HISTORY:", history.metrics_centralized)

        # Model ağırlıklarını kaydet
        model_weights_path = Path(save_path) / "model_weights.pth"
        print(f"Model ağırlıkları kaydedilecek yol: {model_weights_path}")
        
        # Model örneği oluştur
        model = instantiate(cfg.model)
        print("Model örneği oluşturuldu")
        
        # Server'dan global model ağırlıklarını al
        global_weights = get_global_model_weights()
        
        if global_weights is not None:
            print("Global model ağırlıkları bulundu")
            try:
                # Ağırlıkları kontrol et
                print("Ağırlık boyutları:")
                for i, (k, v) in enumerate(zip(model.state_dict().keys(), global_weights)):
                    print(f"{k}: {v.shape if hasattr(v, 'shape') else len(v)}")
                
                state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), global_weights)}
                model.load_state_dict(state_dict)
                print("Model ağırlıkları başarıyla yüklendi")
                
                # Test veri seti üzerinde hızlı bir doğrulama yap
                _, _, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                
                correct = 0
                total = 0
                class_correct = [0, 0]  # Normal ve Katarakt için doğru tahminler
                class_total = [0, 0]    # Normal ve Katarakt için toplam örnekler
                
                with torch.no_grad():
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # Sınıf bazlı doğruluk
                        for i in range(len(labels)):
                            label = labels[i]
                            class_correct[label] += (predicted[i] == label).item()
                            class_total[label] += 1
                
                print("\nTest Sonuçları:")
                print(f"Toplam Doğruluk: {100 * correct / total:.2f}%")
                print(f"Normal Sınıfı Doğruluk: {100 * class_correct[0] / class_total[0]:.2f}%")
                print(f"Katarakt Sınıfı Doğruluk: {100 * class_correct[1] / class_total[1]:.2f}%")
                
                # Modeli kaydet
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'accuracy': history.metrics_centralized["accuracy"][-1][1] if "accuracy" in history.metrics_centralized else 0.0,
                    'class_mapping': {0: 'Normal', 1: 'Katarakt'},
                    'test_accuracy': 100 * correct / total,
                    'class_accuracies': {
                        'Normal': 100 * class_correct[0] / class_total[0],
                        'Katarakt': 100 * class_correct[1] / class_total[1]
                    }
                }
                
                # Kaydetme işlemi
                torch.save(save_dict, str(model_weights_path))
                print(f"\nModel ağırlıkları başarıyla kaydedildi: {model_weights_path}")
                print(f"Son accuracy değeri: {history.metrics_centralized['accuracy'][-1][1] if 'accuracy' in history.metrics_centralized else 'N/A'}")
                
            except Exception as e:
                print(f"Model kaydedilirken hata oluştu: {str(e)}")
        else:
            print("Uyarı: Global model ağırlıkları bulunamadı!")

        # Accuracy grafiğini çiz
        if "accuracy" in history.metrics_centralized:
            accuracies = [metric[1] for metric in history.metrics_centralized["accuracy"]]
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
            plt.title("with FedAdagrad Strategy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig(Path(save_path) / "accuracy_plot.png")
            plt.close()

        # Sonuçları kaydet
        results_path = Path(save_path) / "results.pkl"
        with open(str(results_path), "wb") as h:
            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {str(e)}")
    finally:
        ray.shutdown()

# Bu dosya doğrudan çalıştırıldığında ana fonksiyonu çağır
if __name__ == "__main__":
    main()