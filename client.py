from collections import OrderedDict  # Sıralı sözlük oluşturmak için gerekli modül.
from typing import Dict, Tuple  # Tür belirtimleri için kullanılan modüller.

import flwr as fl  # Flower kütüphanesi, federatif öğrenme için kullanılıyor.
import torch  # PyTorch, derin öğrenme modellerini oluşturmak ve çalıştırmak için kullanılıyor.
from flwr.common import NDArrays, Scalar  # Flower'ın ortak yapılarını içeren modül.
from hydra.utils import instantiate  # Hydra ile tanımlanan yapılandırma dosyalarını nesnelere dönüştürmek için.

from model import test, train  # Model eğitimi ve değerlendirme işlevlerini içeren modüller.


class FlowerClient(fl.client.NumPyClient):
    """Flower'ın standart istemci sınıfını temsil eder."""

    def __init__(self, trainloader, vallodaer, model_cfg) -> None:
        super().__init__()  # Üst sınıfın kurucusunu çağırarak temel özellikleri başlatır.

        self.trainloader = trainloader  # Eğitim veri yükleyicisini atar.
        self.valloader = vallodaer  # Doğrulama veri yükleyicisini atar.

        # Modeli esnek bir şekilde tanımlamak için Hydra yapılandırmasını kullanıyoruz.
        # Yapılandırma dosyasındaki model nesnesini oluşturuyor.
        self.model = instantiate(model_cfg)

        # Eğitim ve doğrulama işlemleri için cihaz seçimi yapar (GPU varsa "cuda:0" kullanılır).
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        # Sunucudan gelen parametreleri modelin durumu ile eşleştirir.
        params_dict = zip(self.model.state_dict().keys(), parameters)  # Model parametre adlarını ve değerlerini eşler.
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})  # Tensor'lara dönüştürerek sıralı bir sözlük oluşturur.
        self.model.load_state_dict(state_dict, strict=True)  # Model durumunu günceller.

    def get_parameters(self, config: Dict[str, Scalar]):
        # Modelin parametrelerini alır ve sunucuya gönderilmek üzere NumPy dizilerine dönüştürür.
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Sunucudan gelen parametrelerle modeli günceller.
        self.set_parameters(parameters)

        # Yapılandırmadan öğrenme oranı, momentum ve yerel epoch sayısını alır.
        lr = config["lr"]  # Öğrenme oranı.
        momentum = config["momentum"]  # Momentum değeri.
        epochs = config["local_epochs"]  # Yerel eğitim döngüsü sayısı.
        weight_decay = config.get("weight_decay", 0.0001)  # L2 regularizasyon

        # Adam optimizasyon algoritmasını oluşturur.
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Yerel eğitim işlemini başlatır.
        train(self.model, self.trainloader, optim, epochs, self.device)

        # Eğitimden sonra model parametrelerini, veri miktarını ve boş bir meta veriyi döndürür.
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Sunucudan gelen parametrelerle modeli günceller.
        self.set_parameters(parameters)

        # Modeli doğrulama veri yükleyicisi üzerinde test eder ve kayıp, doğruluk ve sınıf doğruluklarını döndürür.
        loss, accuracy, class_accuracies = test(self.model, self.valloader, self.device)

        # Doğrulama sonuçlarını, veri miktarını ve doğruluk metriğini döndürür.
        return float(loss), len(self.valloader), {"accuracy": accuracy, "class_accuracies": class_accuracies}


def generate_client_fn(trainloaders, valloaders, model_cfg):
    """FlowerClient nesneleri oluşturan bir işlev döndürür."""

    def client_fn(cid: str):
        # Belirli bir istemci kimliği (cid) için FlowerClient nesnesi oluşturur.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],  # Eğitim yükleyicisini seçer.
            vallodaer=valloaders[int(cid)],  # Doğrulama yükleyicisini seçer.
            model_cfg=model_cfg,  # Model yapılandırmasını kullanır.
        )  # İstemci nesnesi döndürülür..to_client()

    return client_fn  # İstemci oluşturma işlevini döndürür.
