import hydra  # Hydra kütüphanesini kullanarak dinamik yapılandırma sağlar.
from hydra.utils import call, instantiate  # call: fonksiyon çağırma; instantiate: nesne oluşturma.
from omegaconf import DictConfig, OmegaConf  # DictConfig: yapılandırma türü; OmegaConf: yapılandırma işlemleri.

# Basit bir toplama işlemi yapan fonksiyon tanımı.
def function_test(x: int, y: int):
    """A simple function that adds up two integers."""
    print(f"`function_test` received: {x = }, and {y = }")  # Parametrelerin alındığını ekrana yazdırır.
    result = x + y  # x ve y parametrelerini toplar.
    print(f"{result = }")  # Sonucu ekrana yazdırır.

# Bir sınıf tanımlaması yapılır.
class MyClass:
    """A simple class."""
    def __init__(self, x):
        self.x = x  # x parametresi sınıf değişkenine atanır.

    def print_x_squared(self):
        print(f"{self.x**2 = }")  # x'in karesini ekrana yazdırır.

# Hydra kullanılarak nesne oluşturma işlemini örnekleyen daha karmaşık bir sınıf.
class MyComplexClass:
    """A class with some Hydra magic inside."""
    def __init__(self, my_object: MyClass):
        self.object = my_object  # MyClass tipindeki bir nesneyi saklar.

    def instantiate_child(self, value):
        # Hydra `instantiate` fonksiyonu ile yeni bir MyClass nesnesi oluşturur.
        self.object = instantiate(self.object, x=value)

# Hydra ile yapılandırmayı kullanarak program başlatılır.
@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):
    # Yapılandırmayı YAML formatında ekrana yazdırır.
    print(OmegaConf.to_yaml(cfg))

    ## Temel İşlemler
    print("--------" * 7)
    # Yapılandırmadaki elemanlara kolayca erişim sağlar.
    print(f"{cfg.foo = }")  # cfg'deki 'foo' elemanını yazdırır.
    print(f"{cfg.bar.baz = }")  # 'bar' altındaki 'baz' elemanını yazdırır.
    print(f"{cfg.bar.more = }")  # 'bar' altındaki 'more' elemanını yazdırır.
    print(f"{cfg.bar.more.blabla = }")  # 'more' altındaki 'blabla' elemanını yazdırır.

    ## Fonksiyon Çağrımı
    print("--------" * 7)
    # Hydra yapılandırmasındaki `my_func` fonksiyonunu çağırır.
    call(cfg.my_func)

    # Argümanları çalıştırma sırasında değiştirebilir.
    call(cfg.my_func, x=99)  # `x` değerini geçersiz kılarak fonksiyonu çağırır.

    # Eksik argümanlar için `partial` kullanımı.
    partial_fn = call(cfg.my_partial_func)  # `_partial_` yapılandırma değeri `True` olmalıdır.
    partial_fn(y=2023)  # Eksik `y` argümanını tamamlar ve fonksiyonu çağırır.

    ## Nesne Oluşturma
    # Hydra `instantiate` ile nesne oluşturur.
    object: MyClass = instantiate(cfg.my_object)
    object.print_x_squared()  # Oluşturulan nesnenin bir yöntemini çağırır.

    ## Karmaşık Nesne Örnekleri
    print("--------" * 7)
    # İç içe nesnelerle çalışır.
    obj = instantiate(cfg.my_complex_object)
    print(obj.object.x)  # İç nesnedeki x değerini yazdırır.

    # Recursive olmayan nesne oluşturma.
    obj = instantiate(cfg.my_complex_object_non_recursive)
    print(obj.object)  # Nesnenin yapılandırmasını yazdırır.
    obj.instantiate_child(9999)  # Yeni bir çocuk nesne oluşturur.
    print(obj.object.x)  # Yeni oluşturulan çocuk nesnenin `x` değerini yazdırır.

    # Yapılandırmadan PyTorch modeli oluşturma.
    model = instantiate(cfg.toy_model)
    num_parameters = sum([p.numel() for p in model.state_dict().values()])  # Modeldeki toplam parametre sayısını hesaplar.
    print(f"{cfg.toy_model} has: {num_parameters} parameters")  # Modelin parametre sayısını yazdırır.

# Program başlangıç noktası.
if __name__ == "__main__":
    main()
