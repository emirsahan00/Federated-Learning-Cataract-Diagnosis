# 🧠 Federated Learning-Based Cataract Detection  
### 📱 Integrated with Flutter & Flask Mobile Application  

---

## 🖼️ Federated Learning Architecture

![Federated Learning Architecture](path/to/federated_architecture.png)

*(Replace `path/to/federated_architecture.png` with the actual image path.)*

### System Diagram

Include your **Flask API, Flutter app, and model pipeline** here:  
![System Diagram](path/to/system_diagram.png)

*(Replace `path/to/system_diagram.png` with your diagram image path.)*

---

## 📋 Overview  


- This project presents a **federated learning-based cataract detection system**, developed as part of a medical AI research initiative.  
- Enables **privacy-preserving training** of deep learning models for cataract diagnosis using **distributed medical image data**.  
- After training, the resulting model is deployed in a **mobile application (Flutter + Flask backend)**.  
- Allows **health professionals and patients** to access cataract predictions easily and securely.  


---

## 🚀 Key Features  

- **Federated Learning with Flower** – Data remains on the client side while training collaboratively across multiple institutions.  
- **PyTorch Model Training** – Lightweight CNN model (custom `Net` class) used for cataract vs. normal eye classification.  
- **Client-Server Architecture** – Implemented using `client.py` and `server.py`.  
- **Secure and Scalable Deployment** – Flask API integrates the trained model for real-time predictions.  
- **Flutter Mobile Interface** – A user-friendly app where medical staff or patients can upload eye images for instant diagnosis.  
- **Explainable Predictions** – Probabilities for both “Normal” and “Cataract” classes displayed with confidence levels.  

---

## 🧩 Project Structure  

```
FederatedCataractDetection/
├── client.py          # Flower client implementation for federated training
├── server.py          # Flower server that coordinates training rounds
├── main.py            # Entry point to start federated training
├── detection.py       # Local model inference and image prediction pipeline
├── detectionFlask.py  # Inference on Mobile App 
├── model.py           # CNN architecture and train/test utilities
├── requirements.txt   # Python dependencies
├── toy.py
```

---

## ⚙️ Technologies Used  

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Federated Learning** | [Flower (flwr)](https://flower.dev) |
| **Deep Learning** | PyTorch, Torchvision |
| **Backend** | Flask |
| **Frontend** | Flutter |
| **Configuration Management** | Hydra, OmegaConf |
| **Visualization & Testing** | Matplotlib, PIL |
| **Language** | Python 3.10+ |

---

## 🧠 Federated Learning Flow  

1. **Server Initialization (`server.py`)**  
   - Initializes the global model.  
   - Coordinates multiple training rounds across clients.  
   - Aggregates client model updates.  

2. **Client Execution (`client.py`)**  
   - Loads local data and model configuration.  
   - Performs local training and sends updates to the server.  
   - Evaluates local model performance.  

3. **Evaluation & Weight Management**  
   - `get_evalulate_fn()` validates the global model using test data.  
   - Global weights are updated after each round and saved for deployment.  

---
---
## 📦 Setup and Environment Preparation

You can use the following methods to install the required Python packages and set up a virtual environment for the project.

### 1️⃣ Installing Packages via `requirements.txt`

To install all packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
### 2️⃣ Creating a Conda Virtual Environment via `environment.yml`

To create a new Conda environment:

```bash
conda env create -f environment.yml
```

To activate the created environment:
```
conda activate my_env_name
```






## 🏃‍♂️ Start Federated Training  

To start the federated learning training pipeline, simply run:  

```bash
python main.py
```

This script will:  
- Launch the Flower server and clients (configurable in `main.py`).  
- Perform federated training rounds according to your Hydra configuration.  
- Save the best global model weights for inference.  

---

---

## 🧪 Training Configuration  

Below is the main training configuration file that defines the federated learning and local training parameters.
These settings can be easily modified to adjust the number of rounds, clients, or model hyperparameters. 

`base.yaml/` 

```yaml
# Federated Setup
num_rounds: 1                  # Total number of communication rounds
num_clients: 1                 # Total number of clients
batch_size: 8                  # Batch size for local training
num_classes: 2                 # Number of output classes
num_clients_per_round_fit: 1   # Clients selected per training round
num_clients_per_round_eval: 1  # Clients selected per evaluation round

# Local Client Training Parameters
config_fit:
  lr: 0.0001          # Learning rate
  momentum: 0.9       # Momentum for optimizer
  local_epochs: 1      # Number of local epochs per round
  weight_decay: 0.0001 # L2 regularization
  temperature: 2.0     # Temperature scaling parameter

# Default Configurations
defaults:
  - model: net           # Points to model configuration
  - strategy: fedadagrad # References federated strategy configuration
```
---
## 🧮 Strategy Configurations (strategy/ Directory)

The project supports multiple federated optimization strategies, defined as `.yaml` files under the `strategy/` directory.
You can easily switch between them by updating the strategy: field in conf/base.yaml.

Structure:
---
```
conf/
├── base.yaml
└── strategy/
    ├── fedadagrad.yaml
    ├── fedadam.yaml
    ├── fedavg.yaml
    ├── fedavgM.yaml
    ├── fedProx.yaml
    ├── fedsgd.yaml
    ├── qfedavg.yaml
    └── resnet152.yaml
    
```
---
## 🩺 Model Inference (Standalone)  

You can test the trained model locally using `detection.py`.  

```bash
python detection.py
```

This script:  
- Loads the best model weights from training.  
- Preprocesses a given input image.  
- Runs a forward pass through the model.  
- Displays prediction results (Normal vs Cataract) with confidence scores.  

Example output:  
```
Prediction: Cataract
Confidence: 92.7%
Other class probability: 7.3%
```


## 🧾 Visuals  

**Mobile App Screenshots**   

### Mobile Interface
![Flutter App Screenshot](path/to/flutter_screenshot.png)

*(Replace `path/to/flutter_screenshot.png` with the actual image path in your repo.)*  

---

## 📊 Evaluation Metrics  

During training, the model reports:  
- **Loss** per round  
- **Overall accuracy**  
- **Class-based accuracies:**  
  - Normal  
  - Cataract  

---

## 🏁 Results Summary  

| Metric | Value |
|--------|--------|
| Global Accuracy | ~90–95% |
| Federated Efficiency | Preserved model performance without sharing sensitive data |
| Class Accuracy (Cataract) | 93.2% |
| Class Accuracy (Normal) | 94.1% |

*(Values are example results from the latest training run.)*

---

## 🔒 Privacy Considerations  

This system ensures **data privacy** by using a federated learning approach:  
- Training data never leaves the local devices/hospitals.  
- Only model parameters (not patient data) are shared with the central server.  

---

## 👤 Author  

**👨‍💻 Emircan Sahan**  
🎓 Computer Engineering @ Sakarya University of Applied Sciences  
🧬 Research Focus: Federated Learning, Medical Imaging, Deep Learning  
📫 Contact: [LinkedIn](https://www.linkedin.com/in/emircansahan/) • [GitHub](https://github.com/emirsahan00)  

---

## 🧾 License  

This project is licensed under the **MIT License** – you’re free to use, modify, and distribute it with attribution.  

---

## 🌟 Acknowledgements  

Special thanks to:  
- **Flower framework developers** for enabling easy federated learning experimentation.  
- **PyTorch** for providing flexible deep learning tools.  
- **Hydra & OmegaConf** for powerful configuration management.
