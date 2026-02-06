# SecureHealth-ML
## A Privacy-Preserving Federated Learning System with Differential Privacy

[English Version](#english)

---

## English

### Overview
This project demonstrates how to train a Machine Learning model on sensitive healthcare data without ever moving the data out of its original location (e.g., a hospital or a user's phone). By combining **Federated Learning** and **Differential Privacy**, we ensure that individual records cannot be identified from the final model.

### Key Features
* **Decentralized Training**: Uses the Flower framework to train models across multiple local clients
* **Data Privacy**: Implements Differential Privacy (using Opacus) to add mathematical noise to gradients, preventing data leakage
* **Secure Communication**: Only model weights are shared with the central server, never the raw data
* **Healthcare Focus**: Example implementation for medical diagnosis prediction

### Tech Stack
* **Language**: Python 3.9+
* **ML Framework**: PyTorch
* **Federated Learning**: Flower (flwr)
* **Privacy Tool**: Opacus (for Differential Privacy)

### Project Structure
```
securehealth-ml/
├── README.md
├── requirements.txt
├── server.py              # Federated learning server
├── client.py              # Federated learning client
├── model.py               # Neural network model definition
├── data_utils.py          # Data loading and preprocessing
├── config.py              # Configuration settings
└── simulation.py          # Run complete simulation
```

### How to Run

#### 1. Clone the repository
```bash
git clone https://github.com/petersawm/secure-health-ml.git
cd securehealth-ml
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Simulation (Recommended for testing)
```bash
python simulation.py
```

This will simulate 3 clients training on synthetic healthcare data.

#### 4. Manual Setup (Advanced)

**Start the Server:**
```bash
python server.py
```

**Start Clients** (in separate terminals):
```bash
# Terminal 2
python client.py --client-id 0

# Terminal 3
python client.py --client-id 1

# Terminal 4
python client.py --client-id 2
```

### Configuration

Edit `config.py` to adjust:
- Number of federated rounds
- Differential privacy epsilon (ε)
- Model architecture
- Learning rate and batch size

### Privacy Guarantees

This system provides **ε-differential privacy** where:
- **ε (epsilon)**: Privacy budget (lower = more private)
- Default ε = 1.0 provides strong privacy
- ε < 1.0 provides very strong privacy (but may reduce accuracy)

### Example Use Case

Training a disease diagnosis model across multiple hospitals without sharing patient data:
1. Each hospital keeps its patient data locally
2. Each hospital trains the model on its own data
3. Only encrypted model updates are sent to central server
4. Server aggregates updates and sends improved model back
5. No patient data ever leaves the hospital

---

### License
MIT License

### Contributing
Pull requests are welcome! For major changes, please open an issue first.
