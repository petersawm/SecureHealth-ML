# SecureHealth-ML Quick Start Guide

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/petersawm/secure-health-ml.git
cd securehealth-ml
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Quick Simulation (Recommended for Testing)

Run everything in a single process:

```bash
python simulation.py
```

This will:
- Create 3 simulated clients with synthetic healthcare data
- Train a federated model for 10 rounds
- Display accuracy and loss metrics
- Generate a visualization plot

### Option 2: Manual Client-Server Setup

#### Terminal 1 - Start Server
```bash
python server.py
```

#### Terminal 2 - Start Client 0
```bash
python client.py --client-id 0
```

#### Terminal 3 - Start Client 1
```bash
python client.py --client-id 1
```

#### Terminal 4 - Start Client 2
```bash
python client.py --client-id 2
```

**Note**: Start all clients within a reasonable time frame after starting the server.

## Configuration

Edit `config.py` to customize:

```python
# Number of training rounds
NUM_ROUNDS = 10

# Number of clients
NUM_CLIENTS = 3

# Enable/disable differential privacy
USE_DIFFERENTIAL_PRIVACY = True

# Privacy budget (lower = more private)
TARGET_EPSILON = 1.0

# Training parameters
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001
```

## Understanding the Output

### Simulation Output

```
Round 1/10
  Client 0 - Loss: 0.6234, Accuracy: 65.23%, ε: 0.82
  Client 1 - Loss: 0.6123, Accuracy: 67.11%, ε: 0.84
  Client 2 - Loss: 0.6345, Accuracy: 64.89%, ε: 0.81
  
Round 1 Evaluation Results:
  Average Loss: 0.6234
  Average Accuracy: 65.74%
```

- **Loss**: Lower is better (model is learning)
- **Accuracy**: Higher is better (model predictions are more accurate)
- **ε (epsilon)**: Privacy budget spent (lower means more privacy)

### Privacy Guarantees

The system provides **ε-differential privacy**:

- **ε = 1.0**: Strong privacy (default)
- **ε < 1.0**: Very strong privacy (may reduce accuracy)
- **ε > 1.0**: Weaker privacy (better accuracy)

**Trade-off**: More privacy (lower ε) = slightly lower accuracy

## Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you installed all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "Connection refused" (Client-Server mode)
**Solution**: 
1. Make sure the server is running first
2. Check that `SERVER_ADDRESS` in `config.py` is correct
3. Ensure no firewall is blocking the connection

### Issue: "Out of memory"
**Solution**: Reduce batch size or model size in `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
HIDDEN_SIZE = 32  # Reduce from 64
```

### Issue: Opacus not working
**Solution**: The system will automatically fall back to regular training. If you need differential privacy, ensure Opacus is properly installed:
```bash
pip install opacus==1.4.0
```

## What's Happening Under the Hood?

1. **Data Generation**: Each client generates synthetic healthcare data (simulating different hospitals)

2. **Local Training**: Each client trains the model on its own data WITHOUT sharing it

3. **Model Update**: Clients send only the model weights (not data) to the server

4. **Aggregation**: Server combines updates using Federated Averaging (FedAvg)

5. **Privacy Protection**: Differential Privacy adds noise to prevent data leakage

6. **Iteration**: This repeats for multiple rounds, improving the global model

## Real-World Application

This system can be adapted for:

- **Multi-hospital collaboration**: Train diagnostic models without sharing patient data
- **Mobile health apps**: Aggregate insights from user devices
- **Research consortiums**: Collaborative research while maintaining privacy
- **Clinical trials**: Analyze distributed trial data securely

## Next Steps

1. **Experiment with privacy settings**: Try different epsilon values
2. **Modify the model**: Edit `model.py` to use different architectures
3. **Use real data**: Replace synthetic data generation in `data_utils.py`
4. **Add more clients**: Increase `NUM_CLIENTS` in `config.py`
5. **Deploy on real network**: Change `SERVER_ADDRESS` to actual IP address

## Support

For issues or questions:
- Check the main README.md
- Open an issue on GitHub
- Review the code comments in each file

## Privacy Best Practices

1. **Never** disable differential privacy in production with sensitive data
2. **Monitor** epsilon values - they accumulate over rounds
3. **Validate** model performance doesn't leak private information
4. **Use** secure communication (HTTPS/TLS) in production
5. **Audit** privacy budget regularly

---

Happy Federated Learning! 🚀🔒
