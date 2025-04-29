# Federated Sequential Recommendation System

This repository implements a Federated Learning-based Sequential Recommendation System, combining privacy-preserving federated learning techniques with sequence-aware recommendation models.

## Overview

The system predicts a user's next likely item based on historical interactions while keeping user data decentralized by training models locally on client devices. Only model updates, not raw user data, are shared with the central server for aggregation, ensuring data privacy throughout the training process. Multiple clients train models locally, and their updates are aggregated using the Federated Averaging (FedAvg) algorithm.

---

## Project Structure

- **`data/`** - Contains raw datasets.
- **`saved_data/`** - Stores processed data and intermediate results.
- **`src/`** - Core implementation of federated learning and recommendation models.
- **`test/`** - Scripts for model evaluation.
- **`servertest.ipynb`** - Interactive notebook for testing server-side components.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/allenbijo/federated-sequential-rec-sys.git
cd federated-sequential-rec-sys
```

### 2. Set Up the Virtual Environment and Install Dependencies

Ensure Python 3.8 or later is installed, then create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use 'venv\Scripts\activate'
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Data

- Place datasets in the `data/` directory.
- Preprocess the data using relevant scripts from the `src/` directory.

### 4. Configure Training Parameters

- Modify settings in `fedavg_runner.py`, such as the number of clients, training epochs, and model hyperparameters. Refer to the project's documentation or inline comments in the code for detailed configuration examples.

### 5. Start Federated Training

```bash
python fedavg_runner.py
```

This command will initiate federated training, where client models train locally and updates are aggregated on the server.

## Key Features

- **Privacy-Preserving Training:** Keeps user data decentralized.
- **Sequential Recommendations:** Predicts next items based on user history.
- **Model Customization:** Easily adjustable parameters for research and experimentation.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

