## 📁 Repository Structure

The repository is organized into the following key modules:

* **`atk-surrogate/models/`**: Contains surrogate models used to mimic target GNN behaviors for black-box or grey-box adversarial attacks.
* **`attack/`**: Implements various adversarial attack methodologies tailored for graph data and GNN architectures.
* **`bayesopt/`**: Includes Bayesian Optimization scripts for efficient hyperparameter search and optimization strategies.
* **`data/` & `datasets/`**: Directories for storing, preprocessing, and loading graph datasets.
* **`evaluate/`**: Scripts and metrics for evaluating model performance, robustness against attacks, and explanation fidelity.
* **`explainer/`**: Implements GNN explanation methods to interpret node, edge, and graph-level predictions.
* **`gnn/`**: Contains the core Graph Neural Network model architectures.
* **`train/`**: Core training loops and routines for the GNN models and explainers.
* **`utils/`**: Helper functions, data processing utilities, and configuration tools.

## 🚀 Features

* **GNN Robustness & Attacks**: Evaluate and test graph models against adversarial perturbations.
* **Surrogate Modeling**: Train surrogate architectures to approximate complex GNN boundaries.
* **Explainable AI (XAI) for Graphs**: Gain insights into model decisions via dedicated graph explainer modules.
* **Bayesian Optimization**: Efficiently tune complex models and attack parameters.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Esperanto-mega/V-InfoR.git](https://github.com/Esperanto-mega/V-InfoR.git)
   cd V-InfoR
   ```

2. **Set up the environment:**
   It is recommended to use a virtual environment (e.g., Anaconda or `venv`).
   ```bash
   conda create -n vinfor_env python=3.8
   conda activate vinfor_env
   ```

3. **Install dependencies:**
   *(Note: Ensure you have PyTorch and PyTorch Geometric installed according to your CUDA version)*
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Data Preparation
Place your raw datasets in the `datasets/` directory or use the provided data loaders to automatically download standard graph benchmarks.

### 2. Training the GNN
Navigate to the `train/` directory to train your base GNN models:
```bash
python train/train_gnn.py --dataset cora --epochs 200
```

### 3. Running Attacks
To execute adversarial attacks against the trained models using surrogate networks:
```bash
python attack/run_attack.py --target_model gcn --surrogate true
```

### 4. Bayesian Optimization
To run hyperparameter tuning or optimize attack vectors using Bayesian Optimization:
```bash
python bayesopt/optimize.py --config configs/bo_config.yaml
```

### 5. Explaining Predictions
Generate explanations for your trained GNN model's predictions:
```bash
python explainer/explain.py --model_path path/to/saved/model.pth --node_idx 42
```

## 📊 Evaluation
Use the scripts inside the `evaluate/` folder to generate quantitative metrics on model accuracy, attack success rate, and explanation quality.
```bash
python evaluate/eval_metrics.py --results_dir outputs/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, feature requests, or enhancements.

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
