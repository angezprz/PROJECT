## 📌 Project Overview
This project applies deep learning to recognize human activities from smartphone and smartwatch sensor data using the WISDM dataset.
Activities such as walking, jogging, sitting, standing, and climbing stairs are classified based on accelerometer and gyroscope signals, enabling accurate detection of motion patterns in real-world scenarios.
The workflow includes data cleaning, exploratory data analysis (EDA), feature extraction through time-windowing, and model development using recurrent neural networks.
By combining rigorous preprocessing with advanced deep learning techniques, this project demonstrates the potential of Human Activity Recognition (HAR) for applications in health monitoring, fitness tracking, and context-aware computing.
## 📂 Repository Structure
```bash
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned dataset
├── notebooks/            # Jupyter notebooks for EDA & modeling
├── src/                  # Python scripts for preprocessing & models
├── README.md             # Project documentation
```

## ⚙️ Installation
```bash
git clone https://github.com/angezprz/HAR-Activity-Recognition.git
cd HAR-Activity-Recognition
pip install -r requirements.txt

```

## 📊 Exploratory Data Analysis (EDA)

- Dataset cleaned and standardized (removed duplicates, missing values, mapped activity codes to readable labels).

- Distribution of activities visualized.

- Sensor-device combinations analyzed (phone/watch with accel/gyro).

- Sample accelerometer signal plots for walking and standing.

## 🛠 Preprocessing

- Windowing: 3-second windows with 1.5-second step (90% purity threshold).

- Features used: x, y, z, and magnitude.

- Standardization: per-window normalization for stable training.

- Handling imbalance: class weights applied during training.

## 🤖 Model Development

- Architecture:

- Bidirectional LSTM (64 units) + LSTM (32 units)

- BatchNormalization, Dropout (0.4 and 0.3)

- Dense(64, ReLU) + Softmax output layer

- Optimizer: Adam (learning rate = 1e-4)

- Loss Function: Categorical Crossentropy

- Evaluation Metrics: Accuracy, Precision, Recall, Macro F1

## ✅ Results

- Best model performance on test set:

- Accuracy: ~85%

- Precision: ~85%

- Recall: ~85%

- Macro F1: ~85%

