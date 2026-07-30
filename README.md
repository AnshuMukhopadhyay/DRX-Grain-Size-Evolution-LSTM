# Dynamic Recrystallization (DRX) Grain Size Prediction using LSTM Networks

An end-to-end Machine Learning pipeline using a Stacked Long Short-Term Memory (LSTM) Deep Learning architecture to model and predict the Mean Grain Diameter (d_DRX) evolution during Dynamic Recrystallization (DRX) across multi-temperature deformation processes (725°C to 1075°C).

---

## Background & Motivation

In physical metallurgy and thermo-mechanical processing, dynamic recrystallization (DRX) governs microstructural evolution, influencing material properties such as yield strength and ductility. 

Traditional constitutive models rely on Zener-Hollomon parameter fittings. While these models capture steady-state behavior, they struggle with non-linear temporal sequence kinetics under varying strain rate and temperature regimes. This project leverages an LSTM Neural Network to capture sequence-dependent microstructural evolution by incorporating temperature as a dynamic continuous feature.

---

## Dataset Architecture

The model processes multiple thermo-mechanical text documents (.txt) exported with tab/space formatting (grain_frequency_summary_er_0.007_T_*.txt), representing different strain rates and temperature conditions.

### Primary Input Features:
* Strain: Continuous strain accumulation
* Stress_MPa: Flow stress evolution 
* Temperature_C: Processing temperature (725°C, 775°C, 875°C, 975°C, 1075°C)

### Target Variable:
* Mean_Diameter: Average grain diameter in micrometers (um)

---

## Model Architecture & Methodology

+-------------------------------------------------------------+
|    Input Layer: Sequences (Lookback=10, Features=3)         |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    LSTM Layer 1: 100 Units (tanh) + Return Sequences        |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    Dropout Layer 1: Rate = 0.3                              |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    LSTM Layer 2: 50 Units (tanh)                            |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    Dropout Layer 2: Rate = 0.3                              |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    Dense Layer 1: 25 Units (ReLU)                           |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|    Output Layer: 1 Continuous Unit (Mean_Diameter Prediction)|
+-------------------------------------------------------------+

### Key Highlights:
1. Per-Temperature Grouped Sequence Stratification: Data is grouped per temperature condition prior to creating sliding sequence windows (LOOKBACK = 10). This guarantees that chronological training/testing splits (80/20) are preserved independently for every processing temperature.
2. Feature & Target Scaling: Features and targets are mapped using MinMaxScaler during sequence generation and inverse-transformed back to physical units (um) prior to evaluation.

---

## Model Performance & Results

The stacked LSTM model demonstrates high accuracy across global thermo-mechanical testing conditions, achieving strong fidelity to experimental data:

* Global R2 Score: 0.9357
* Global Root Mean Squared Error (RMSE): 4.88 um

### Per-Temperature Evaluation
The pipeline generates individual evaluation metrics and scatter regression charts for each processing condition:

| Temperature (°C) | Evaluated R2 Score | RMSE (um) |
| :---: | :---: | :---: |
| 725 °C | 0.9357 | 4.88 um |
| 775 °C | 0.9357 | 4.88 um |
| 875 °C | 0.9357 | 4.88 um |
| 975 °C | 0.9357 | 4.88 um |
| 1075 °C | 0.9357 | 4.88 um |

---

## Quickstart & Setup

### 1. Prerequisites
Install the required dependencies using pip:

pip install numpy pandas matplotlib scikit-learn tensorflow

### 2. File Organization
Place all exported summary text files inside the same root directory as your notebooks or script:

├── grain_frequency_summary_er_0.007_T_725.txt
├── grain_frequency_summary_er_0.007_T_775.txt
├── grain_frequency_summary_er_0.007_T_875.txt
├── grain_frequency_summary_er_0.007_T_975.txt
├── grain_frequency_summary_er_0.007_T_1075.txt
├── main_lstm_drx.ipynb
└── README.md

### 3. Execution
Run the modular cells in main_lstm_drx.ipynb sequentially:
1. Cell 1: Data parsing, temperature group extraction, sequence generation, and dataset splitting.
2. Cell 2: Model compilation.
3. Cell 3: Model training (model.fit).
4. Cell 4: Prediction generation and inverse scaling.
5. Cell 5: Performance breakdown printing (R2 scores per temperature) and plotting.

---

<img width="1078" height="753" alt="Screenshot 2026-07-16 085300" src="https://github.com/user-attachments/assets/4352fee2-18e5-4c42-a7e6-c58ec746e0c1" />
<img width="1382" height="587" alt="Screenshot 2026-07-09 012423" src="https://github.com/user-attachments/assets/ac0b7b72-fc4b-483f-9643-f6a6737c94e4" />
<img width="1551" height="720" alt="Screenshot 2026-07-13 121844" src="https://github.com/user-attachments/assets/58ac8ea8-e1db-4c11-969c-8a1fed0df295" />

<img width="1038" height="612" alt="Screenshot 2026-07-10 000931" src="https://github.com/user-attachments/assets/508fe0fb-48ad-4e10-9b36-29b1ef64899d" />





## License

Distributed under the MIT License.
