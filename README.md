# Towards Pervasive WLAN Localization Leveraging Collaborative Mobile Sites: A Multi-Agent Deep Reinforcement Learning Approach

Official implementation of a **collaborative indoor WLAN localization** framework:

- **MADRL (QMIX, CTDE)** coordinates multiple mobile sites/robots to efficiently collect WiFi fingerprints.
- **MS-FM (Multi-Site Fingerprint Matching)** performs high-precision localization using the collaboratively collected fingerprints.

---

## 📂 Repository Structure

```
.
├── Data/
│   ├── 0rece_rss.sql
│   └── Processed sample data/
├── MS-FM model/
│   ├── my_model.py
│   ├── Data_Load.py
│   ├── Classification_Single_Side.py
│   └── data_enhancement.py
│   └──  ...
├── The MADRL-based Collaborative Fingerprint Acquisition module/
│   ├── QMIX.py
│   └── ...
└── Package sending and receiving platform/
	├── receive_tf.py
	├── issue_position.py
	└── ...

````

---

## 🧩 Module Overview

### 1) MS-FM Model (`/MS-FM model`)

This module handles fingerprint feature extraction and localization inference.

- `my_model.py`: MS-FM architecture.
- `Data_Load.py`: data pipeline that converts RSS sequences into model-ready tensors.
- `data_enhancement.py`: augmentation utilities to improve robustness under noise/environment changes.

### 2) MADRL Collaborative Acquisition (`/The MADRL-based Collaborative Fingerprint Acquisition module`)

This module learns multi-agent coordination policies for efficient fingerprint acquisition.

- **Algorithm**: QMIX for CTDE.
- **Core idea**: each agent learns a local policy; a mixing network factorizes a global action-value.
- **Objective**: reduce acquisition time and improve localization accuracy.

### 3) Data Management (`/Data`)

- `data.sql`: raw RSS observations stored in SQL format.
- `Processed sample data/`: processed datasets for MS-FM training and MADRL validation.

### 4) Hardware & ROS Integration (`/Package sending and receiving platform`)

Scripts for online packet capture, coordinate tracking, and robot navigation/control.

- Monitor-mode setup for WiFi sniffing.
- ROS utilities: `receive_tf.py`, `issue_position.py` for pose/coordinate streaming and control.

---

## 🗄️ Data Preparation

### Step 1 — Initialize Database

Import `Data/data.sql` into your SQL engine (e.g., MySQL / SQLite depending on your dump format).

### Step 2 — Process Raw RSS

Run the preprocessing script to generate cleaned/structured samples (adjust script name/path as in your repo):

```bash
python data_processing.py
```

### Step 3 — Build Training Tensors for MS-FM

Use `MS-FM model/Data_Load.py` to convert processed samples into dataloaders/tensors:

* Output: model-ready datasets used by `Classification_Single_Side.py` (or your training entry file).

---

## 🚀 Training & Evaluation

### 1) Train MS-FM

Example entry:

```bash
python Classification_Single_Side.py
```


   ### 2) Train MADRL (QMIX)

   ```bash
   python QMIX.py
   ```

   **Expected behavior**

   * launches multi-agent training (CTDE)
   * periodically saves QMIX agent weights / mixer weights
   * logs episodic rewards, coverage metrics, etc.

### For a more detailed description, please refer to the corresponding readme file in the relevant working directory.



