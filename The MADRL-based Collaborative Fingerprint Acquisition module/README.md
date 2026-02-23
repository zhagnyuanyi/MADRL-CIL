## **MADRL-based Collaborative Fingerprint Acquisition**

This module implements the Multi-Agent Deep Reinforcement Learning (MADRL) framework for collaborative WiFi fingerprint acquisition. It is designed to coordinate multiple agents to efficiently map indoor signal environments.

### **Core Components**

* **Environment Setup:** Configuration files and logic for the collaborative fingerprinting simulation/real-world environment.
* **QMIX Implementation:** A state-of-the-art MARL algorithm utilizing **Centralized Training with Decentralized Execution (CTDE)**.
* **Neural Network Architectures:**
* **Agent Network:** Individual agent policies designed for local observation processing.
* **Mixing Network:** A monotonic value function factorizer that integrates individual agent utilities into a global Q-value.



### **Usage**

To train or evaluate the collaborative acquisition module, ensure your `wheeltec` conda environment is active and run:

```bash
conda activate wheeltec
python QMIX.py

```