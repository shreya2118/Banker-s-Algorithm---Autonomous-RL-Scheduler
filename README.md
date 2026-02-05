# 🏦 Banker's Algorithm - Autonomous RL Scheduler

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI](https://img.shields.io/badge/AI-Reinforcement%20Learning-green)
![OS](https://img.shields.io/badge/OS-Deadlock%20Avoidance-orange)

### **Can an AI solve Deadlocks better than a predefined algorithm?**
This project explores the intersection of **Operating Systems** and **Reinforcement Learning (RL)**. It replaces the traditional static Banker's Algorithm with an autonomous agent capable of learning "safe sequences" to avoid system deadlocks.

It features a **Dual-Agent Architecture** that compares two different AI approaches in real-time:
1.  **Global Agent (Offline):** A "Super Brain" pre-trained on 50,000+ random scenarios. It relies on generalized intuition.
2.  **JIT Agent (Online):** A "Specialist" that trains from scratch *instantly* on the specific problem presented.

---

## 🚀 Key Features
* **Interactive GUI:** Built with `Tkinter`, allowing users to dynamically modify Allocation, Max Need, and Available Resources.
* **🤖 Auto-Pilot Mode:** Watch the AI solve the deadlock step-by-step in the GUI.
* **📊 Benchmark Arena:** Runs a head-to-head competition between the Offline and Online agents, comparing:
    * **Success Rate:** Can it find a safe path?
    * **Execution Time:** How fast is the decision?
    * **Step Efficiency:** Did it take the shortest path?
* **Anti-Farming Logic:** Custom RL Environment (`bankers_env.py`) prevents the agent from exploiting easy rewards, forcing it to solve the actual deadlock.

---

## 🧠 How It Works (The "Brain")
The core logic utilizes **Q-Learning (Tabular)** to make decisions.

### 1. The Environment (`bankers_env.py`)
* **State:** A tuple representing which processes are finished `(False, True, False...)`.
* **Action:** Choosing which Process ID (P0, P1...) to execute next.
* **Reward System:**
    * `+100`: Successfully finding a Safe Sequence (All processes finished).
    * `-100`: Choosing a process that is already finished.
    * `-50`: Choosing a process that leads to an unsafe state (Deadlock).
    * `+10`: Valid, safe move.

### 2. The Comparison (Global vs. JIT)
* **Global Agent:** Trained on Google Colab for 50,000 episodes. It creates a robust policy file (`global_brain.pkl`). It is instant but can fail on unique "edge cases" it hasn't seen before.
* **Just-In-Time (JIT) Agent:** Uses the "Overfitting" concept to its advantage. It trains for 2,000 episodes in ~0.5 seconds on the *exact* user input. It is slightly slower to start but mathematically guarantees a solution if one exists.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* NumPy

```bash
pip install numpy
