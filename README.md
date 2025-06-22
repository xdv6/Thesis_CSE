# Cube Stacking & Reward Machines in Robosuite

This repository contains experiments and infrastructure for using Reward Machines with Robosuite environments, with a focus on cube stacking and sequence-based lifting tasks.

---

## 🔀 Branch Overview

- **`main`**:  
  Contains the original files and setup for the **cube stacking problem**.

- **`new-env`**:  
  Contains the updated environment and scripts for the **cube sequence lifting task**.

---

## 📁 Repository Structure

- `reward_machines/`:  
  Adapted version of [Rodrigo Toro Icarte's Reward Machines repo](https://github.com/RodrigoToroIcarte/reward_machines).  
  - The RM environment for Robosuite is located at:  
    `reward_machines/reward_machines/envs/robosuite_rm/`

- `dockerfile`:  
  Docker setup file with all required dependencies for running Reward Machines with Robosuite.

- `setup.sh`:  
  Shell script for use on **UGent's GPU lab (GPULab)** to activate and start the Docker container environment.

- `robosuite_test_script.py`:  
  Python script containing test code for **reward structures** and **transition logic** for the Robosuite + RM integration.

---

## 🛠️ Getting Started

Clone the repository and check out the appropriate branch depending on your task:

```bash
# Clone the repo
git clone <repo-url>
cd <repo-name>

# Checkout cube sequence lifting branch
git checkout new-env
