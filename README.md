# Coarse‑Grained Polymer Simulation

This repository contains code and notebooks for simulating the static and kinetic behavior of coarse‑grained polymer chains using random walk and Metropolis Monte Carlo methods.

## Table of Contents

- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Running Tests](#running-tests)  
- [Dependencies](#dependencies)  
- [License](#license)  

---

## Project Structure
pcba_project/ ← project root
│
├── .github/ ← CI/CD workflows
│ └── workflows/
│ └── ci.yml
│
├── notebooks/ ← analysis & visualization notebooks
│ ├── development.ipynb
│ ├── analysis_plots.ipynb
│ └── installation_test.ipynb
│
├── src/ ← source code
│ └── pcba_project/ ← Python package
│ ├── init.py
│ ├── random_walk.py
│ ├── monte_carlo.py
│ ├── dynamics.py
│ ├── model.py
│ ├── stress_tensor.py
│ └── utils_plot.py
│
├── tests/ ← unit tests
│ ├── test_random_walk.py
│ ├── test_monte_carlo.py
│ ├── test_model.py
│ └── test_stress_tensor.py
│
├── .gitignore ← ignored files (caches, checkpoints, etc.)
├── LICENSE ← project license (MIT)
├── README.md ← this file
├── requirements.txt ← project dependencies
└── setup.py ← installation script

---

## Installation

**Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/pcba_project.git
   cd pcba_project
**Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt
# or install in editable mode:
pip install -e .

# run the main simulation script
python -m pcba_project.random_walk
jupyter notebook notebooks/development.ipynb


