# isaaclab-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/wuji-technology/isaaclab-sim)](https://github.com/wuji-technology/isaaclab-sim/releases)

Simulation demo (IsaacSim): minimal demo for loading and controlling the Wuji Hand in IsaacSim simulator. Loads pre-built USD assets with PBR materials and plays trajectory in a loop. Supports both left- and right-hand configurations via `--side` argument.

https://github.com/user-attachments/assets/2f58ad84-7ed6-46fe-94c1-b4148068bec3

## Table of Contents

- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Running](#running)
- [Contact](#contact)

## Repository Structure

```text
├── assets/                        // Demo videos and screenshots
├── data/
│   └── wave.npy                   // Pre-recorded trajectory data
├── wuji_hand_description/         // Submodule: URDF, MJCF, USD, meshes
├── run_sim.py                     // Main simulation script
└── README.md
```

## Usage

### Prerequisites

```bash
git clone --recurse-submodules https://github.com/wuji-technology/isaaclab-sim.git
```

Follow the [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up your environment.

### Running

```bash
# Right hand (default)
python run_sim.py

# Left hand
python run_sim.py --side left
```

The script loads the pre-built USD model from the submodule and plays the trajectory in a loop.

## Contact

For any questions, please contact support@wuji.tech.
