# isaaclab-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/wuji-technology/isaaclab-sim)](https://github.com/wuji-technology/isaaclab-sim/releases)

Simulation demo (IsaacSim): minimal demo for loading and controlling the Wuji Hand in IsaacSim simulator. The script loads the default right hand model and plays the trajectory in a loop. Supports both left and right hand configurations.

https://github.com/user-attachments/assets/3fffb009-f78a-4dda-93ed-94de20b93811

## Table of Contents

- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Running](#running)
- [Contact](#contact)

## Repository Structure

```text
├── assets/
├── data/
│   └── wave.npy
├── wuji_hand_description/
├── run_sim.py
├── wuji_hand.py
└── README.md
```

### Directory Description

| Directory / File | Description |
|------------------|-------------|
| `assets/` | Asset files for simulation |
| `data/` | Trajectory data files including `wave.npy` |
| `wuji_hand_description/` | Git submodule containing Wuji Hand model descriptions (URDF, MJCF, meshes) |
| `run_sim.py` | Main simulation script |
| `wuji_hand.py` | Wuji Hand model implementation |

## Usage

### Prerequisites

```bash
git clone --recurse-submodules https://github.com/wuji-technology/isaaclab-sim.git
```

Follow the [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up your environment.

### Running

```bash
python run_sim.py
```

The script loads the default right hand model and plays the trajectory from `data/wave.npy` in a loop.

To use the left hand, edit `HAND_SIDE = "left"` in `run_sim.py`.

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
