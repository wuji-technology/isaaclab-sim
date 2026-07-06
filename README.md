# isaaclab-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  [![Release](https://img.shields.io/github/v/release/wuji-technology/isaaclab-sim)](https://github.com/wuji-technology/isaaclab-sim/releases)

Simulation demo for IsaacSim. This repository provides a minimal example for loading and controlling the Wuji Hand in IsaacSim simulator. Loads pre-built USD assets with PBR materials and plays trajectory in a loop, supporting both left- and right-hand configurations via `--side` argument.

**Get started with [Quick Start](#quick-start). For detailed documentation, please refer to [Isaac Lab Simulation Example](https://docs.wuji.tech/docs/en/wuji-description/latest/related-repos/#42-isaac-lab-simulation-example) on Wuji Docs Center.**

https://github.com/user-attachments/assets/2f58ad84-7ed6-46fe-94c1-b4148068bec3

## Repository Structure

```text
├── assets/                        // Demo videos and screenshots
├── data/
│   └── wave.npy                   // Pre-recorded trajectory data
├── wuji_hand_description/         // Submodule: URDF, MJCF, USD, meshes
├── run_sim.py                     // Main simulation script
└── README.md
```

## Quick Start

### Installation

```bash
git clone --recurse-submodules https://github.com/wuji-technology/isaaclab-sim.git
cd isaaclab-sim
```

Follow the [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up your IsaacSim environment.

### Running

```bash
# Right hand (default)
python run_sim.py

# Left hand
python run_sim.py --side left
```

The script loads the pre-built USD model from the submodule and plays the trajectory in a loop.

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
