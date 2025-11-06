# WujiHand IsaacLab Simulation

A minimal demo for loading and controlling the WujiHand model in the IsaacLab simulator.

<video src="./assets/video.mp4" controls=""></video>

## Requirements

* Follow the official documentation to set up your environment: 
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Quick Start

Run the simulation with trajectory playback:
```bash
python run_sim.py
```

The script loads the default left hand model and plays the trajectory from `data/wave.npy` in a loop. To use the left hand, edit `side = "left"` in `run_sim.py`.