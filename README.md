# WujiHand IsaacSim Simulation

Minimal demo for WujiHand in IsaacSim simulator.


https://github.com/user-attachments/assets/3fffb009-f78a-4dda-93ed-94de20b93811

<video src="./assets/video.mp4" controls=""></video>

## Setup
```bash
git clone --recurse-submodules https://github.com/Wuji-Technology-Co-Ltd/isaaclab-sim.git
```
* Follow the official documentation to set up your environment: 
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Quick Start

Run the simulation with trajectory playback:
```bash
python run_sim.py
```

The script loads the default right hand model and plays the trajectory from `data/wave.npy` in a loop. To use the left hand, edit `HAND_SIDE = "left"` in `run_sim.py`.
