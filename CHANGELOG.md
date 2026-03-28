# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Replaced URDF runtime conversion with pre-built USD asset loading from submodule
- Removed standalone hand configuration module; inlined articulation config with PD gains and collision filters read from USD
- Added `--side` CLI argument for left/right hand selection
- Updated hand description submodule to v0.2.3 with USD assets, PBR materials, and collision filter pairs

## [0.1.0] - 2026-02-02

### Added

- Simulation script for loading and controlling Wuji Hand in IsaacSim
- Trajectory playback with loop support
- Support for both left- and right-hand configurations
- Wuji Hand model as git submodule

[Unreleased]: https://github.com/wuji-technology/isaaclab-sim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wuji-technology/isaaclab-sim/releases/tag/v0.1.0
