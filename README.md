# Minimal VABD

This repository contains the minimal implementation of Vibrational Affine-body Dynamics for the squishy-ball example. It includes weight generation, SQEM target generation, neural medial training, medial DCD/CCD simulation.

The commands below assume Windows PowerShell and are run from the repository root.

## Prerequisites

- 64-bit Windows and Python 3.10. The native modules use the CPython 3.10 ABI.
- An NVIDIA CUDA GPU. The bundled extension projects currently target compute capability 8.6.
- A CUDA 12 toolkit and a compatible NVIDIA driver.
- Visual Studio 2022 with the Desktop development with C++ workload.
- CMake, vcpkg, Eigen3, and pybind11.
- NVIDIA cuDSS for the `mtk` and optional `dxslv` builds.
- MATLAB with its Python engine is recommended for weight eigenanalysis. The weight tools can instead use SciPy with `--backend scipy`.

The repository expects the custom native source packages in this layout:

```text
ext/
  mtk/      # medial geometry and CCD; required
  sqem/     # SQEM fitting; required only to regenerate neural targets
  dxslv/    # cuDSS reduced solver; optional, with a SciPy fallback
```

Their CMake projects require Eigen3 and pybind11 from vcpkg, and `mtk`/`dxslv` additionally require the CUDA toolkit and cuDSS. Update the toolchain, CUDA, and cuDSS paths in each local `setup.py` before building on a machine with different installation locations.

The release also contains Windows CPython 3.10 builds of `mtk` and `sqem` at the repository root. They can be used directly when their CUDA runtime dependencies are available. Building from `ext/` is necessary for other Python or platform ABIs.

## Installation

Create and activate a Python 3.10 environment, then install the public runtime dependencies:

```powershell
conda create -n nwarp python=3.10 -y
conda activate nwarp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Weight generation

Generate the squishy model's scalar weights and modal-warping basis in order:

```powershell
python -m weight_gen.fast_cd --weights 20 --backend auto
python -m weight_gen.modal_warping --backend auto
python -m weight_gen.basis_span --kind translation-rotation --weights 20
```

The commands read `assets/squishy/squishy.tobj` and write their cached eigenanalysis and final weights below `data/`. Use `--backend matlab` to require MATLAB or `--backend scipy` to run eigenanalysis entirely with SciPy (not recommanded). Add `--viewer` to inspect a result interactively.

## Neural medial training

The checked-in runtime checkpoint is already exported, so training is not required to run the simulator. To regenerate it, first create SQEM targets for the included 2,000 samples and then train the network:

```powershell
python -m g2m.squishy_data
python -m g2m.train_squishy --epochs 6000 --width 64
```

Target generation resumes from its partial checkpoint. Training uses a fixed 1,500/500 split, checks physical prediction and analytical derivative errors, and refuses approval when its quality gate fails. A successful run writes:

```text
data/squishy_neural_medial.pth       # training checkpoint
data/squishy_neural_medial.json      # qualification metrics
data/squishy_neural_medial_warp.npz  # Torch-free simulator weights
```

PyTorch is used only by target training/export. Simulation evaluates the NPZ network and its analytical center Jacobian in batched Warp kernels.

## Running

Open the default 10-body neural-medial DCD simulation:

```powershell
python medial_vabd.py
```

Run headlessly for a fixed number of frames:

```powershell
python medial_vabd.py --frames 200 --quiet
```

Available simulation configurations are:

```powershell
# Neural medial DCD with split L-BFGS (default)
python medial_vabd.py

# Linear medial DCD instead of neural medial DCD
python medial_vabd.py --linear-medial

# Medial CCD
python medial_vabd.py --ccd-only-vabd

# Exact Newton instead of split L-BFGS on the DCD path
python medial_vabd.py --no-split-lbfgs
```

Use `--bodies N` to change the number of pyramid bodies (max 361). Pure CCD modes save restart checkpoints every 25 frames by default; override this with `--checkpoint-interval N` and resume a saved state with `--from-frame N`.
