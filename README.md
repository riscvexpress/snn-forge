# snn-forge

**snn-forge** trains spiking neural networks (SNNs) with snnTorch, quantizes them, and converts them for the NPX accelerator of **RISC-V eXpress (RVX)**.

[https://riscvexpress.github.io](https://riscvexpress.github.io)

## Prerequisites

### Linux

`make`, `curl` (or `wget`), and `git`

### Windows

Install `git` and `make` with [Scoop](https://scoop.sh) in PowerShell.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
scoop install git make
```

## Get the source

```sh
git clone https://github.com/riscvexpress/snn-forge.git
```

## Install

Run the following in the top directory of this repository.

```sh
make setup_python     # copy ../rvx_config_python.mh, or install Python 3.9 into ./.rvx_venv
make check_python     # confirm the selected Python
make config           # generate ./rvx_setup.sh (Windows: register SNN_FORGE_HOME with setx)
make install          # install the Python packages (PyTorch, snnTorch, tonic, NIR, ...)
```

- `make setup_python_manual` lists the Python interpreters found on this machine in `./rvx_config_python.mh`; uncomment the one to use.
- On Linux, `make preinstall` installs the `sqlite` and `xz` development packages with `sudo` (yum or apt). It is only needed when the selected Python lacks the `sqlite3` or `lzma` module.

### PyTorch: CPU or CUDA

`make install` picks the PyTorch 2.8.0 build from the CUDA version that `nvidia-smi` reports:

| Driver CUDA version | Build |
|---|---|
| 12.9 or later | `cu129` |
| 12.8 | `cu128` |
| 12.6 | `cu126` |
| older, or no NVIDIA GPU/driver | `cpu` |

The GPU's compute capability can lower that choice, because the `cu128` and `cu129` builds no longer support Maxwell, Pascal, or Volta GPUs (e.g. V100, sm_70). With several GPUs, the oldest one decides.

| GPU compute capability | Build |
|---|---|
| below 7.5 (e.g. V100, P100) | `cu126` at most |
| below 5.0 | `cpu` |
| unknown (`nvidia-smi` cannot report it) | `cu126` at most |

The selected build is printed as `torch backend: ...`. To choose it yourself:

```sh
make install TORCH_BACKEND=cpu
```

## Setup

- Linux: run `source ./rvx_setup.sh` in every new terminal.
- Windows: open a new terminal, since `SNN_FORGE_HOME` is registered with `setx`.

## Usage

Each application is a `<name>.snn.jsonc` file (JSON with `//` and `/* */` comments) in `app`, for example `mnist_lenet5.snn.jsonc`:

```jsonc
{
  "preprocess": { "input": "mnist_opendataset", "timesteps": 4, "step_generation": "direct" },
  "train":      { "epoch": 10, "kfold": 5, "repeat": 1 },
  "global":     { "input_size": [28, 28], "input_channels": 1, "output_classes": 10, "neuron_type": "ws8" },
  "nodes": [    // the layers, in order
    { "name": "conv2d_1", "type": "Conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1 },
    { "name": "leaky_1",  "type": "Leaky",  "beta": 1.0, "reset_mechanism": "subtract" },
    // ...
    { "name": "shortcut_1", "type": "Shortcut", "skip_from": "leaky_1", "op": "Identity" }
  ]
}
```

- A node holds the same options as a section of the old `.cfg` format, plus a `name` and a `type`.
- A `Shortcut` names the node it skips from (`skip_from`) and its operation (`op`: `Identity`, `Conv2d`, `MaxPool2d`, `AvgPool2d`).
- The old `.cfg` format is no longer read. `python3 npx_cfg_convert.py <file>.cfg` converts an old file into `<file>.snn.jsonc`, comments included.
- `python3 npx_cfg_convert.py <file>.snn.jsonc` writes a [NIR](https://neuroir.org) graph (`.nir`, zero weights), and `python3 npx_cfg_convert.py <file>.nir` converts a NIR graph back into `<file>.snn.jsonc`. Only what snn-forge can build converts: Conv2d, Linear, Flatten, AvgPool2d/MaxPool2d and Leaky (LIF/IF) in a chain, plus residual adds. What NIR cannot express (`preprocess`, `train`, `global`, `neuron_type`, `reset_mechanism`, MaxPool2d) travels in NIR `metadata`, so a `.snn.jsonc` → `.nir` → `.snn.jsonc` round trip keeps the network.
- `npx_cfg_convert.py` picks the input format from the file name and writes the input's default target; `-t <format>` picks another one (for example `-t nir` straight from a `.cfg`), and `--list` shows the formats. Every format converts to and from one in-memory app, so a new format needs only a reader and/or a writer registered in `FORMAT_LIST`.

```sh
make app_list                # list the applications
make prepare_dataset         # download the open datasets into ./dataset
make mnist_lenet5            # train, quantize, test, and convert one application
make mnist_lenet5 GPU_ID=0   # same, on GPU 0 (default GPU_ID=-1: CPU)
```

Results are written to `result`.
