# k-TES Tissue-Ensemble Simulation

This directory contains the reproducibility scripts for the k-space tissue-ensemble simulation experiments reported in the manuscript.

The experiments use the local `kTES.jl` package located at the root of this repository. The package does not need to be registered in the Julia General Registry.

## 1. Clone the repository

Clone the complete repository, not only this directory:

```bash
git clone https://github.com/gsahonero/kTES.jl.git
cd kTES.jl
```

The expected structure is:

```text
<REPOSITORY>/
├── Project.toml
├── src/
├── ....
└── experiments/
    └── k-space tissue ensemble simulation/
        ├── Project.toml
        ├── setup_and_run.jl
        ├── README.md
        ├── .cpu_benchmark/
        ├── data/
        ├── results/
        └── scripts/
```

## 2. Enter this directory

### Windows PowerShell

```powershell
cd "experiments\k-space tissue ensemble simulation"
```

### Linux or macOS

```bash
cd "experiments/k-space tissue ensemble simulation"
```

## 3. Run the launcher

```bash
julia setup_and_run.jl
```

The launcher will:

1. activate the experiments environment;
2. link the local `kTES.jl` package;
3. instantiate the required dependencies;
4. check GPU availability;
5. display the available experiments.

Select the desired experiment from the menu.

## Important: local package path

The launcher contains:

```julia
Pkg.develop(path=REPOSITORY_ROOT)
```

This links the experiment environment to the local `kTES.jl` source directory.

Therefore, run the launcher from a complete clone of the repository. Do not copy this folder alone.

## Data requirements

Some experiments require data files that are included in the repository. Because of their size, they are divided but `setup_and_run.jl` will combine the divided compressed files.

### Experiment 1

Requires:

```text
data/phantoms.jld2
```

The file is distributed in split parts. The launcher reconstructs the file automatically after confirmation.

### Experiment 5

Requires:

```text
data/subject_1.mat
```

The dataset is available from Dryad:

- DOI: <https://doi.org/10.5061/dryad.brv15dvpj>
- Dataset page: <https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvpj>

The file is distributed in split parts. The launcher reconstructs the file automatically after confirmation.

## Output files

Results are written to:

```text
results/
```

The CPU scalability benchmark produces:

```text
results/benchmark_results_cpu_1.bin
results/benchmark_results_cpu_auto.bin
```

The GPU scalability benchmark produces:
```text
results/benchmark_results.bin
```

The scalability plots require the CPU benchmark results and the GPU benchmark results to be available first.

All experiments, except for 4A and 4B, produce an output image in PDF format.

## CPU benchmark

Experiment 4A runs:

- a one-thread CPU benchmark;
- a maximum-thread CPU benchmark.

When prompted, answer:

```text
Run the one-thread benchmark? [Y/n]: Y
Run the maximum-thread benchmark? [Y/n]: Y
```

The maximum-thread run uses the number of CPU threads reported by Julia.

## GPU benchmark

The GPU benchmark is available only when CUDA detects a functional CUDA-compatible GPU.

If no compatible GPU is detected, Experiment 4B is marked unavailable and the remaining CPU experiments can still be run.