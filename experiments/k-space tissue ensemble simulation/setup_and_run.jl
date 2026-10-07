using Pkg

const PAPER_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(PAPER_DIR, "..", ".."))
const SCRIPTS_DIR = joinpath(PAPER_DIR, "scripts")
const DATA_DIR = joinpath(PAPER_DIR, "data")
const RESULTS_DIR = joinpath(PAPER_DIR, "results")
const GPU_VERIFIER = joinpath(SCRIPTS_DIR, "verify_gpu.jl")

const DATASET_DOI = "https://doi.org/10.5061/dryad.brv15dvpj"

const DATASET_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvpj"

const SUBJECT_1_FILENAME = "subject_1.mat"
const SUBJECT_1_PATH = joinpath(DATA_DIR, SUBJECT_1_FILENAME)
const SUBJECT_1_MIN_SIZE = 500 * 1024^2
const SUBJECT_1_PART_COUNT = 12
const SUBJECT_1_PART_PREFIX = "subject_1.mat."
const SUBJECT_1_PART_DIGITS = 3

const PHANTOMS_FILENAME = "phantoms.jld2"
const PHANTOMS_PATH = joinpath(DATA_DIR, PHANTOMS_FILENAME)
const PHANTOMS_MIN_SIZE = 1 * 1024^2
const PHANTOMS_PART_COUNT = 10
const PHANTOMS_PART_PREFIX = "phantoms.jld2."
const PHANTOMS_PART_DIGITS = 3

const CPU_BENCHMARK_FILENAME = "exp_4_CPU.jl"
const CPU_BENCHMARK_OUTPUT = "benchmark_results_cpu_auto.bin"
const CPU_BENCHMARK_ONE_OUTPUT = "benchmark_results_cpu_1.bin"
const CPU_BENCHMARK_AUTO_OUTPUT = "benchmark_results_cpu_auto.bin"

function setup_environment()
    println("\nActivating paper environment:")
    println("  ", PAPER_DIR)

    Pkg.activate(PAPER_DIR)

    println("\nLinking local kTES.jl package:")
    println("  ", REPOSITORY_ROOT)

    Pkg.develop(path=REPOSITORY_ROOT)

    println("\nInstantiating dependencies...")
    Pkg.instantiate()

    println("\nEnvironment ready.")
end

function verify_gpu()
    if !isfile(GPU_VERIFIER)
        println("\nGPU status: unavailable.")
        println("GPU verifier not found:")
        println("  ", GPU_VERIFIER)
        return false
    end

    println("\nVerifying CUDA GPU availability...")

    command = `$(Base.julia_cmd()) --project=$(PAPER_DIR) $GPU_VERIFIER`

    try
        output = read(command, String)
        print(output)

        if occursin("GPU_AVAILABLE", output)
            return true
        end

        println("GPU status: unavailable.")
        return false

    catch error
        println("\nGPU status: unavailable.")
        println("CUDA verification failed:")
        showerror(stdout, error)
        println()
        return false
    end
end

function split_file_paths(prefix, count, digits)
    return [
        joinpath(
            DATA_DIR,
            prefix * lpad(string(index), digits, '0'),
        )
        for index in 1:count
    ]
end

function subject_1_part_paths()
    return split_file_paths(
        SUBJECT_1_PART_PREFIX,
        SUBJECT_1_PART_COUNT,
        SUBJECT_1_PART_DIGITS,
    )
end

function phantoms_part_paths()
    return split_file_paths(
        PHANTOMS_PART_PREFIX,
        PHANTOMS_PART_COUNT,
        PHANTOMS_PART_DIGITS,
    )
end

function file_is_valid(path, minimum_size)
    return isfile(path) && filesize(path) >= minimum_size
end

function report_file(path)
    file_size = filesize(path)

    println("\nFound valid file:")
    println("  ", path)
    println(
        "  Size: ",
        round(file_size / 1024^2; digits=2),
        " MB",
    )
end

function concatenate_parts(
    output_path,
    part_paths,
    minimum_size,
    label,
)
    missing_parts = filter(path -> !isfile(path), part_paths)

    if !isempty(missing_parts)
        println("\nThe following ", label, " parts are missing:")

        for path in missing_parts
            println("  ", basename(path))
        end

        println("\nExpected files:")

        for path in part_paths
            println("  ", path)
        end

        return false
    end

    temporary_path = output_path * ".part"

    println("\nReconstructing ", basename(output_path), " from:")
    println("  ", length(part_paths), " parts")

    try
        open(temporary_path, "w") do output
            for (index, part_path) in enumerate(part_paths)
                println(
                    "  Joining part ",
                    index,
                    "/",
                    length(part_paths),
                    ": ",
                    basename(part_path),
                )

                open(part_path, "r") do input
                    write(output, input)
                end
            end
        end

        reconstructed_size = filesize(temporary_path)

        if reconstructed_size < minimum_size
            error(
                "The reconstructed file is unexpectedly small: " *
                "$(round(reconstructed_size / 1024^2; digits=2)) MB",
            )
        end

        mv(temporary_path, output_path; force=true)

        println("\nReconstruction completed.")
        report_file(output_path)

        return true

    catch error
        isfile(temporary_path) && rm(temporary_path; force=true)

        println("\nReconstruction failed:")
        showerror(stdout, error)
        println()

        return false
    end
end

function prepare_subject_1()
    mkpath(DATA_DIR)

    if file_is_valid(SUBJECT_1_PATH, SUBJECT_1_MIN_SIZE)
        report_file(SUBJECT_1_PATH)
        return true
    end

    println("\nExperiment 5 requires ", SUBJECT_1_FILENAME, ".")
    println("Dataset DOI: ", DATASET_DOI)
    println("Dataset page: ", DATASET_URL)

    println("\nThe launcher expects these parts in:")
    println("  ", DATA_DIR)

    println("\nRequired parts:")

    for path in subject_1_part_paths()
        println("  ", basename(path))
    end

    print(
        "\nHave all 12 subject_1.mat parts been placed in the data " *
        "directory? [y/N]: ",
    )

    answer = lowercase(strip(readline()))

    if answer != "y"
        println("\nExperiment 5 was not started.")
        return false
    end

    return concatenate_parts(
        SUBJECT_1_PATH,
        subject_1_part_paths(),
        SUBJECT_1_MIN_SIZE,
        "subject_1.mat",
    )
end

function prepare_phantoms()
    mkpath(DATA_DIR)

    if file_is_valid(PHANTOMS_PATH, PHANTOMS_MIN_SIZE)
        report_file(PHANTOMS_PATH)
        return true
    end

    println("\nExperiment 1 requires ", PHANTOMS_FILENAME, ".")

    println("\nThe launcher expects these parts in:")
    println("  ", DATA_DIR)

    println("\nRequired parts:")

    for path in phantoms_part_paths()
        println("  ", basename(path))
    end

    print(
        "\nHave all 10 phantoms.jld2 parts been placed in the data " *
        "directory? [y/N]: ",
    )

    answer = lowercase(strip(readline()))

    if answer != "y"
        println("\nExperiment 1 was not started.")
        return false
    end

    return concatenate_parts(
        PHANTOMS_PATH,
        phantoms_part_paths(),
        PHANTOMS_MIN_SIZE,
        "phantoms.jld2",
    )
end

function available_experiments(gpu_available)
    gpu_description = gpu_available ?
        "Experiment 4B: GPU scalability benchmark" :
        "Experiment 4B: GPU scalability benchmark [GPU unavailable]"

    return [
        (
            "1",
            "Experiment 1: Tissue-basis signal approximation",
            "exp_1.jl",
            true,
        ),

        (
            "2",
            "Experiment 2: Microscopic inhomogeneity and T2*",
            "exp_2.jl",
            true,
        ),

        (
            "3",
            "Experiment 3: k-TES validation",
            "exp_3.jl",
            true,
        ),

        (
            "4",
            "Experiment 4A: CPU scalability benchmark",
            "exp_4_CPU.jl",
            true,
        ),

        (
            "5",
            gpu_description,
            "exp_4_GPU.jl",
            gpu_available,
        ),

        (
            "6",
            "Experiment 4C: Scalability plots " *
            "(Experiments 4A and 4B must run first)",
            "exp_4_plots.jl",
            true,
        ),

        (
            "7",
            "Experiment 5: Heterogeneous phantom clustering",
            "exp_5.jl",
            true,
        ),
    ]
end

function choose_experiment(experiments)
    println("\nAvailable experiments:")

    for (id, description, _, available) in experiments
        status = available ? "" : " -- unavailable"
        println("  [", id, "] ", description, status)
    end

    println("  [q] Quit")

    while true
        print("\nSelect an experiment: ")
        choice = lowercase(strip(readline()))

        choice == "q" && return nothing

        for experiment in experiments
            id, _, _, available = experiment

            if choice == id
                if !available
                    println(
                        "\nThis experiment requires a functional " *
                        "CUDA-compatible GPU.",
                    )
                    continue
                end

                return experiment
            end
        end

        println("Invalid selection. Please enter one of the listed numbers.")
    end
end

function run_cpu_benchmark(threads, output_filename)
    script_path = joinpath(SCRIPTS_DIR, CPU_BENCHMARK_FILENAME)

    if !isfile(script_path)
        error("CPU benchmark script not found: $script_path")
    end

    mkpath(RESULTS_DIR)

    temporary_root = joinpath(PAPER_DIR, ".cpu_benchmark")
    mkpath(temporary_root)

    temporary_dir = mktempdir(temporary_root)
    child_results_dir = joinpath(temporary_dir, "results")

    mkpath(child_results_dir)

    try
        println("\nRunning CPU benchmark with ", threads, " thread(s).")
        println("Script: ", script_path)
        println("Temporary working directory: ", temporary_dir)

        command = `$(Base.julia_cmd())
            --project=$(PAPER_DIR)
            --threads=$(threads)
            $script_path`

        run(setenv(command, dir=temporary_dir))

        generated_output = joinpath(
            child_results_dir,
            CPU_BENCHMARK_OUTPUT,
        )

        if !isfile(generated_output)
            error(
                "The CPU benchmark completed, but the expected output " *
                "was not found at:\n  $(generated_output)",
            )
        end

        final_output = joinpath(RESULTS_DIR, output_filename)
        cp(generated_output, final_output; force=true)

        println("\nSaved benchmark result:")
        println("  ", final_output)

    finally
        println("\nTemporary benchmark files were preserved at:")
        println("  ", temporary_dir)

        println(
            "They can be deleted manually after OneDrive finishes " *
            "synchronizing.",
        )
    end
end

function run_cpu_benchmarks()
    max_threads = Sys.CPU_THREADS

    if max_threads < 1
        error("Julia reported an invalid number of CPU threads.")
    end

    println("\nCPU benchmark configuration:")
    println("  One-thread run: 1 thread")
    println("  Maximum-thread run: ", max_threads, " threads")

    print("\nRun the one-thread benchmark? [Y/n]: ")
    run_one = lowercase(strip(readline())) != "n"

    print("Run the maximum-thread benchmark? [Y/n]: ")
    run_max = lowercase(strip(readline())) != "n"

    if !run_one && !run_max
        println("\nNo CPU benchmark was selected.")
        return
    end

    if run_one
        run_cpu_benchmark(
            1,
            CPU_BENCHMARK_ONE_OUTPUT,
        )
    end

    if run_max
        run_cpu_benchmark(
            max_threads,
            CPU_BENCHMARK_AUTO_OUTPUT,
        )
    end

    println("\nCPU benchmark execution completed.")
end

function run_standard_experiment(experiment)
    _, description, filename, _ = experiment
    script_path = joinpath(SCRIPTS_DIR, filename)

    if !isfile(script_path)
        error("Experiment script not found: $script_path")
    end

    if filename == "exp_1.jl"
        prepare_phantoms() || return
    elseif filename == "exp_5.jl"
        prepare_subject_1() || return
    end

    println("\nRunning ", description)
    println("Script: ", script_path, "\n")

    command = `$(Base.julia_cmd()) --project=$(PAPER_DIR) $script_path`

    run(command)
end

function run_experiment(experiment)
    _, _, filename, _ = experiment

    if filename == CPU_BENCHMARK_FILENAME
        run_cpu_benchmarks()
    else
        run_standard_experiment(experiment)
    end
end

function main()
    setup_environment()

    gpu_available = verify_gpu()
    experiments = available_experiments(gpu_available)

    experiment = choose_experiment(experiments)

    if experiment === nothing
        println("\nNo experiment selected.")
        return
    end

    run_experiment(experiment)

    println("\nExperiment completed.")
end

main()