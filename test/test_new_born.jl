# Import necessary packages
using ITensors
using StatsBase
using BenchmarkTools

ITensors.disable_warn_order()

# ------------------------------
# Old Implementation
# ------------------------------

# Original get_h_tensor function
function get_h_tensor_old()
    Hamming_matrix = zeros(Float64, (2, 2))
    Hamming_matrix[1, 1] = 1
    Hamming_matrix[2, 2] = 1
    Hamming_matrix[2, 1] = -0.5
    Hamming_matrix[1, 2] = -0.5
    a = Index(2, "a")
    b = Index(2, "b")
    Hamming_tensor = ITensor(Hamming_matrix, a, b)
    return Hamming_tensor, a, b
end

# Original get_purity_direct_single_meas_setting function
function get_purity_direct_single_meas_setting_old(data::Array{Int})
    # Extract dimensions
    NM, N = size(data)

    # Generate site indices for qubits
    ξ = siteinds("Qubit", N)

    # Compute the Born probability tensor
    prob = get_Born(data, ξ)

    # Initialize Hamming tensor and purity accumulator
    Hamming_tensor, a, b = get_h_tensor_old()
    purity_temp = prob * Hamming_tensor * δ(a, ξ[1]) * δ(b, prime(ξ[1]))

    # Iterate over all sites to accumulate the purity tensor
    for i in 2:N
        purity_temp *= Hamming_tensor * δ(a, ξ[i]) * δ(b, prime(ξ[i]))
    end

    # Finalize purity computation
    purity_temp *= prime(prob)
    purity = real(scalar(purity_temp)) * 2^N
    purity = purity * NM^2 / (NM * (NM - 1)) - 2^N / (NM - 1)

    return purity
end

# ------------------------------
# New Implementation
# ------------------------------

# Modified get_h_tensor function accepting indices
function get_h_tensor_new(s::Index, s_prime::Index)
    Hamming_tensor = ITensor(Float64,s, s_prime)
    Hamming_tensor[s => 1, s_prime => 1] = 1
    Hamming_tensor[s => 2, s_prime => 2] = 1
    Hamming_tensor[s => 1, s_prime => 2] = -0.5
    Hamming_tensor[s => 2, s_prime => 1] = -0.5
    return Hamming_tensor
end

# Streamlined get_purity_direct_single_meas_setting function

function get_purity_direct_single_meas_setting_new(data::Array{Int})
    NM, N = size(data)
    ξ = siteinds("Qubit", N)
    prob = get_Born(data, ξ)
    purity_temp = prob
    for i in 1:N
        purity_temp *= get_h_tensor_new(ξ[i], ξ[i]')
    end
    purity_temp *= prob'
    purity = real(scalar(purity_temp)) * 2.0^N
    purity = purity * NM^2 / (NM * (NM - 1)) - 2.0^N / (NM - 1)
    return purity
end

function get_purity_direct_single_meas_setting_newest(data::Array{Int})
    NM, N = size(data)
    ξ = siteinds("Qubit", N)
    prob = get_Born(data, ξ)


    # Apply Hamming tensor to each leg sequentially in a compact way
    purity_temp = foldl((p_temp, i) -> p_temp * get_h_tensor_new(ξ[i], prime(ξ[i])),1:N;init = prob)

    # purity_temp = prob
    # for i in 1:N
    #     H_i = get_h_tensor_new(ξ[i], ξ[i]')
    #     purity_temp = purity_temp * H_i
    # end
    purity_temp *= prob'
    purity = real(scalar(purity_temp)) * 2.0^N
    purity = purity * NM^2 / (NM * (NM - 1)) - 2.0^N / (NM - 1)
    return purity
end

# ------------------------------
# Common Helper Functions
# ------------------------------

# get_Born function (common to both implementations)
function get_Born(data::Array{Int}, ξ::Vector{Index{Int64}})
    # Get dimensions: NM is the number of measurements, N is the number of sites
    NM, N = size(data)
    # Count occurrences of each unique binary state in the dataset
    probf = StatsBase.countmap(eachrow(data))  # Dictionary: {state => count}

    # Initialize a dense tensor to store probabilities
    prob = zeros(Int64, (2 * ones(Int, N))...)

    # Populate the tensor with counts from the dictionary
    for (state, val) in probf
        prob[state...] = val
    end

    # Normalize the tensor by the total number of measurements
    probT = ITensor(prob, ξ) / NM
    return probT
end

# ------------------------------
# Benchmarking Script
# ------------------------------

# Function to generate random measurement data
function generate_data(NM::Int, N::Int)
    rand(1:2, NM, N)
end

# Function to benchmark both implementations
function benchmark_purity_functions(NM::Int, N::Int)
    println("Benchmarking with NM = $NM and N = $N")
    println("Generating data...")
    data = generate_data(NM, N)

    # Warm-up runs to ensure fair benchmarking
    println("Running warm-up computations...")
    purity_old = get_purity_direct_single_meas_setting_old(data)
    purity_new = get_purity_direct_single_meas_setting_new(data)
    purity_newest = get_purity_direct_single_meas_setting_newest(data)

    # Verify that both functions return the same result
    @assert isapprox(purity_old, purity_new; atol=1e-10) "Purity values do not match!"

    # Benchmark old function
    println("\nBenchmarking old function:")
    bench_old = @btime get_purity_direct_single_meas_setting_old($data)

    # Benchmark new function
    println("\nBenchmarking new function:")
    bench_new = @btime get_purity_direct_single_meas_setting_new($data)

    # Benchmark newest function
    println("\nBenchmarking newest function:")
    bench_newest = @btime get_purity_direct_single_meas_setting_newest($data)

    # Display purity values
    println("\nPurity (Old): $purity_old")
    println("Purity (New): $purity_new")
    println("Purity (Newest): $purity_newest")
    println("--------------------------------------------------\n")
end

# ------------------------------
# Run Benchmarking Tests
# ------------------------------

# Define parameters for benchmarking
NM_list = [100, 1000, 5000]  # Different numbers of measurements
N_list = [4, 6, 8, 16]           # Different numbers of qubits

# Run benchmarks for different NM and N values
for NM in NM_list
    for N in N_list
        benchmark_purity_functions(NM, N)
    end
end
