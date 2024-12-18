using RandomMeas
using MAT
ITensors.disable_warn_order()

# Parameters
NU, NM = 500, 150

# times
times = [1,3,5] # quench times in ms

# Directory to save exported files
export_dir = "/Users/aelben/Data/RandomMeas_SampleBrydges/MeasurementDataExports/"  # Directory to save exported files
isdir(export_dir) || mkpath(export_dir) # Ensure the export directory exists

# Samples
#samples = 5 # Number of times, we simulate the Brydges experiment with the same parameters

@showprogress  for t in times
    # Load and process the quantum state
    qstate = matread("../examples/BrydgesScience2019data/rho_10_XY_10_-1.00_0.00"*string(t)*"_1_1_1_flr_1.mat")["rho"]
    qstate = reshape(qstate, tuple((2 * ones(Int, 2 * N))...))
    rho = ITensor(qstate, vcat(ξ', ξ))
    rho = MPO(rho, ξ, cutoff=1e-10)

    for sample in 101:102
        # Generate measurement settings and simulate results
        measurement_settings = RandomMeas.LocalUnitaryMeasurementSettings(N, NU, site_indices=ξ)
        measurement_data = simulate_local_measurements(rho, NM; mode="dense", measurement_settings=measurement_settings)

        # Construct a meaningful filename
        filename = joinpath(export_dir, "simulated_measurement_data_t_$(t)_sample_$(sample)_NU_$(NU)_NM_$(NM).npz")

        # Export the data
        export_measurement_data(measurement_data, filename)
        println("Exported: $filename")

    end
end
