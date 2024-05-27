using RandomMeas
using Test

filenames = ["test_simple.jl",
  "test_examples.jl"
]

@testset "RandomMeas.jl" begin
    @testset "$filename" for filename in filenames
        if startswith(filename, "test_") && endswith(filename, ".jl")
            println("Running $filename")
            @time include(filename)
        end
    end
end
