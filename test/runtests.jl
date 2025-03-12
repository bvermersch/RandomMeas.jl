using RandomMeas
using Test

filenames = filter(x->occursin("test_",x), readdir())

@show filenames

@testset "RandomMeas.jl" begin
    @testset "$filename" for filename in filenames
        if startswith(filename, "test_") && endswith(filename, ".jl")
            println("Running $filename")
            @time include(filename)
        end
    end
end
