using RandomMeas
using Documenter

DocMeta.setdocmeta!(RandomMeas, :DocTestSetup, :(using RandomMeas); recursive=true)

makedocs(;
    modules=[RandomMeas],
    authors="Benoit Vermersch & Andreas Elben",
    sitename="RandomMeas.jl",
    format=Documenter.HTML(;
        canonical="https://bvermersch.github.io/RandomMeas.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bvermersch/RandomMeas.jl",
    devbranch="main",
)
