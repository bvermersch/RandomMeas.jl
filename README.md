# RandomMeas: The randomized measurements toolbox in julia

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bvermersch.github.io/RandomMeas.jl/dev/)
[![Build Status](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src="Pics/RandomMeas.png" alt="drawing" width="500"/>. 

## Install
In a Julia terminal, install the package RandomMeas
```julia
julia> ]

pkg> add RandomMeas
```

RandomMeas relies heavily on ITensors, and for some examples, uses  PastaQ

## Documentation
[dev](https://bvermersch.github.io/RandomMeas.jl/dev/) -- documentation of the in-development version.


## Presentation

1) Routines to prepare on QPUs/simulate the data acquisition

	```julia
	using ITensors
	using RandomMeas
	
	N  = 10
	χ = 2^(N÷2)
	nu = 2000
	NM = 100
	ξ = siteinds("Qubit", N)
	ψ = randomMPS(ξ; linkdims=χ);
	data = zeros(Int8,(nu,NM,N))
	datat = zeros(Int8,(NM,N))
	u = Vector{Vector{ITensor}}()
	for r in 1:nu
	    push!(u,get_rotations(ξ,1)) #Haar rotations in A
	    data[r,:,:] = get_RandomMeas(ψ,u[r],NM)
	end
	``` 

2) Postprocessing routines for randomized measurements, eg to get the purity
	
	```julia
	purity = get_purity_hamming(data,ξ)
	``` 
	
2) Interface with matrix-product-states simulations with ITensors.jl & PastaQ.jl to simulate large-scale randomized measurements protocols.
	
3) Jupyter notebooks to present various recent case studies.

## Examples with jupyter notebooks

1)  Energy/Energy variance measurements with classical shadows
	+ Related Paper [Huang et al, Nat Phys 2020](https://doi.org/10.1038/s41567-020-0932-7)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/EnergyVarianceMeasurements.ipynb)

2) Fidelities from common randomized measurements
	+ Related paper [Vermersch, al, PRXQ 2024](https://doi.org/10.1103/PRXQuantum.5.010352)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/FidelityCommonRandomizedMeasurements.ipynb)

3) Cross-Platform verication
	+ Related paper [Elben et al, PRL 2019 ](https://doi.org/10.1103/PhysRevLett.124.010504)[Zhu et al, Nat. Comm 2022](https://www.nature.com/articles/s41467-022-34279-5)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CrossPlatform.ipynb)

4) Entanglement Entropy of pure states & The page curve"
	+ Related paper [Brydges et al, Science 2019](https://doi.org/10.1126/science.aau4963)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/PureStateEntanglement.ipynb)

5) Mixed-state entanglement: "The $p_3$-PPT condition" and batch shadows
	+ Related papers [Elben et al, PRL 2020](https://link.aps.org/doi/10.1103/PhysRevLett.125.200501)   [Rath et al, PRXQ 2023](https://doi.org/10.1103/PRXQuantum.4.010318)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/MixedStateEntanglement.ipynb)
	
6) Analyzing the experimental data of Brydges et al, Science 2019
	+ Related paper [Brydges et al, Science 2019](https://doi.org/10.1126/science.aau4963)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/examples/BrydgesScience2019.ipynb)
	
7) Robust Shadow tomography
	+ Related papers
	[Chen et al PRX Q 2021](https://doi.org/10.1103/PRXQuantum.2.030348)
	[Vitale et al, arxiv: arXiv:2307.16882]
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/RobustShadowTomography.ipynb)
	
8) Process Shadow tomography
 	+ Related paper [Kunjummen et al, Phys. Rev. A 107, 042403 (2023)](
 	https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042403)
 	[ Levy,et al Phys. Rev. Research 2024](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013029)
 	+ Related paper [Vermersch, al, PRXQ 2024](https://doi.org/10.1103/PRXQuantum.5.010352)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ProcessShadowTomography.ipynb)

9) Virtual distillation
	+ Related paper [Seif et al, PRX Quantum 4, 010303 2023](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010303)
	+ [Jupyter notebook](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/VirtualDistillation.ipynb)