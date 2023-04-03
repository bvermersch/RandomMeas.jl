function get_h_tensor()
	Hamming_matrix = zeros(Float64,(2,2))	
	Hamming_matrix[1,1] = 1
	Hamming_matrix[2,2] = 1
	Hamming_matrix[2,1] = -0.5
	Hamming_matrix[1,2] = -0.5
	a = Index(2,"a")
	b = Index(2,"b")
	Hamming_tensor = itensor(Hamming_matrix,a,b)
	return Hamming_tensor,a,b
end

function get_overlap(prob1::ITensor,prob2::ITensor,sites::Vector{Index{Int64}},NA::Int64)
	Hamming_tensor,a,b = get_h_tensor()
	h = Hamming_tensor*delta(a,sites[1])*delta(b,prime(sites[1]))
	global overlap_temp = prob1*h
	for i in 2:NA
		h = Hamming_tensor*delta(a,sites[i])*delta(b,prime(sites[i]))
		global overlap_temp = overlap_temp*h
	end
	overlap_temp = overlap_temp*prime(prob2)
	#println("contraction done")
	overlap_temp = real(scalar(overlap_temp))
	X = overlap_temp*2^NA
	return X
end

function get_X_data(data::Array{Int64},sites::Vector{Index{Int64}},NA::Int64,Pz::Vector{Float64}=[-1.])
	NM = size(data,1)
	#println("get_Born 1")
	prob = get_Born_data(data,sites)


	Hamming_matrix = zeros(Float64,(NA,2,2))	
	for i in 1:NA
		if Pz[1]<0 # NO CORRECTION
			Hamming_matrix[i,1,1] = 1
			Hamming_matrix[i,2,2] = 1
			Hamming_matrix[i,2,1] = -0.5
			Hamming_matrix[i,1,2] = -0.5
		else
			Hamming_matrix[i,1,1] = (Pz[i]+3)/(4*Pz[i])
			Hamming_matrix[i,2,2] = (Pz[i]+3)/(4*Pz[i])
			Hamming_matrix[i,1,2] = (Pz[i]-3)/(4*Pz[i])
			Hamming_matrix[i,2,1] = (Pz[i]-3)/(4*Pz[i])
		end
	end

	a = Index(2,"a")
	b = Index(2,"b")
	Hamming_tensor = itensor(Hamming_matrix[1,:,:],a,b)
	h = Hamming_tensor*delta(a,sites[1])*delta(b,prime(sites[1]))
	global purity_temp = prob*h
	for i in 2:NA
		Hamming_tensor = itensor(Hamming_matrix[i,:,:],a,b)
		h = Hamming_tensor*delta(a,sites[i])*delta(b,prime(sites[i]))
		global purity_temp = purity_temp*h
	end
	purity_temp = purity_temp*prime(prob)
	#println("contraction done")
	purity_temp = real(scalar(purity_temp))
	X = purity_temp*2^NA
	X = X*NM^2/(NM*(NM-1)) - 2^NA/(NM-1)
	return X
end

function get_Born_data(data::Array{Int64},sitesA::Vector{Index{Int64}})
	NM = size(data,1)
	NA = size(sitesA,1)
	nh = 2^NA
	prob = StatsBase.counts(data,nh)
	prob = reshape(prob,tuple((2*ones(Int,NA))...))
	prob = ITensor(prob,sitesA)/NM
	return prob
end
