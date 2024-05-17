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


function get_purity_shadows(data::Array{Int8},u::Vector{Vector{ITensor}},ξ::Vector{Index{Int64}};G::Union{Vector{Float64},Nothing}=nothing)
    nu,NM,NA = size(data)
    shadow = ITensor(vcat(ξ,ξ'))
    shadow2 = ITensor(vcat(ξ,ξ'))
    for r in 1:nu
        P = get_Born_data_binary(data[r,:,:],ξ)
        shadow_temp = get_shadow(P,ξ,u[r];G=G)
        shadow += shadow_temp
        shadow2 += power(shadow_temp,2)
    end
    return real(trace(power(shadow,2),ξ)-trace(shadow2,ξ))/(nu*(nu-1))
end

function get_purity_hamming(data::Array{Int8},ξ::Vector{Index{Int64}})
    nu,NM,NA = size(data)
    p2 = 0.
    for r in 1:nu
        p2 += get_X_data(data[r,:,:],ξ)/nu
    end
    return p2
end


function get_X_data(data::Array{Int8},sites::Vector{Index{Int64}})
	NM = size(data,1)
  NA = size(data,2)
	#println("get_Born 1")
	prob = get_Born_data_binary(data,sites)
	Hamming_matrix = zeros(Float64,(NA,2,2))	
	for i in 1:NA
			Hamming_matrix[i,1,1] = 1
			Hamming_matrix[i,2,2] = 1
			Hamming_matrix[i,2,1] = -0.5
			Hamming_matrix[i,1,2] = -0.5
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

function get_Born_data_binary(data::Array{Int8},sitesA::Vector{Index{Int64}})
	NM = size(data,1)
	NA = size(data,2)
	probf = StatsBase.countmap(eachrow(data))
	prob = zeros(Int64,(2*ones(Int,NA))...)
	for (state,val) in probf
		prob[state...] = val
	end
	probT = ITensor(prob,sitesA)/NM
	return probT
end
