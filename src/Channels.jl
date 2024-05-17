function Apply_depo_channel!(ρ::MPO,ξ::Vector{Index{Int64}},p::Vector{Float64})
    N = length(ρ)
    for i in 1:N
        s = ξ[i]
        X = ρ[i]*δ(s,s')
        ρ[i]=(1-p[i])*ρ[i]+p[i]/2*X*δ(s,s')
    end
end
