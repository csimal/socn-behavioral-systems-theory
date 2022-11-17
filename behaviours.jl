using ControlSystemsBase
using LinearAlgebra

include("utils.jl")


function ss2BT(A,B,C,D,Π, T)
    n = size(A,1)
    m = size(B,2)
    p = size(C,1)
    q = m + p
    Q = zeros(q*T, m*T+n)
    for i in 1:(m*T)
        Q[i,i] = 1.0
    end

end

"""
    ss2BT_datadriven(sys, T)

Compute an orthonormal basis of ℬ_T by sampling q*T trajectories.
"""
function ss2BT_datadriven(sys,T)
    _,m,p = sizes(sys)
    q = m + p
    W = zeros(q*T, q*T)
    for i in 1:q*T
        w = random_trajectory(sys,T)
        W[:,i] .= vec(w)
    end
    return extract_basis(W)
end


"""
    ss2BT_hankel(sys, T, L)

Compute an orthonormal basis of ℬ_L using the Hankel matrix of a random trajectory.
"""
function ss2BT_hankel(sys,T,L)
    w = random_trajectory(sys,T)
    ℋ = hankel_matrix(w,L)
    return extract_basis(ℋ)
end