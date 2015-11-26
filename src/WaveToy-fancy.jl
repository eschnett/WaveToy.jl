module WaveToy

import Base: .+, .-, .*, ./
export .+, .-, .*, ./

for op in [:.+, :.-, :.*, :./]
    @eval begin
        function $op{D,T1,T2}(x::NTuple{D,T1}, y::NTuple{D,T2})
            ntuple(d->$op(x[d], y[d]), D)
        end
        $op{D,T}(x::NTuple{D,T}, y::T) = ntuple(d->$op(x[d], y), D)
        $op{D,T}(x::T, y::NTuple{D,T}) = ntuple(d->$op(x, y[d]), D)
    end
end

export Parameters
module Parameters
    const dim = 3
    const np = (5, 5, 5)
    const xmin = (0.0, 0.0, 0.0)
    const xmax = (1.0, 1.0, 1.0)
    const dx = (xmax .- xmin) ./ (np .- 1)

    const A = 1.0
end

export init
function init()
    p = Parameters
    u = Array(Float64, p.np)
    kx = 2*pi ./ (p.xmax .- p.xmin)
    for k in 1:p.np[3], j in 1:p.np[2], i in 1:p.np[1]
        x = p.xmin .+ ((i,j,k) .- 1) .* p.dx
        u[i,j,k] = p.A * prod(ntuple(d -> sin(x[d]*kx[d]), p.dim))
    end
    u
end

end
