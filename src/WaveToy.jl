module WaveToy



# Parameters

const dim = 3

const ni = 4
const nj = 4
const nk = 4

const xmin = 0.0
const ymin = 0.0
const zmin = 0.0
const xmax = 1.0
const ymax = 1.0
const zmax = 1.0

const dx = (xmax - xmin) / (ni - 2)
const dy = (ymax - ymin) / (nj - 2)
const dz = (zmax - zmin) / (nk - 2)

x(i) = xmin + (i - 2) * dx
y(j) = ymin + (j - 2) * dy
z(k) = zmin + (k - 2) * dz

const tmin = 0.0

const dt = 1//4 * min(dx, dy, dz)
const maxL = max(xmax - xmin, ymax - ymin, zmax - zmin)
const niters = ceil(Int, maxL / dt)
const tmax = tmin + niters * dt

const A = 1.0
const kx = π / (xmax - xmin)
const ky = π / (ymax - ymin)
const kz = π / (zmax - zmin)
const ω = sqrt(kx^2 + ky^2 + kz^2)



function periodic!{T}(a::Array{T,dim})
    a[:, :, 1] = a[:, :, end-1]
    a[:, :, end] = a[:, :, 2]
    a[:, 1, 2:end-1] = a[:, end-1, 2:end-1]
    a[:, end, 2:end-1] = a[:, 2, 2:end-1]
    a[1, 2:end-1, 2:end-1] = a[end-1, 2:end-1, 2:end-1]
    a[end, 2:end-1, 2:end-1] = a[2, 2:end-1, 2:end-1]
    a
end



# Norms

export Norm
immutable Norm
    count::Float64
    sum::Float64
    sum2::Float64
    sumabs::Float64
    min::Float64
    max::Float64
    maxabs::Float64
    Norm() = new(0.0, 0.0, 0.0, 0.0, typemax(Float64), typemin(Float64), 0.0)
    function Norm(x0::Real)
        x = Float64(x0)
        new(1.0, x, x^2, abs(x), x, x, abs(x))
    end
    function Norm(x::Norm, y::Norm)
        new(x.count+y.count, x.sum+y.sum, x.sum2+y.sum2,
            x.sumabs+y.sumabs, min(x.min, y.min), max(x.max, y.max),
            max(x.maxabs, y.maxabs))
    end
end
import Base: count, sum, min, max
export count, sum, min, max, avg, sdv, norm1, norm2, norminf
count(n::Norm) = n.count
sum(n::Norm) = n.sum
min(n::Norm) = n.min
max(n::Norm) = n.max
avg(n::Norm) = n.sum / n.count
sdv(n::Norm) = sqrt(max(0.0, n.count * n.sum2 - n.sum^2)) / n.count
norm1(n::Norm) = n.sumabs / n.count
norm2(n::Norm) = sqrt(n.sum2 / n.count)
norminf(n::Norm) = n.maxabs
import Base: zero, +
export zero, +
zero(::Type{Norm}) = Norm()
+(x::Norm, y::Norm) = Norm(x, y)



# Cells

export Cell
immutable Cell
    u::Float64
    ρ::Float64
    vx::Float64
    vy::Float64
    vz::Float64
end

import Base: +, -, *, /, .+, .-, .*, ./
export +, -, *, /, .+, .-, .*, ./
+(c::Cell) = Cell(+c.u, +c.ρ, +c.vx, +c.vy, +c.vz)
-(c::Cell) = Cell(-c.u, -c.ρ, -c.vx, -c.vy, -c.vz)
+(c1::Cell, c2::Cell) =
    Cell(c1.u + c2.u, c1.ρ + c2.ρ, c1.vx + c2.vx, c1.vy + c2.vy, c1.vz + c2.vz)
-(c1::Cell, c2::Cell) =
    Cell(c1.u - c2.u, c1.ρ - c2.ρ, c1.vx - c2.vx, c1.vy - c2.vy, c1.vz - c2.vz)
*(α::Float64, c::Cell) = Cell(α * c.u, α * c.ρ, α * c.vx, α * c.vy, α * c.vz)
*(c::Cell, α::Float64) = Cell(c.u * α, c.ρ * α, c.vx * α, c.vy * α, c.vz * α)
/(c::Cell, α::Float64) = Cell(c.u / α, c.ρ / α, c.vx / α, c.vy / α, c.vz / α)
.+(c::Cell) = +(c)
.-(c::Cell) = -(c)
.+(c1::Cell, c2::Cell) = +(c1, c2)
.-(c1::Cell, c2::Cell) = -(c1, c2)
.*(α::Float64, c::Cell) = *(α, c)
.*(c::Cell, α::Float64) = *(c, α)
./(c::Cell, α::Float64) = /(c, α)

Norm(c::Cell) = Norm(c.u) + Norm(c.ρ) + Norm(c.vx) + Norm(c.vy) + Norm(c.vz)

export energy
@fastmath function energy(c::Cell)
    1//2 * (c.ρ^2 + c.vx^2 + c.vy^2 + c.vz^2)
end

export analytic
@fastmath function analytic(t::Float64, x::Float64, y::Float64, z::Float64)
    # Standing wave
    u = A * sin(ω * t) * sin(kx * x) * sin(ky * y) * sin(kz * z)
    ρ = A * ω * cos(ω * t) * sin(kx * x) * sin(ky * y) * sin(kz * z)
    vx = A * kx * sin(ω * t) * cos(kx * x) * sin(ky * y) * sin(kz * z)
    vy = A * ky * sin(ω * t) * sin(kx * x) * cos(ky * y) * sin(kz * z)
    vz = A * kz * sin(ω * t) * sin(kx * x) * sin(ky * y) * cos(kz * z)
    Cell(u, ρ, vx, vy, vz)
end

export init
function init(t::Float64, x::Float64, y::Float64, z::Float64)
    analytic(t, x, y, z)
end

import Base: error
export error
function error(c::Cell, t::Float64, x::Float64, y::Float64, z::Float64)
    c .- analytic(t, x, y, z)
end

export rhs
@fastmath function rhs(c::Cell, bxm::Cell, bxp::Cell, bym::Cell, byp::Cell,
                       bzm::Cell, bzp::Cell)
    udot = c.ρ
    ρdot = ((bxp.vx - bxm.vx) / 2dx +
            (byp.vy - bym.vy) / 2dy +
            (bzp.vz - bzm.vz) / 2dz)
    vxdot = (bxp.ρ - bxm.ρ) / 2dx
    vydot = (byp.ρ - bym.ρ) / 2dy
    vzdot = (bzp.ρ - bzm.ρ) / 2dz
    Cell(udot, ρdot, vxdot, vydot, vzdot)
end



# Grid

export Grid
type Grid
    time::Float64
    cells::Array{Cell,3}
    Grid(t::Real, c::Array{Cell,3}) = new(Float64(t), c)
end

import Base: +, -, *, /, .+, .-, .*, ./
export +, -, *, /, .+, .-, .*, ./
+(g::Grid) = Grid(+g.time, .+(g.cells))
-(g::Grid) = Grid(-g.time, .-(g.cells))
+(g1::Grid, g2::Grid) = Grid(g1.time + g2.time, g1.cells + g2.cells)
-(g1::Grid, g2::Grid) = Grid(g1.time - g2.time, g1.cells - g2.cells)
*(α::Float64, g::Grid) = Grid(α * g.time, map(x->α*x, g.cells))
*(g::Grid, α::Float64) = Grid(g.time * α, map(x->x*α, g.cells))
/(g::Grid, α::Float64) = Grid(g.time / α, map(x->x/α, g.cells))
.+(g::Grid) = +(g)
.-(g::Grid) = -(g)
.+(g1::Grid, g2::Grid) = +(g1, g2)
.-(g1::Grid, g2::Grid) = -(g1, g2)
.*(α::Float64, g::Grid) = *(α, g)
.*(g::Grid, α::Float64) = *(g, α)
./(g::Grid, α::Float64) = /(g, α)

Norm(g::Grid) = mapreduce(Norm, +, g.cells[2:end-1, 2:end-1, 2:end-1])

export energy
function energy(g::Grid)
    c = g.cells
    ϵ = similar(c, Float64)
    @inbounds @simd for i in eachindex(c)
        ϵ[i] = energy(c[i])
    end
    ϵ
end

import Base: error
export error
function error(g::Grid)
    t = g.time
    c = g.cells
    err = similar(c)
    @inbounds for k=1:size(c,3), j=1:size(c,2)
        @simd for i=1:size(c,1)
            err[i,j,k] = error(c[i,j,k], t, x(i), y(j), z(k))
        end
    end
    Grid(t, err)
end

export init
function init(t::Real)
    c = Array{Cell}(ni,nj,nk)
    @inbounds for k=1:size(c,3), j=1:size(c,2)
        @simd for i=1:size(c,1)
            c[i,j,k] = init(t, x(i), y(j), z(k))
        end
    end
    Grid(t, c)
end

export rhs
function rhs(g::Grid)
    c = g.cells
    r = similar(c)
    @inbounds for k=2:size(c,3)-1, j=2:size(c,2)-1
        @simd for i=2:size(c,1)-1
            r[i,j,k] = rhs(c[i,j,k], c[i-1,j,k], c[i+1,j,k], c[i,j-1,k],
                           c[i,j+1,k], c[i,j,k-1], c[i,j,k+1])
        end
    end
    periodic!(r)
    Grid(1, r)
end



# Simulation

export State
type State
    iter::Int
    state::Grid
    rhs::Grid
    ϵ::Array{Float64,3}
    function State(i::Integer, g::Grid)
        r = rhs(g)::Grid
        ϵ = energy(g)::Array{Float64,3}
        new(Int(i), g, r, ϵ)::State
    end
end

export init
function init()
    State(0, init(tmin))
end

export rk2
function rk2(s::State)
    s0 = s.state
    r0 = s.rhs
    s1 = s0 + (dt/2) * r0
    r1 = rhs(s1)
    s2 = s0 + dt * r1
    State(s.iter+1, s2)
end

end
