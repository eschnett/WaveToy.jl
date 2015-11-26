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



# Cells

export Cell
immutable Cell
    u::Float64
    ρ::Float64
    vx::Float64
    vy::Float64
    vz::Float64
end

import Base: .+, .-, .*, ./
export .+, .-, .*, ./
.+(c::Cell) = Cell(+c.u, +c.ρ, +c.vx, +c.vy, +c.vz)
.-(c::Cell) = Cell(-c.u, -c.ρ, -c.vx, -c.vy, -c.vz)
.+(c1::Cell, c2::Cell) =
    Cell(c1.u + c2.u, c1.ρ + c2.ρ, c1.vx + c2.vx, c1.vy + c2.vy, c1.vz + c2.vz)
.-(c1::Cell, c2::Cell) =
    Cell(c1.u - c2.u, c1.ρ - c2.ρ, c1.vx - c2.vx, c1.vy - c2.vy, c1.vz - c2.vz)
.*(α::Float64, c::Cell) = Cell(α * c.u, α * c.ρ, α * c.vx, α * c.vy, α * c.vz)
.*(c::Cell, α::Float64) = Cell(c.u * α, c.ρ * α, c.vx * α, c.vy * α, c.vz * α)
./(c::Cell, α::Float64) = Cell(c.u / α, c.ρ / α, c.vx / α, c.vy / α, c.vz / α)

import Base: norm
export norm
norm(c::Cell, p=2) = norm((c.u, c.ρ, c.vx, c.vy, c.vz), p)

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
    iter::Int
    time::Float64
    cells::Array{Cell,3}
    Grid(iter, time, cells) = new(iter, time, cells)
end

export energy
@fastmath function energy(c::Cell)
    1//2 * (c.ρ^2 + c.vx^2 + c.vy^2 + c.vz^2)
end
function energy(g::Grid)
    c = g.cells
    ϵ = similar(Float64, c)
    @inbounds @simd for i in eachindex(ϵ)
        ϵ[i] = energy(c[i])
    end
    ϵ
end

export error
function error(g::Grid)
    t = g.time
    c = g.cells
    err = similar(c)
    @inbounds for k=1:size(c,3), j=1:size(c,2)
        @simd for i=1:size(c,1)
            err[i,j,k] = c[i,j,k] .- analytic(t, x(i), y(j), z(k))
        end
    end
    err
end

import Base: norm
export norm
norm(g::Grid, p=2) = dx*dy*dz * norm(g.cells[2:end-1, 2:end-1, 2:end-1], p)

export init
function init(t::Float64)
    c = Array{Cell}(ni+2,nj+2,nk+2)
    @inbounds for k=1:size(c,3), j=1:size(c,2)
        @simd for i=1:size(c,1)
            c[i,j,k] = analytic(t, x(i), y(j), z(k))
        end
    end
    Grid(0, t, c)
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
    Grid(g.iter, 0.0, r)
end



# Simulation

end
