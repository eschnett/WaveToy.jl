module WaveToy

using HDF5



# Parameters

const dim = 3

const ni = 64
const nj = 64
const nk = 64

const xmin = 0.0
const ymin = 0.0
const zmin = 0.0
const xmax = 1.0
const ymax = 1.0
const zmax = 1.0
const maxL = max(xmax - xmin, ymax - ymin, zmax - zmin)

const dx = (xmax - xmin) / ni
const dy = (ymax - ymin) / nj
const dz = (zmax - zmin) / nk

x(i) = xmin + (i - 2) * dx
y(j) = ymin + (j - 2) * dy
z(k) = zmin + (k - 2) * dz

const A = 1.0
const kx = ni==1 ? 0.0 : 2π / (xmax - xmin)
const ky = nj==1 ? 0.0 : 2π / (ymax - ymin)
const kz = nk==1 ? 0.0 : 2π / (zmax - zmin)
const ω = sqrt(kx^2 + ky^2 + kz^2)

const tmin = 0.0
const tmax = maxL / (ω/2π)

const cfl = 1 / (ω/2π)
const dt = cfl * min(dx, dy, dz)
const niters = ceil(Int, (tmax - tmin) / dt)

const outinfo_every = niters ÷ 16
const outfile_every = 0



# Multitasking

immutable Future{T}
    task::Task
    Future(t::Task) = new(t)
    Future(val::T) = new(@schedule val)
end
import Base: wait, isready, get
export wait, isready, get
macro future(T, expr)
    :(Future{$(esc(T))}(@schedule $(esc(expr))))
end
wait(ftr::Future) = (wait(ftr.task); nothing)
isready(ftr::Future) = istaskdone(ftr.task)
get{T}(ftr::Future{T}) = wait(ftr.task)::T



# Array operations

function periodic!{T}(a::Array{T,dim})
    ni,nj,nk = size(a)
    a[1, 2:nj-1, 2:nk-1] = sub(a, ni-1, 2:nj-1, 2:nk-1)
    a[ni, 2:nj-1, 2:nk-1] = sub(a, 2, 2:nj-1, 2:nk-1)
    a[:, 1, 2:nk-1] = sub(a, :, nj-1, 2:nk-1)
    a[:, nj, 2:nk-1] = sub(a, :, 2, 2:nk-1)
    a[:, :, 1] = sub(a, :, :, nk-1)
    a[:, :, end] = sub(a, :, :, 2)
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
    @inline function Norm()
        new(0.0, 0.0, 0.0, 0.0, typemax(Float64), typemin(Float64), 0.0)
    end
    @inline function Norm(x0::Real)
        x = Float64(x0)
        new(1.0, x, x^2, abs(x), x, x, abs(x))
    end
    # TODO: @fastmath
    @inline function Norm(x::Norm, y::Norm)
        new(x.count+y.count, x.sum+y.sum, x.sum2+y.sum2,
            x.sumabs+y.sumabs, min(x.min, y.min), max(x.max, y.max),
            max(x.maxabs, y.maxabs))
    end
end
import Base: count, sum, min, max
export count, sum, min, max, avg, sdv, norm1, norm2, norminf
@inline count(n::Norm) = n.count
@inline sum(n::Norm) = n.sum
@inline min(n::Norm) = n.min
@inline max(n::Norm) = n.max
@inline avg(n::Norm) = n.sum / n.count
@inline sdv(n::Norm) =
    sqrt(max(0.0, n.count * n.sum2 - n.sum^2)) / n.count
@inline norm1(n::Norm) = n.sumabs / n.count
@inline norm2(n::Norm) = sqrt(n.sum2 / n.count)
@inline norminf(n::Norm) = n.maxabs
import Base: zero, +
export zero, +
@inline zero(::Type{Norm}) = Norm()
@inline +(x::Norm, y::Norm) = Norm(x, y)



# Cells

export Cell
immutable Cell
    u::Float64
    ρ::Float64
    vx::Float64
    vy::Float64
    vz::Float64
end

import Base: +, -, *, /, \
export +, -, *, /, \
@inline +(c::Cell) = Cell(+c.u, +c.ρ, +c.vx, +c.vy, +c.vz)
@inline -(c::Cell) = Cell(-c.u, -c.ρ, -c.vx, -c.vy, -c.vz)
@inline +(c1::Cell, c2::Cell) =
    Cell(c1.u + c2.u, c1.ρ + c2.ρ, c1.vx + c2.vx, c1.vy + c2.vy, c1.vz + c2.vz)
@inline -(c1::Cell, c2::Cell) =
    Cell(c1.u - c2.u, c1.ρ - c2.ρ, c1.vx - c2.vx, c1.vy - c2.vy, c1.vz - c2.vz)
@inline *(α::Float64, c::Cell) =
    Cell(α * c.u, α * c.ρ, α * c.vx, α * c.vy, α * c.vz)
@inline *(c::Cell, α::Float64) =
    Cell(c.u * α, c.ρ * α, c.vx * α, c.vy * α, c.vz * α)
@inline \(α::Float64, c::Cell) =
    Cell(α \ c.u, α \ c.ρ, α \ c.vx, α \ c.vy, α \ c.vz)
@inline /(c::Cell, α::Float64) =
    Cell(c.u / α, c.ρ / α, c.vx / α, c.vy / α, c.vz / α)
@inline lincom(c1::Cell, α::Float64, c2::Cell) =
    Cell(c1.u + α * c2.u, c1.ρ + α * c2.ρ, c1.vx + α * c2.vx, c1.vy + α * c2.vy,
        c1.vz + α * c2.vz)

@inline Norm(c::Cell) =
    Norm(c.u) + Norm(c.ρ) + Norm(c.vx) + Norm(c.vy) + Norm(c.vz)

export energy
@inline function energy(c::Cell)
    1/2 * (c.ρ^2 + c.vx^2 + c.vy^2 + c.vz^2)
end

export analytic
@inline function analytic(t::Float64, x::Float64, y::Float64, z::Float64)
    # Standing wave
    u = A * sin(ω * t) * cos(kx * x) * cos(ky * y) * cos(kz * z)
    ρ = A * ω * cos(ω * t) * cos(kx * x) * cos(ky * y) * cos(kz * z)
    vx = - A * kx * sin(ω * t) * sin(kx * x) * cos(ky * y) * cos(kz * z)
    vy = - A * ky * sin(ω * t) * cos(kx * x) * sin(ky * y) * cos(kz * z)
    vz = - A * kz * sin(ω * t) * cos(kx * x) * cos(ky * y) * sin(kz * z)
    Cell(u, ρ, vx, vy, vz)
end

export init
@inline function init(t::Float64, x::Float64, y::Float64, z::Float64)
    analytic(t, x, y, z)
end

import Base: error
export error
@inline function error(c::Cell, t::Float64, x::Float64, y::Float64,
        z::Float64)
    c - analytic(t, x, y, z)
end

export rhs
@inline function rhs(c::Cell, bxm::Cell, bxp::Cell, bym::Cell,
        byp::Cell, bzm::Cell, bzp::Cell)
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

import Base: +, -, *, /, \
export +, -, *, /, \
# +(g::Grid) = Grid(+g.time, .+(g.cells))
# -(g::Grid) = Grid(-g.time, .-(g.cells))
# +(g1::Grid, g2::Grid) = Grid(g1.time + g2.time, g1.cells + g2.cells)
# -(g1::Grid, g2::Grid) = Grid(g1.time - g2.time, g1.cells - g2.cells)
# *(α::Float64, g::Grid) = Grid(α * g.time, map(x->α*x, g.cells))
# *(g::Grid, α::Float64) = Grid(g.time * α, map(x->x*α, g.cells))
# /(g::Grid, α::Float64) = Grid(g.time / α, map(x->x/α, g.cells))
function +(g::Grid)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = +c[i]
        end
        Grid(+g.time, r)
    end
end
function -(g::Grid)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = -c[i]
        end
        Grid(-g.time, r)
        end
end
function +(g1::Grid, g2::Grid)
    begin
        c1,c2 = g1.cells, g2.cells
        r = similar(c1)
        @assert size(c2) == size(c1)
        @inbounds @simd for i in eachindex(r)
            r[i] = c1[i] + c2[i]
        end
        Grid(g1.time + g1.time, r)
    end
end
function -(g1::Grid, g2::Grid)
    begin
        c1,c2 = g1.cells, g2.cells
        r = similar(c1)
        @assert size(c2) == size(c1)
        @inbounds @simd for i in eachindex(r)
            r[i] = c1[i] - c2[i]
        end
        Grid(g1.time - g1.time, r)
    end
end
function *(α::Float64, g::Grid)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = α * c[i]
        end
        Grid(α * g.time, r)
    end
end
function *(g::Grid, α::Float64)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = c[i] * α
        end
        Grid(g.time * α, r)
    end
end
function \(α::Float64, g::Grid)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = α \ c[i]
        end
        Grid(α \ g.time, r)
    end
end
function /(g::Grid, α::Float64)
    begin
        c = g.cells
        r = similar(c)
        @inbounds @simd for i in eachindex(r)
            r[i] = c[i] / α
        end
        Grid(g.time / α, r)
    end
end
function lincom(g1::Grid, α::Float64, g2::Grid)
    c1,c2 = g1.cells, g2.cells
    r = similar(c1)
    @assert size(c2) == size(c1)
    @inbounds @simd for i in eachindex(r)
        r[i] = c1[i] + α * c2[i]
    end
    Grid(g1.time + α * g1.time, r)
end
function lincom!(g1::Grid, α::Float64, g2::Grid)
    c1,c2 = g1.cells, g2.cells
    @assert size(c2) == size(c1)
    @inbounds @simd for i in eachindex(c1)
        c1[i] = c1[i] + α * c2[i]
    end
    g1.time = g1.time + α * g2.time
    g1
end


# Norm(g::Grid) = mapreduce(Norm, +, g.cells[2:end-1, 2:end-1, 2:end-1])
# Norm(a::Array{Float64,dim}) = mapreduce(Norm, +, a[2:end-1, 2:end-1, 2:end-1])
function Norm{T}(a::Array{T,dim})
    n = Norm()
    # TODO: use @simd
    @inbounds for k=2:size(a,3)-1, j=1:size(a,2)-1, i=2:size(a,1)-1
        n += Norm(a[i,j,k])
    end
    n
end
Norm(g::Grid) = Norm(g.cells)

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
    c = Array{Cell}(ni+2,nj+2,nk+2)
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
    state::Future{Grid}
    rhs::Future{Grid}
    ϵ::Future{Array{Float64,3}}
    etot::Future{Float64}
    function State(i::Integer, g::Future{Grid})
        r = @future Grid rhs(get(g))
        ϵ = @future Array{Float64,3} energy(get(g))
        e = @future Float64 sum(Norm(get(ϵ))) * dx*dy*dz
        new(Int(i), g, r, ϵ, e)
    end
end

export init
function init()
    State(0, @future Grid init(tmin))
end

export rk2
function rk2(s::State)
    s0 = s.state
    r0 = s.rhs
    # s1 = s0 + (dt/2) * r0
    s1 = @future Grid lincom(get(s0), dt/2, get(r0))
    r1 = @future Grid rhs(get(s1))
    # s2 = s0 + dt * r1
    s2 = @future Grid lincom(get(s0), dt, get(r1))
    # s2 = s0
    # lincom!(s2, dt, r1)
    State(s.iter+1, s2)
end



# Output

function output_info(tok::Future{Void}, s::State)
    @future Void begin
        wait(tok)
        @printf "   n: %4d" s.iter
        @printf "   t: %5.2f" get(s.state).time
        @printf "   ρ[0]: %7.3f" get(s.state).cells[2,2,2].ρ
        @printf "   E: %6.3f" get(s.etot)
        println()
        nothing
    end
end

function output_file(tok::Future{Void}, s::State)
    if s.iter == 0
        try rm("output.h5") end
    end
    @future Void begin
        for field in (:u, :ρ, :vx, :vy, :vz)
            h5write("output.h5", "State/iter=$(s.iter)/$field",
                map(c->c.(field),
                    sub(get(s.state).cells, 2:ni+1, 2:nj+1, 2:nk+1)))
        end
        wait(tok)
        nothing
    end
end

export init_output, output, finish_output
function init_output()
    Future{Void}(nothing), Future{Void}(nothing)
end
function output(tok::Tuple{Future{Void}, Future{Void}}, s::State)
    itok, ftok = tok
    if (outinfo_every > 0 &&
            (s.iter == niters || mod(s.iter, outinfo_every) == 0))
        itok = output_info(itok, s)
    end
    if (outfile_every > 0 &&
            (s.iter == niters || mod(s.iter, outfile_every) == 0))
        ftok = output_file(ftok, s)
    end
    return itok, ftok
end
function wait_output(tok::Tuple{Future{Void}, Future{Void}})
    itok, ftok = tok
    wait(itok)
    wait(ftok)
end



function main()
    println("WaveToy   [$ni,$nj,$nk]")
    tok = init_output()
    s = init()
    tok = output(tok, s)
    while s.iter < niters
        s = rk2(s)
        tok = output(tok, s)
    end
    wait_output(tok)
    println("Done.")
end

end
