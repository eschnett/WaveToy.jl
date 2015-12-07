module WaveToy

using HDF5



# Parameters

const dim = 3

const ni = 8
const nj = 1
const nk = 1

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

xpos(i) = xmin + (i - 2) * dx
ypos(j) = ymin + (j - 2) * dy
zpos(k) = zmin + (k - 2) * dz
ipos(x) = mod(round(Int, (x - xmin) / dx), ni) + 2
jpos(y) = mod(round(Int, (y - ymin) / dy), nj) + 2
kpos(z) = mod(round(Int, (z - zmin) / dz), nk) + 2

const np = 1
const pm = 1.0
const pq = 1.0

const A = 1.0
const kx = ni==1 ? 0.0 : 2π / (xmax - xmin)
const ky = nj==1 ? 0.0 : 2π / (ymax - ymin)
const kz = nk==1 ? 0.0 : 2π / (zmax - zmin)
const ω = sqrt(kx^2 + ky^2 + kz^2)

const epsdiss = 0.01

const tmin = 0.0
const tmax = maxL / (ω/2π)

const cfl = 1 / (ω/2π) / 2
const dt = cfl * min(dx, dy, dz)
const niters = ceil(Int, (tmax - tmin) / dt)

const outinfo_every = 1 # niters ÷ 16
const outfile_every = 1 # niters ÷ 16



# Multitasking

import Base: wait, isready, get
export wait, isready, get

#=
immutable Future{T}
    task::Task
    Future(t::Task) = new(t)
    Future(val::T) = new(@schedule val)
end
macro future(T, expr)
    :(Future{$(esc(T))}(@schedule $(esc(expr))))
end
wait(ftr::Future) = (wait(ftr.task); nothing)
isready(ftr::Future) = istaskdone(ftr.task)
get{T}(ftr::Future{T}) = wait(ftr.task)::T
=#

immutable Future{T}
    val::T
    Future(val::T) = new(val)
end
macro future(T, expr)
    :(Future{$(esc(T))}($(esc(expr))))
end
wait(ftr::Future) = nothing
isready(ftr::Future) = true
get{T}(ftr::Future{T}) = ftr.val



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
        new(x.count + y.count, x.sum + y.sum, x.sum2 + y.sum2,
            x.sumabs + y.sumabs, min(x.min, y.min), max(x.max, y.max),
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
@inline sdv(n::Norm) = sqrt(max(0.0, n.count * n.sum2 - n.sum^2)) / n.count
@inline norm1(n::Norm) = n.sumabs / n.count
@inline norm2(n::Norm) = sqrt(n.sum2 / n.count)
@inline norminf(n::Norm) = n.maxabs
import Base: zero, +
export zero, +
@inline zero(::Type{Norm}) = Norm()
@inline +(x::Norm, y::Norm) = Norm(x, y)



# Physics

"""
    L_field    = Int 1/2 [-u,t^2 + u,a^2]
    L_particle = Sum [- 1/(2m) p_a^2 + q u]

    u,tt  = u,aa
    q_a,t = p_a/m
    p_a,t = - q u,a
"""
function physics end



# Cells

export Cell
immutable Cell
    u::Float64
    ft::Float64
    fx::Float64
    fy::Float64
    fz::Float64
end

import Base: +, -, *, /, \
export +, -, *, /, \, axpy
@inline +(c::Cell) = Cell(+c.u, +c.ft, +c.fx, +c.fy, +c.fz)
@inline -(c::Cell) = Cell(-c.u, -c.ft, -c.fx, -c.fy, -c.fz)
@inline +(c1::Cell, c2::Cell) =
    Cell(c1.u + c2.u, c1.ft + c2.ft, c1.fx + c2.fx, c1.fy + c2.fy,
        c1.fz + c2.fz)
@inline -(c1::Cell, c2::Cell) =
    Cell(c1.u - c2.u, c1.ft - c2.ft, c1.fx - c2.fx, c1.fy - c2.fy,
        c1.fz - c2.fz)
@inline *(α::Float64, c::Cell) =
    Cell(α * c.u, α * c.ft, α * c.fx, α * c.fy, α * c.fz)
@inline *(c::Cell, α::Float64) =
    Cell(c.u * α, c.ft * α, c.fx * α, c.fy * α, c.fz * α)
@inline \(α::Float64, c::Cell) =
    Cell(α \ c.u, α \ c.ft, α \ c.fx, α \ c.fy, α \ c.fz)
@inline /(c::Cell, α::Float64) =
    Cell(c.u / α, c.ft / α, c.fx / α, c.fy / α, c.fz / α)
@inline axpy(α::Float64, c1::Cell, c2::Cell) =
    Cell(α * c1.u + c2.u, α * c1.ft + c2.ft, α * c1.fx + c2.fx,
        α * c1.fy + c2.fy, α * c1.fz + c2.fz)

@inline Norm(c::Cell) =
    Norm(c.u) + Norm(c.ft) + Norm(c.fx) + Norm(c.fy) + Norm(c.fz)

#=
export constraints
@inline function constraints(c::Cell)
    # TODO: u,i == fi
    # 0 = u,ij - u,ji == fi,j - fj,i
    # wi = epsilon_i^jk dj fk
    wx = (bpy.fz - bmy.fz) / 2dy - (bpz.fy - bmz.fy) / 2dz
    wy
    wz
end
=#

export energy
@inline function energy(c::Cell)
    1/2 * (c.ft^2 + c.fx^2 + c.fy^2 + c.fz^2)
end

"No wave"
@inline function empty(t::Float64, i::Int, j::Int, k::Int)
    Cell(0, 0, 0, 0, 0)
end

"Standing wave"
@inline function standing(t::Float64, i::Int, j::Int, k::Int)
    x,y,z = xpos(i), ypos(j), zpos(k)
    u = A * sin(ω * t) * cos(kx * x) * cos(ky * y) * cos(kz * z)
    ft = A * ω * cos(ω * t) * cos(kx * x) * cos(ky * y) * cos(kz * z)
    fx = - A * kx * sin(ω * t) * sin(kx * x) * cos(ky * y) * cos(kz * z)
    fy = - A * ky * sin(ω * t) * cos(kx * x) * sin(ky * y) * cos(kz * z)
    fz = - A * kz * sin(ω * t) * cos(kx * x) * cos(ky * y) * sin(kz * z)
    Cell(u, ft, fx, fy, fz)
end

export analytic
@inline function analytic(t::Float64, i::Int, j::Int, k::Int)
    standing(t, i, j, k)
end

export init
@inline function init(t::Float64, i::Int, j::Int, k::Int)
    analytic(t, i, j, k)
end

import Base: error
export error
@inline function error(c::Cell, t::Float64, i::Int, j::Int, k::Int)
    c - analytic(t, i, j, k)
end

export rhs
@inline function rhs(c::Cell, bxm::Cell, bxp::Cell, bym::Cell, byp::Cell,
        bzm::Cell, bzp::Cell)
    udot = c.ft
    ftdot = ((bxp.fx - bxm.fx) / 2dx +
             (byp.fy - bym.fy) / 2dy +
             (bzp.fz - bzm.fz) / 2dz)
    fxdot = (bxp.ft - bxm.ft) / 2dx
    fydot = (byp.ft - bym.ft) / 2dy
    fzdot = (bzp.ft - bzm.ft) / 2dz
    ftdot += epsdiss * ((bxm.ft - 2*c.ft + bxp.ft) / dx +
                        (bym.ft - 2*c.ft + byp.ft) / dy +
                        (bzm.ft - 2*c.ft + bzp.ft) / dz)
    fxdot += epsdiss * ((bxm.fx - 2*c.fx + bxp.fx) / dx +
                        (bym.fx - 2*c.fx + byp.fx) / dy +
                        (bzm.fx - 2*c.fx + bzp.fx) / dz)
    fydot += epsdiss * ((bxm.fy - 2*c.fy + bxp.fy) / dx +
                        (bym.fy - 2*c.fy + byp.fy) / dy +
                        (bzm.fy - 2*c.fy + bzp.fy) / dz)
    fzdot += epsdiss * ((bxm.fz - 2*c.fz + bxp.fz) / dx +
                        (bym.fz - 2*c.fz + byp.fz) / dy +
                        (bzm.fz - 2*c.fz + bzp.fz) / dz)
    Cell(udot, ftdot, fxdot, fydot, fzdot)
end

#TODO: rhs(c::Cell, p::Particle)



# Particles

export Particle
immutable Particle
    m::Float64
    q::Float64
    qx::Float64
    qy::Float64
    qz::Float64
    px::Float64
    py::Float64
    pz::Float64
end

import Base: +, -, *, /, \
export +, -, *, /, \, axpy
@inline +(p::Particle) =
    Particle(+p.m, +p.q, +p.qx, +p.qy, +p.qz, +p.px, +p.py, +p.pz)
@inline -(p::Particle) =
    Particle(-p.m, -p.q, -p.qx, -p.qy, -p.qz, -p.px, -p.py, -p.pz)
@inline +(p1::Particle, p2::Particle) =
    Particle(p1.m + p2.m, p1.q + p2.q, p1.qx + p2.qx, p1.qy + p2.qy,
        p1.qz + p2.qz, p1.px + p2.px, p1.py + p2.py, p1.pz + p2.pz)
@inline -(p1::Particle, p2::Particle) =
    Particle(p1.m - p2mq, p1.q - p2.q, p1.qx - p2.qx, p1.qy - p2.qy,
        p1.qz - p2.qz, p1.px - p2.px, p1.py - p2.py, p1.pz - p2.pz)
@inline *(α::Float64, p::Particle) =
    Particle(α * p.m, α * p.q, α * p.qx, α * p.qy, α * p.qz, α * p.px, α * p.py,
        α * p.pz)
@inline *(p::Particle, α::Float64) =
    Particle(p.m * α, p.q * α, p.qx * α, p.qy * α, p.qz * α, p.px * α, p.py * α,
        p.pz * α)
@inline \(α::Float64, p::Particle) =
    Particle(α \ p.m, α \ p.q, α \ p.qx, α \ p.qy, α \ p.qz, α \ p.px, α \ p.py,
        α \ p.pz)
@inline /(p::Particle, α::Float64) =
    Particle(p.m / α, p.q / α, p.qx / α, p.qy / α, p.qz / α, p.px / α, p.py / α,
        p.pz / α)
@inline axpy(α::Float64, p1::Particle, p2::Particle) =
    Particle(α * p1.m + p2.m, α * p1.q + p2.q, α * p1.qx + p2.qx,
        α * p1.qy + p2.qy, α * p1.qz + p2.qz, α * p1.px + p2.px,
        α * p1.py + p2.py, α * p1.pz + p2.pz)

@inline Norm(p::Particle) =
    (Norm(p.m) + Norm(p.q) + Norm(p.qx) + Norm(p.qy) + Norm(p.qz) + Norm(p.px) +
        Norm(p.py) + Norm(p.pz))

export energy
@inline function energy(p::Particle)
    1/(2p.m) * (p.px^2 + p.py^2 + p.pz^2)
end
@inline function energy(p::Particle, c::Cell)
    p.q * c.u
end

"resting"
@inline function resting(t::Float64, i::Int)
    Particle(pm, pq, (xmin+xmax)/4, (ymin+ymax)/4, (zmin+zmax)/4, 0, 0, 0)
end

export analytic
@inline function analytic(t::Float64, i::Int)
    resting(t, i)
end

export init
@inline function init(t::Float64, i::Int)
    analytic(t, i)
end

import Base: error
export error
@inline function error(c::Particle, t::Float64, i::Int)
    p - analytic(t, i)
end

export rhs
@inline function rhs(p::Particle)
    mdot = 0
    qdot = 0
    qxdot = p.px / p.m
    qydot = p.py / p.m
    qzdot = p.pz / p.m
    pxdot = 0
    pydot = 0
    pzdot = 0
    Particle(mdot, qdot, qxdot, qydot, qzdot, pxdot, pydot, pzdot)
end
@inline function rhs(p::Particle, c::Cell)
    mdot = 0
    qdot = 0
    qxdot = 0
    qydot = 0
    qzdot = 0
    pxdot = - p.q * c.fx
    pydot = - p.q * c.fy
    pzdot = - p.q * c.fz
    Particle(mdot, qdot, qxdot, qydot, qzdot, pxdot, pydot, pzdot)
end



# Grids

export Grid
type Grid
    time::Float64
    cells::Array{Cell,3}
    particles::Vector{Particle}
    Grid(t::Real, c::Array{Cell,3}, p::Vector{Particle}) = new(Float64(t), c, p)
end

import Base: copy
export copy
copy(g::Grid) = Grid(g.time, copy(g.cells), copy(g.particles))

import Base: +, -, *, /, \, axpy!
export +, -, *, /, \, axpy, axpy!
+(g::Grid) = Grid(+g.time, +g.cells, +g.particles)
-(g::Grid) = Grid(-g.time, -g.cells, -g.particles)
+(g1::Grid, g2::Grid) =
    Grid(g1.time + g2.time, g1.cells + g2.cells, g1.particles + g2.particles)
-(g1::Grid, g2::Grid) =
    Grid(g1.time - g2.time, g1.cells - g2.cells, g1.particles - g2.particles)
*(α::Float64, g::Grid) = Grid(α * g.time, α * g.cells, α * g.particles)
*(g::Grid, α::Float64) = Grid(g.time * α, g.cells * α, g.particles * α)
\(α::Float64, g::Grid) = Grid(α \ g.time, α \ g.cells, α \ g.particles)
/(g::Grid, α::Float64) = Grid(g.time / α, g.cells / α, g.particles / α)
axpy(α::Float64, x::Grid, y::Grid) = axpy!(α, x, copy(y))
function axpy!(α::Float64, x::Grid, y::Grid)
    y.time += α * x.time
    axpy!(α, x.cells, y.cells)
    axpy!(α, x.particles, y.particles)
    y
end

# Norm(g::Grid) = mapreduce(Norm, +, g.cells[2:end-1, 2:end-1, 2:end-1])
# Norm(a::Array{Float64,dim}) = mapreduce(Norm, +, a[2:end-1, 2:end-1, 2:end-1])
function Norm{T}(a::Array{T,dim})
    n = Norm()
    # TODO: use @simd
    @inbounds for elt in sub(a, 2:size(a,1)-1, 2:size(a,2)-1, 2:size(a,3)-1)
        n += Norm(elt)
    end
    n
end
function Norm{T}(a::Vector{T})
    n = Norm()
    # TODO: use @simd
    @inbounds for elt in a
        n += Norm(elt)
    end
    n
end
Norm(g::Grid) = Norm(g.cells) + Norm(g.particles)

export energy, total_energy
function energy(g::Grid)
    c = g.cells
    ϵc = similar(c, Float64)
    @inbounds @simd for i in eachindex(c)
        ϵc[i] = energy(c[i])
    end
    p = g.particles
    ϵp = similar(p, Float64)
    @inbounds @simd for n in eachindex(p)
        ϵp[n] = energy(p[n])
    end
    ϵc, ϵp
end
function total_energy(ϵc::Array{Float64,dim}, ϵp::Vector{Float64})
    sum(Norm(ϵc)) * (dx*dy*dz) + sum(Norm(ϵp))
end

import Base: error
export error
function error(g::Grid)
    t = g.time
    c = g.cells
    ec = similar(c)
    @inbounds for k in 1:size(ec,3), j in 1:size(ec,2)
        @simd for i in 1:size(ec,1)
            ec[i,j,k] = error(c[i,j,k], t, i, j, k)
        end
    end
    p = g.particles
    ep = similar(p)
    @inbounds @simd for n in eachindex(ep)
        ep[n] = error(p[n], t)
    end
    Grid(t, ec, ep)
end

export init
function init(t::Real)
    c = Array{Cell}(ni+2,nj+2,nk+2)
    @inbounds for k in  1:size(c,3), j in 1:size(c,2)
        @simd for i in 1:size(c,1)
            c[i,j,k] = init(t, i, j, k)
        end
    end
    p = Vector{Particle}(np)
    @inbounds @simd for n in eachindex(p)
        p[n] = init(t, n)
    end
    Grid(t, c, p)
end

export rhs
function rhs(g::Grid)
    c = g.cells
    rc = similar(c)
    @inbounds for k in 2:size(rc,3)-1, j in 2:size(rc,2)-1
        @simd for i in 2:size(rc,1)-1
            rc[i,j,k] = rhs(c[i,j,k], c[i-1,j,k], c[i+1,j,k], c[i,j-1,k],
                            c[i,j+1,k], c[i,j,k-1], c[i,j,k+1])
        end
    end
    periodic!(rc)
    p = g.particles
    rp = similar(p)
    @inbounds @simd for n in eachindex(rp)
        rp[n] = rhs(p[n])
        i,j,k = ipos(p[n].qx), jpos(p[n].qy), kpos(p[n].qz)
        rp[n] += rhs(p[n], c[i,j,k])
    end
    Grid(1, rc, rp)
end



# Simulations

export State
type State
    iter::Int
    state::Future{Grid}
    rhs::Future{Grid}
    ϵ::Future{Tuple{Array{Float64,3}, Vector{Float64}}}
    etot::Future{Float64}
    function State(i::Integer, g::Future{Grid})
        r = @future Grid rhs(get(g))
        ϵ = @future Tuple{Array{Float64,3}, Vector{Float64}} energy(get(g))
        e = @future Float64 total_energy(get(ϵ)...)
        new(Int(i), g, r, ϵ, e)
    end
end

export init
function init()
    State(0, @future Grid init(tmin))
end

export rk2_central
"Central RK2 (is not TVD)"
function rk2_central(s::State)
    s0 = s.state
    r0 = s.rhs
    # s1 = s0 + (dt/2) * r0
    s1 = @future Grid axpy(dt/2, get(r0), get(s0))
    r1 = @future Grid rhs(get(s1))
    # s2 = s0 + dt * r1
    s2 = @future Grid axpy(dt, get(r1), get(s0))
    # s2 = s0
    # axpy!(dt, r1, s2)
    s2 = @future Grid axpy!(dt/2, get(r1), axpy(dt/2, get(r0), get(s0)))
    State(s.iter+1, s2)
end

export rk2_tvd
"TVD RK2"
function rk2_tvd(s::State)
    s0 = s.state
    r0 = s.rhs
    # s1 = s0 + dt * r0
    s1 = @future Grid axpy(dt, get(r0), get(s0))
    r1 = @future Grid rhs(get(s1))
    # s2 = 1/2 s0 + 1/2 s1 + 1/2 dt r1
    #    = s0 + 1/2 dt r0 + 1/2 dt r1
    s2 = @future Grid axpy!(dt/2, get(r1), axpy(dt/2, get(r0), get(s0)))
    State(s.iter+1, s2)
end



# Output

function output_info(tok::Future{Void}, s::State)
    @future Void begin
        wait(tok)
        @printf "   n: %4d" s.iter
        @printf "   t: %5.2f" get(s.state).time
        @printf "   ft[0]: %7.3f" get(s.state).cells[2,2,2].ft
        if !isempty(get(s.state).particles)
            @printf "   qx[0]: %7.3f" get(s.state).particles[1].qx
        end
        @printf "   E: %6.3f" get(s.etot)
        println()
        flush(STDOUT)
        nothing
    end
end

function output_file(tok::Future{Void}, s::State)
    if s.iter == 0
        try rm("output.h5") end
    end
    @future Void begin
        h5open("output.h5", "w") do f
            for field in (:u, :ft, :fx, :fy, :fz)
                write(f, "State/iter=$(s.iter)/cells/$field",
                    map(c->c.(field),
                        sub(get(s.state).cells, 2:ni+1, 2:nj+1, 2:nk+1)))
            end
            if !isempty(get(s.state).particles)
                for field in (:m, :q, :qx, :qy, :qz, :px, :py, :pz)
                    write(f, "State/iter=$(s.iter)/particles/$field",
                        map(p->p.(field), get(s.state).particles))
                end
            end
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



# Driver

function main()
    println("WaveToy   [$ni,$nj,$nk],[$np]")
    tok = init_output()
    s = init()
    tok = output(tok, s)
    while s.iter < niters
        s = rk2_tvd(s)
        tok = output(tok, s)
    end
    wait_output(tok)
    println("Done.")
end

end
