using WaveToy
using Base.Test

# Test norms
@test count(Norm()) == 0
n0 = Norm(0)
@test sum(n0) == 0
n1 = Norm(1)
@test min(n1) == 1
@test count(zero(Norm)) == 0
n2 = mapreduce(Norm, +, collect(1:10))
@test count(n2) == 10
@test sum(n2) == 55
@test min(n2) == 1
@test max(n2) == 10
@test avg(n2) == 5.5
@test sdv(n2) == sqrt(8.25)
@test norm1(n2) == 5.5
@test norm2(n2) == sqrt(38.5)
@test norminf(n2) == 10

# Test cells
ci = init(0.0, 1, 1, 1)
ce = error(ci, 0.0, 1, 1, 1)
@test ce == Cell(0.0, 0.0, 0.0, 0.0, 0.0)
c0 = Cell(0.0, 0.0, 0.0, 0.0, 0.0)
cn = Norm(c0)
@test norminf(cn) == 0
cϵ = energy(ce)
@test cϵ == 0.0
cr = rhs(c0, c0,c0,c0,c0,c0,c0)
@test cr == Cell(0.0, 0.0, 0.0, 0.0, 0.0)

# Test particles
pi = init(0.0, 1)
pe = error(pi, 0.0, 1)
@test pe == Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
p0 = Particle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
pn = Norm(p0)
@test norm1(pn) == 2/10
pϵ = energy(pe)
@test pϵ == 0.0
pr = rhs(p0)
pr += rhs(p0, c0)
@test pr == Particle(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Test grids
g0 = init(0.0)
ge = error(g0)
gn = Norm(ge)
@test norminf(gn) == 0
gϵ = energy(g0)
gr = rhs(g0)

# Test states

s0 = init()
s1 = rk2_tvd(s0)
