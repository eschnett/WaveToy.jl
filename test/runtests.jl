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
c0 = init(0.0, 0.0, 0.0, 0.0)
ce = error(c0, 0.0, 0.0, 0.0, 0.0)
@test ce == Cell(0.0, 0.0, 0.0, 0.0, 0.0)
cn = Norm(ce)
@test norminf(cn) == 0
cϵ = energy(c0)
@test cϵ == 0.0
cr = rhs(c0, c0,c0,c0,c0,c0,c0)
@test cr == Cell(0.0, 0.0, 0.0, 0.0, 0.0)

# Test grids
g0 = init(0.0)
ge = error(g0)
gn = Norm(ge)
@test norminf(gn) == 0
gϵ = energy(g0)
gr = rhs(g0)

# Test states

s0 = init()
s1 = rk2(s0)
