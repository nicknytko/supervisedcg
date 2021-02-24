# Navier-Stokes equations
# =======================

from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

M = Mesh('bfs.msh')
V = VectorFunctionSpace(M, "CG", 2) # Velocity
W = FunctionSpace(M, "CG", 1) # Pressure
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(800.0)

F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), 2), # Inlet velocity = (1,0)
        DirichletBC(Z.sub(1), Constant(0), 3),      # Outlet pressure = 0
        DirichletBC(Z.sub(0), Constant((0, 0)), 9)] # Wall velocity = (0,0)

appctx = {"Re": Re, "velocity_space": 0}

solve(F==0, up, bcs=bcs,
      solver_parameters={"ksp_type": "gmres",
                         "mat_type": "aij",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"})


u, p = up.split()
u.rename("Velocity")
p.rename("Pressure")

plt.figure(figsize=(8,2))
tripcolor(u, cmap='plasma', axes=plt.gca())
plt.show()

plt.figure(figsize=(8,2))
tripcolor(p, cmap='plasma', axes=plt.gca())
plt.show()
