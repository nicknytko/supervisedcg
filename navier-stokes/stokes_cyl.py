# Cylinder flow problem
# Adapted from https://www.firedrakeproject.org/demos/navier_stokes.py.html

from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

M = Mesh('cylflow.msh')
V = VectorFunctionSpace(M, "CG", 2) # Velocity
W = FunctionSpace(M, "CG", 1) # Pressure
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(100.0)
F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), 2), # Inlet velocity = (1,0)
#        DirichletBC(Z.sub(1), Constant(0), 3),      # Outlet pressure = 0
        DirichletBC(Z.sub(0), Constant((0, 0)), [4,5])] # No-slip walls

solver_params = {
    "ksp_type": "gmres",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
solve(F==0, up, bcs=bcs, solver_parameters=solver_params)

u, p = up.split()

plt.figure(figsize=(8,2))
plt.title('Navier-Stokes')
tripcolor(u, cmap='plasma', axes=plt.gca())

wind = u

# Convection-Diffusion code from Scott

p = 1
V = FunctionSpace(M, 'CG', p)

x, y = SpatialCoordinate(M)

scale = Constant(5.0)
shift = Constant(0.25)
f = exp(-scale * ((x-shift) * (x-shift)+(y*y)))

u = TrialFunction(V)
v = TestFunction(V)

dirichlet_bcs = [
    DirichletBC(V, Constant(1.0), [2,5]), # Heat inlet and cylinder
    DirichletBC(V, Constant(0.0), [3,4]), # Unheated everything else
]

k = Constant(0.1)
b = as_vector(wind)

A = k * inner(grad(v), grad(u))*dx + dot(b, grad(u)) * v * dx
L = f * v * dx

m_k = 1.0/3.0
h_k = sqrt(2) * CellVolume(M) / CellDiameter(M)
b_norm = sqrt(dot(b, b))
Pe_k = m_k * b_norm * h_k / (2.0 * k)
eps_k = conditional(gt(Pe_k, Constant(1.0)), Constant(1.0), Pe_k)
tau_k = h_k / (2.0 * b_norm) * eps_k

A += inner((dot(b, grad(u)) - k*div(grad(u))), tau_k * dot(b, grad(v))) * dx
L += f * tau_k * dot(b, grad(v)) * dx

u_sol = Function(V)
solver = LinearVariationalSolver(LinearVariationalProblem(A, L, u_sol, dirichlet_bcs))
solver.solve()

plt.figure(figsize=(8,2))
plt.title('Convection-Diffusion')
tripcolor(u_sol, cmap='plasma', axes=plt.gca())
plt.show()

# save matrix and rhs
Amat = assemble(A, bcs=dirichlet_bcs)
indptr, indices, data = Amat.petscmat.getValuesCSR()
Asp = sp.csr_matrix((data, indices, indptr))
with assemble(L).dat.vec_ro as v:
    b = v.array
sio.savemat('cylflow.mat', {
    'A': Asp,
    'b': b
})
