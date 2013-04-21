from numpy import meshgrid, linspace
from scipy import rand, pi
from scipy.linalg import norm
from pyamg import *
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import poisson

n=1000
A = poisson((n,n),format='csr')         
b = rand(A.shape[0])                    

ml = smoothed_aggregation_solver(A)     

res1 = []
x = ml.solve(b, tol=1e-12, residuals=res1)
print norm(b-A*x)

