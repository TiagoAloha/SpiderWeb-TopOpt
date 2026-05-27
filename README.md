# SpiderWeb-TopOpt
A Topology Optimization algorithm for a bio-inspired compliant web-structure made with FFF. 

Algorithmic Specification: Nonlinear Compliant Plate Topology Optimization
1. Project Overview
This document outlines the numerical framework for designing a 2D plate-based structure capable of large out-of-plane displacements (hinging) for load transfer, subject to a localized failure strain limit. The algorithm transitions from standard linear topology optimization to geometrically nonlinear analysis using the von Kármán strain formulation.
2. Mathematical Formulation
2.1 Domain & Degrees of Freedom
Domain: 2D grid of Mindlin-Reissner plate elements.
DOFs:  per node.
Design Variable: Density field , mapped to physical density  via density filter and Heaviside projection.
2.2 Objective Function (Penalized Flexibility)
The objective is to maximize out-of-plane displacement (hinging) while penalizing strains exceeding a failure threshold ():
Where:
 is the target load vector.
 is the converged displacement vector.
 is the penalty weight.
 is the maximum principal von Kármán strain.
2.3 Constraints
Volume: 
Bounds: 
3. Nonlinear Finite Element Analysis (FEA)
Because of the geometric nonlinearity, the internal force vector  is non-constant. We solve for equilibrium  using the Newton-Raphson method:
Assembly: Construct Tangent Stiffness Matrix .
Linear Solve: Compute incremental displacement .
Update:  until convergence.
4. Adjoint Sensitivity Analysis
To compute gradients for the MMA optimizer, we solve the adjoint equation using the converged tangent stiffness matrix:
The sensitivity of the objective with respect to physical density  is:
5. Implementation Roadmap
Nonlinear Kernel: Develop element-level routines for internal forces and tangent stiffness matrices incorporating the  term.
Newton-Raphson Solver: Implement an iterative solver. Use pypardiso (as found in your current scripts) to manage the global linear system solve at each iteration.
Stress/Strain Relaxation: Implement element-level scaling for  to prevent singularities in the  matrix.
Sensitivity Chain: Integrate the derivative of the penalty term into your existing sensitivity assembly routines.
MMA Integration: Update your main loop to handle the converged nonlinear displacements and pass the aggregated gradients to mma.py.
Note: This specification assumes a 2.5D Mindlin plate approach to capture bending/membrane coupling efficiently without the computational overhead of full 3D solid elements.
