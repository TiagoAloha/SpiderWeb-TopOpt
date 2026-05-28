import sympy as sp
import time

def generate_nonlinear_element():
    print("Initializing Symbolic Variables...")
    
    # Material and geometric parameters
    E, nu, t = sp.symbols('E nu t', real=True, positive=True)
    a, b = sp.symbols('a b', real=True, positive=True) # Half-width, half-height of element
    
    # 20 Degrees of Freedom for a 4-node element (u, v, w, thx, thy per node)
    u_dofs = sp.Matrix(sp.sympify("u1:5, v1:5, w1:5, thx1:5, thy1:5"))
    
    # Natural coordinates
    xi, eta = sp.symbols('xi eta', real=True)
    
    # Shape functions for 4-node quadrilateral
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    N = sp.Matrix([N1, N2, N3, N4])
    
    # Map DOFs to continuous fields
    u   = (N.T * u_dofs[0:4])[0]
    v   = (N.T * u_dofs[4:8])[0]
    w   = (N.T * u_dofs[8:12])[0]
    thx = (N.T * u_dofs[12:16])[0]
    thy = (N.T * u_dofs[16:20])[0]
    
    # Jacobian of transformation (assuming rectangular element size 2a x 2b)
    dx_dxi, dy_deta = a, b
    
    print("Constructing von Karman Kinematics...")
    # Derivatives wrt physical coordinates x, y
    u_x = sp.diff(u, xi) / dx_dxi
    u_y = sp.diff(u, eta) / dy_deta
    v_x = sp.diff(v, xi) / dx_dxi
    v_y = sp.diff(v, eta) / dy_deta
    w_x = sp.diff(w, xi) / dx_dxi
    w_y = sp.diff(w, eta) / dy_deta
    
    # 1. Membrane Strains (with nonlinear von Karman terms)
    eps_m = sp.Matrix([
        u_x + 0.5 * w_x**2,
        v_y + 0.5 * w_y**2,
        u_y + v_x + w_x * w_y
    ])
    
    # 2. Bending Strains (Curvatures)
    eps_b = sp.Matrix([
        sp.diff(thy, xi) / dx_dxi,
        -sp.diff(thx, eta) / dy_deta,
        sp.diff(thy, eta) / dy_deta - sp.diff(thx, xi) / dx_dxi
    ])
    
    # 3. Shear Strains (Mindlin)
    eps_s = sp.Matrix([
        w_x + thy,
        w_y - thx
    ])
    
    # Constitutive Matrices (Plane Stress)
    factor_m = E * t / (1 - nu**2)
    D_m = factor_m * sp.Matrix([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    factor_b = E * t**3 / (12 * (1 - nu**2))
    D_b = factor_b * sp.Matrix([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    # Shear correction factor k = 5/6
    factor_s = 5.0/6.0 * E * t / (2 * (1 + nu))
    D_s = factor_s * sp.Matrix([
        [1, 0],
        [0, 1]
    ])
    
    print("Integrating Strain Energy...")
    # Strain Energy Density
    W = 0.5 * (eps_m.T * D_m * eps_m + eps_b.T * D_b * eps_b + eps_s.T * D_s * eps_s)[0]
    
    # Gauss Quadrature (2x2 for membrane/bending, 1x1 for shear to prevent locking)
    # For simplicity in this script, we apply 1-point integration to the entire energy field. 
    # (In a production environment, split the integration points to avoid spurious zero-energy modes).
    U_total = W.subs({xi: 0, eta: 0}) * (4 * a * b) 
    
    print("Deriving Internal Forces and Tangent Stiffness (This may take a minute)...")
    t0 = time.time()
    
    # Internal Force Vector (20x1)
    f_int = sp.Matrix([sp.diff(U_total, dof) for dof in u_dofs])
    
    # Tangent Stiffness Matrix (20x20)
    K_T = f_int.jacobian(u_dofs)
    
    # Maximum Principal Strain for failure penalty
    eps_x, eps_y, gam_xy = eps_m[0], eps_m[1], eps_m[2]
    eps_principal = (eps_x + eps_y)/2 + sp.sqrt(((eps_x - eps_y)/2)**2 + (gam_xy/2)**2)
    deps_du = sp.Matrix([sp.diff(eps_principal, dof) for dof in u_dofs])
    
    print(f"Derivations complete in {time.time() - t0:.2f} seconds.")
    print("Exporting to generated_routines.py...")
    
    # Export using lambdify or direct string generation
    # To avoid Sympy overhead during the optimization loop, we write explicit Python code.
    with open("generated_routines.py", "w") as f:
        f.write("import numpy as np\n\n")
        f.write("def evaluate_element(u, E_val, nu_val, t_val, a_val, b_val):\n")
        f.write("    # Unpack DOFs\n")
        for i in range(20):
            f.write(f"    {u_dofs[i]} = u[{i}]\n")
        f.write("    E = E_val\n    nu = nu_val\n    t = t_val\n    a = a_val\n    b = b_val\n\n")
        
        # Write f_int
        f.write("    f_int = np.zeros(20)\n")
        for i in range(20):
            if f_int[i] != 0:
                f.write(f"    f_int[{i}] = {sp.pycode(f_int[i])}\n")
                
        # Write K_T
        f.write("\n    K_T = np.zeros((20, 20))\n")
        for i in range(20):
            for j in range(20):
                if K_T[i,j] != 0:
                    f.write(f"    K_T[{i},{j}] = {sp.pycode(K_T[i,j])}\n")
                    
        # Write Strain and its derivative
        f.write(f"\n    eps_principal = {sp.pycode(eps_principal)}\n")
        f.write("    deps_du = np.zeros(20)\n")
        for i in range(20):
            if deps_du[i] != 0:
                f.write(f"    deps_du[{i}] = {sp.pycode(deps_du[i])}\n")
                
        f.write("\n    return f_int, K_T, eps_principal, deps_du\n")

if __name__ == "__main__":
    generate_nonlinear_element()