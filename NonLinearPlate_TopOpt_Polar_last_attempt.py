import numpy as np
import warnings
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, splu
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from numba import njit, prange
import time

# Import Svanberg's MMA subproblem solver and our Numba-compiled kinematics
from mma import mmasub
from polar_kinematics import evaluate_gauss_point

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration (Adapted for Spider Web Compliant Design) ---
NELR       = 30       
NELT_SECTOR = 15      
N_SLICES   = 12       
VOLFRAC    = 0.5     # Increased to budget for the solid catchment rings
PENAL      = 2.0
R_INNER    = 1.0
R_OUTER    = 10.0
THICKNESS  = 1
MAX_ITER   = 150      

# Relaxed strain penalties to allow deep "net-like" deformation
LAMBDA_PEN = 2.0      
EPS_FAIL   = 0.25   
VOID_THRES = 0.05
NU         = 0.3

# Filter Configuration
R_MIN      = 0.3    

# ==========================================
# PRECOMPUTATION & JIT KERNELS
# ==========================================

def precompute_cyclic_filter(nelr, nelt, r_inner, r_outer, n_slices, r_min):
    n_elements = nelr * nelt
    H = np.zeros((n_elements, n_elements))
    
    r_c = np.zeros(n_elements)
    t_c = np.zeros(n_elements)
    dr = (r_outer - r_inner) / nelr
    dt = (2 * np.pi / n_slices) / nelt
    
    for er in range(nelr):
        for et in range(nelt):
            idx = er * nelt + et
            r_c[idx] = r_inner + er * dr + (dr / 2.0)
            t_c[idx] = et * dt + (dt / 2.0)
            
    slice_angle = 2 * np.pi / n_slices
    
    for i in range(n_elements):
        for j in range(n_elements):
            dist_r = r_c[i] - r_c[j]
            dt_raw = np.abs(t_c[i] - t_c[j])
            dist_t = min(dt_raw, slice_angle - dt_raw)
            avg_r = (r_c[i] + r_c[j]) / 2.0
            dist = np.sqrt(dist_r**2 + (avg_r * dist_t)**2)
            
            if dist < r_min:
                H[i, j] = r_min - dist
                
    H_sum = np.sum(H, axis=1)
    H_norm = H / H_sum[:, np.newaxis]
    return H_norm

def precompute_isoparametric_data(elem_coords, n_total):
    G_mats = np.zeros((n_total, 5, 12, 20))
    weights = np.zeros((n_total, 5))
    is_shear = np.array([False, False, False, False, True])
    
    gp2 = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w2 = 1.0
    gp1 = np.array([0.0])
    w1 = 2.0
    
    for i in range(n_total):
        coords = elem_coords[i]
        ip_idx = 0
        
        for xi in gp2:
            for eta in gp2:
                G_mats[i, ip_idx], weights[i, ip_idx] = compute_G_and_weight(xi, eta, coords, w2*w2, shear=False)
                ip_idx += 1
                
        for xi in gp1:
            for eta in gp1:
                G_mats[i, ip_idx], weights[i, ip_idx] = compute_G_and_weight(xi, eta, coords, w1*w1, shear=True)
                ip_idx += 1
                
    return G_mats, weights, is_shear

def compute_G_and_weight(xi, eta, coords, w_base, shear):
    N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
    dNx = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dNe = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    dN = np.vstack((dNx, dNe))
    
    J = dN @ coords
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dN_xy = invJ @ dN
    
    G = np.zeros((12, 20))
    for k in range(4):
        c = 5 * k
        G[0, c], G[1, c] = dN_xy[0, k], dN_xy[1, k]
        G[2, c+1], G[3, c+1] = dN_xy[0, k], dN_xy[1, k]
        G[4, c+2], G[5, c+2] = dN_xy[0, k], dN_xy[1, k]
        G[6, c+3], G[7, c+3] = dN_xy[0, k], dN_xy[1, k]
        G[8, c+4], G[9, c+4] = dN_xy[0, k], dN_xy[1, k]
        G[10, c+3], G[11, c+4] = N[k], N[k]
    
    if shear: G[0:10, :] = 0.0
    else:     G[10:12, :] = 0.0
        
    return G, detJ * w_base

@njit(parallel=True, fastmath=True)
def assemble_system_numba(U_global, x_full, edofMat, G_mats, weights, is_shear, penal, nu, t, void_thres, n_total):
    f_int_all = np.zeros((n_total, 20))
    sK_all = np.zeros((n_total, 400))
    eps_max_arr = np.zeros(n_total)
    deps_dU_arr = np.zeros((n_total, 20))
    dFint_dx_arr = np.zeros((n_total, 20))
    
    E_min = 0.02 
    
    for i in prange(n_total):
        x_p = x_full[i]
        E_val = E_min + (x_p ** penal) * (1.0 - E_min)
        dE_dx = penal * (x_p ** (penal-1)) * (1.0 - E_min)
        
        edofs = edofMat[i]
        
        eff_factor = x_p / void_thres if x_p < void_thres else 1.0
        u_eff = np.zeros(20)
        for d in range(20):
            u_eff[d] = U_global[edofs[d]] * eff_factor
            
        f_int_e = np.zeros(20)
        K_T_e = np.zeros((20, 20))
        deps_du_e = np.zeros(20)
        max_eps = 0.0
        
        for ip in range(5):
            G = G_mats[i, ip]
            w = weights[i, ip]
            
            kin_vec = np.zeros(12)
            for row in range(12):
                for col in range(20):
                    kin_vec[row] += G[row, col] * u_eff[col]
                    
            dW, d2W, eps_p, deps = evaluate_gauss_point(kin_vec, E_val, nu, t)
            
            f_int_e += (G.T @ dW) * w
            K_T_e += (G.T @ d2W @ G) * w
            
            if not is_shear[ip]:
                max_eps = max(max_eps, eps_p)
                deps_du_e += (G.T @ deps) * w

        f_int_all[i] = f_int_e
        
        idx = 0
        for row in range(20):
            for col in range(20):
                sK_all[i, idx] = K_T_e[row, col]
                idx += 1
        
        if x_p >= void_thres:
            eps_max_arr[i] = max_eps
            deps_dU_arr[i] = deps_du_e
            for d in range(20):
                dFint_dx_arr[i, d] = (f_int_e[d] / E_val) * dE_dx
                
    return f_int_all, sK_all, eps_max_arr, deps_dU_arr, dFint_dx_arr

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 75)
    print(f"Bio-Inspired Web TopOpt | Elements: {NELR}x{NELT_SECTOR} ({N_SLICES} Slices)")
    print("=" * 75)
    
    t_start_setup = time.perf_counter()
    
    nnodes_r = NELR + 1
    nnodes_t_sector = NELT_SECTOR + 1 
    n_nodes_raw = nnodes_r * nnodes_t_sector
    n_master = NELR * NELT_SECTOR
    
    # 1. Mesh Generation
    r_vec = np.linspace(R_INNER, R_OUTER, nnodes_r)
    t_vec = np.linspace(0, 2 * np.pi / N_SLICES, nnodes_t_sector)
    R, T = np.meshgrid(r_vec, t_vec, indexing='ij')
    node_x, node_y = R * np.cos(T), R * np.sin(T)
    
    # --- CYCLIC DOF CONDENSATION MAP ---
    dof_map = np.arange(5 * n_nodes_raw)
    for er in range(nnodes_r):
        left_node = er * nnodes_t_sector + 0
        right_node = er * nnodes_t_sector + (nnodes_t_sector - 1)
        for d in range(5):
            dof_map[right_node * 5 + d] = left_node * 5 + d
            
    unique_dofs = np.unique(dof_map)
    
    # --- ELEMENT MESHING ---
    n_elements = NELR * NELT_SECTOR
    edofMat_raw = np.zeros((n_elements, 20), dtype=int)
    elem_coords = np.zeros((n_elements, 4, 2))
    
    for er in range(NELR):
        for et in range(NELT_SECTOR):
            idx = er * NELT_SECTOR + et
            n1 = er * nnodes_t_sector + et
            n2 = er * nnodes_t_sector + (et + 1)
            n3 = (er + 1) * nnodes_t_sector + (et + 1)
            n4 = (er + 1) * nnodes_t_sector + et
            
            for i, n in enumerate([n1, n2, n3, n4]):
                edofMat_raw[idx, i*5:(i+1)*5] = np.arange(n*5, n*5+5)
                
            elem_coords[idx] = np.array([
                [node_x[er, et], node_y[er, et]],
                [node_x[er, et+1], node_y[er, et+1]],
                [node_x[er+1, et+1], node_y[er+1, et+1]],
                [node_x[er+1, et], node_y[er+1, et]]
            ])
            
    edofMat = dof_map[edofMat_raw]
    iK = np.kron(edofMat, np.ones((20,1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1,20), dtype=int)).flatten()
    
    G_mats, weights, is_shear = precompute_isoparametric_data(elem_coords, n_elements)
    H_filter = precompute_cyclic_filter(NELR, NELT_SECTOR, R_INNER, R_OUTER, N_SLICES, R_MIN)
    
    # --- BOUNDARY CONDITIONS ---
    outer_node_left = NELR * nnodes_t_sector + 0
    fixed_dofs_raw = [outer_node_left * 5 + d for d in range(5)]
    fixed_dofs = np.unique(dof_map[fixed_dofs_raw])
    free_dofs = np.setdiff1d(unique_dofs, fixed_dofs)
    
    # --- EXPLICIT GEOMETRIC & LOAD ENFORCEMENT ---
    F_ext_raw = np.zeros(5 * n_nodes_raw)
    
    passive_solid = np.zeros(n_elements, dtype=bool)
    # for er in range(NELR):
    #     passive_solid[er * NELT_SECTOR + 0] = True 

    # Define 6 target radii where objects are likely to be caught
    n_rings = 6
    target_ring_radii = np.linspace(R_INNER + 1.2, R_OUTER - 1.2, n_rings)
    ring_half_width = 0.20  

    for idx in range(n_elements):
        er = idx // NELT_SECTOR
        dr = (R_OUTER - R_INNER) / NELR
        elem_r = R_INNER + er * dr + (dr / 2.0)
        
        for target_r in target_ring_radii:
            if abs(elem_r - target_r) <= ring_half_width:
                edofs = edofMat_raw[idx]
                for corner in range(4):
                    z_dof_idx = edofs[corner * 5 + 2]
                    F_ext_raw[z_dof_idx] += -0.25 
                
                # # Enforce solid material at catchment zones to prevent "void" cheating
                # passive_solid[idx] = True 
                break

    # Condense cyclic boundary conditions
    F_ext = np.zeros(5 * n_nodes_raw)
    np.add.at(F_ext, dof_map, F_ext_raw)

    U_global_raw = np.zeros(5 * n_nodes_raw)
    for er in range(nnodes_r):
        for et in range(nnodes_t_sector):
            node_idx = er * nnodes_t_sector + et
            r_norm = (r_vec[er] - R_INNER) / (R_OUTER - R_INNER)
            U_global_raw[5 * node_idx + 2] = 0.01 * THICKNESS * (1.0 - r_norm**2)
    U_global = U_global_raw.copy()

    # --- Design Space Constraints ---
    x_master = VOLFRAC * np.ones(n_elements)
        
    low = np.ones(n_elements) * 0.001 
    upp = np.ones(n_elements)
    
    low[passive_solid] = 1.0
    x_master[passive_solid] = 1.0

    # --- MMA SETUP ---
    xold1, xold2 = x_master.copy(), x_master.copy()
    m_mma = 1
    amma, cmma, dmma = np.zeros((1, 1)), 1000.0 * np.ones((1, 1)), np.zeros((1, 1))
    
    change = 1.0
    loop = 0
    f0_scale = 1.0
    
    t_end_setup = time.perf_counter()
    print(f"Mesh setup and geometry initialization completed in: {(t_end_setup - t_start_setup)*1000:.2f} ms\n")
    print(f"{'It':<4} | {'Obj(Scale)':<10} | {'Vol':<5} | {'Change':<6} || {'t_Asm':<7} | {'t_Sol':<7} | {'t_Adj':<7} | {'Tot_ms'}")
    print("-" * 85)

    plt.ion() 
    fig_live, ax_live = plt.subplots(figsize=(6, 6))
    
    live_x, live_y, live_triangles = [], [], []
    live_node_offset = 0
    for s in range(N_SLICES):
        angle = s * (2 * np.pi / N_SLICES)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        live_x.extend(node_x.flatten() * cos_a - node_y.flatten() * sin_a)
        live_y.extend(node_x.flatten() * sin_a + node_y.flatten() * cos_a)
        for er in range(NELR):
            for et in range(NELT_SECTOR):
                n1 = er * nnodes_t_sector + et + live_node_offset
                n2 = er * nnodes_t_sector + (et + 1) + live_node_offset
                n3 = (er + 1) * nnodes_t_sector + (et + 1) + live_node_offset
                n4 = (er + 1) * nnodes_t_sector + et + live_node_offset
                live_triangles.extend([[n1, n2, n3], [n1, n3, n4]])
        live_node_offset += n_nodes_raw
        
    live_triangulation = mtri.Triangulation(np.array(live_x), np.array(live_y), np.array(live_triangles))
    
    # --- OPTIMIZATION LOOP ---
    while change > 0.01 and loop < MAX_ITER:
        loop += 1
        t_loop_start = time.perf_counter()
        
        x_phys = H_filter @ x_master
        x_phys[passive_solid] = 1.0 
                
        accum_t_asm = 0.0
        accum_t_sol = 0.0
        total_nr_iters = 0
        
        num_load_steps = 4 
        lu_factorization = None 
        
        # Physics Forward Solve
        for step in range(1, num_load_steps + 1):
            current_F_ext = F_ext * (step / num_load_steps)
            res_norm = 1.0
            nr_iter = 0
            
            while res_norm > 1e-4 and nr_iter < 12:
                nr_iter += 1
                total_nr_iters += 1
                
                t0 = time.perf_counter()
                f_int_all, sK_all, eps_max_arr, deps_dU_arr, dFint_dx_arr = assemble_system_numba(
                    U_global, x_phys, edofMat_raw, G_mats, weights, is_shear, PENAL, NU, THICKNESS, VOID_THRES, n_elements
                )
                F_int = np.zeros(5 * n_nodes_raw)
                np.add.at(F_int, edofMat_raw.flatten(), f_int_all.flatten())
                F_int_condensed = np.zeros_like(F_int)
                np.add.at(F_int_condensed, dof_map, F_int)
                sK = sK_all.flatten()
                accum_t_asm += (time.perf_counter() - t0)
                
                R = current_F_ext - F_int_condensed
                res_norm = np.linalg.norm(R[free_dofs])
                
                if res_norm > 1e-4:
                    t0 = time.perf_counter()
                    K_T = csc_matrix((sK, (iK, jK)), shape=(5 * n_nodes_raw, 5 * n_nodes_raw))
                    K_free = K_T[free_dofs, :][:, free_dofs]
                    
                    lu = splu(K_free)
                    if step == num_load_steps: 
                        lu_factorization = lu 
                        
                    dU_free = lu.solve(R[free_dofs])
                    dU_free = np.clip(dU_free, -0.2 * R_OUTER, 0.2 * R_OUTER)
                    
                    U_global[free_dofs] += dU_free
                    U_global = U_global[dof_map] 
                    accum_t_sol += (time.perf_counter() - t0)

        # --- ADJOINT SENSITIVITIES ---
        t_adj_start = time.perf_counter()
        
        # Maximize Energy Absorption (Minimum Compliance) against Transverse loads
        disp_obj = np.dot(F_ext, U_global) 
        
        pen_viol = np.maximum(0, eps_max_arr - EPS_FAIL)
        penalty_raw = LAMBDA_PEN * np.sum(pen_viol**2)
        
        f0val_raw = disp_obj + penalty_raw
        
        df0_dU = F_ext.copy()
        
        df0_dU_condensed = np.zeros_like(df0_dU)
        np.add.at(df0_dU_condensed, dof_map, df0_dU)
        
        Lambda = np.zeros(5 * n_nodes_raw)
        Lambda[free_dofs] = lu_factorization.solve(-df0_dU_condensed[free_dofs])
        Lambda = Lambda[dof_map] 
        
        df0dx_raw = np.array([np.dot(Lambda[edofMat_raw[i]], dFint_dx_arr[i]) for i in range(n_elements)])
        df0dx_raw[passive_solid] = 0.0 
        
        if loop == 1:
            f0_scale = 10.0 / max(abs(f0val_raw), 1e-10)
            print(f"   [Auto-Scaler] Calibrated objective scale factor to: {f0_scale:.4e}")
            
        f0val = f0val_raw * f0_scale
        df0dx_master = (H_filter.T @ df0dx_raw) * f0_scale

        fval = np.array([[x_phys.mean() - VOLFRAC]])
        df1dx_raw = np.ones(n_elements) / n_elements
        df1dx_raw[passive_solid] = 0.0 
        df1dx_master = H_filter.T @ df1dx_raw
        
        t_adj = (time.perf_counter() - t_adj_start)

        # --- MMA Bounds Update ---
        move_limit = 0.1
        low_mma = np.maximum(low, x_master - move_limit)
        upp_mma = np.minimum(upp, x_master + move_limit)
        low_mma[passive_solid] = 1.0 

        xnew, _, _, _, _, _, _, _, _, _, _ = mmasub(
            m_mma, n_elements, loop,
            x_master[:, np.newaxis], np.zeros((n_master, 1)), np.ones((n_master, 1)),
            xold1[:, np.newaxis], xold2[:, np.newaxis],
            f0val, df0dx_master[:, np.newaxis], fval, df1dx_master[:, np.newaxis].T,
            low_mma[:, np.newaxis], upp_mma[:, np.newaxis], 1.0, amma, cmma, dmma
        )
        
        xold2[:], xold1[:], x_master[:] = xold1, x_master, xnew[:, 0]
        change = np.max(np.abs(x_master - xold1))
        
        t_loop_total = (time.perf_counter() - t_loop_start)
        
        print(f"{loop:<4} | {f0val:<10.5f} | {x_phys.mean():<5.3f} | {change:<6.4f} || "
              f"{accum_t_asm*1000:<7.1f} | {accum_t_sol*1000:<7.1f} | {t_adj*1000:<7.1f} | {t_loop_total*1000:.1f} (NR: {total_nr_iters})")
        
        # --- UPDATE LIVE VIEW ---
        ax_live.clear()
        
        live_densities = []
        for s in range(N_SLICES):
            live_densities.extend(x_phys) 
            live_densities.extend(x_phys) 
            
        ax_live.tripcolor(live_triangulation, facecolors=np.array(live_densities), cmap='Greys', vmin=0, vmax=1)
        ax_live.set_aspect('equal')
        ax_live.set_title(f"Iteration {loop} | Vol: {x_phys.mean():.3f} | Change: {change:.4f}")
        ax_live.axis('off')
        
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()
        time.sleep(0.01) 

    plt.ioff() 
    plt.close(fig_live) 

    # ==========================================
    # FINAL 360° PLOT GENERATION 
    # ==========================================
    print("\nOptimization Complete. Reconstructing full 360° geometry...")
    
    final_densities = H_filter @ x_master
    final_densities[passive_solid] = 1.0 
    
    nodal_densities = np.zeros(n_nodes_raw)
    for er in range(NELR):
        for et in range(NELT_SECTOR):
            elem_idx = er * NELT_SECTOR + et
            for n in [er * nnodes_t_sector + et, er * nnodes_t_sector + (et + 1), 
                      (er + 1) * nnodes_t_sector + (et + 1), (er + 1) * nnodes_t_sector + et]:
                nodal_densities[n] = max(nodal_densities[n], final_densities[elem_idx])
                
    expanded_nodal_densities = nodal_densities

    all_x, all_y, all_z = [], [], []
    all_densities, all_triangles = [], []
    node_offset = 0
    
    U_expanded = U_global[dof_map]
    w_disp_raw = U_expanded[2::5]  
    
    for s in range(N_SLICES):
        angle = s * (2 * np.pi / N_SLICES)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        x_rot = node_x.flatten() * cos_a - node_y.flatten() * sin_a
        y_rot = node_x.flatten() * sin_a + node_y.flatten() * cos_a
        
        all_x.extend(x_rot)
        all_y.extend(y_rot)
        
        all_z.extend(w_disp_raw * (expanded_nodal_densities ** 2))  
        
        for er in range(NELR):
            for et in range(NELT_SECTOR):
                n1 = er * nnodes_t_sector + et + node_offset
                n2 = er * nnodes_t_sector + (et + 1) + node_offset
                n3 = (er + 1) * nnodes_t_sector + (et + 1) + node_offset
                n4 = (er + 1) * nnodes_t_sector + et + node_offset
                
                all_triangles.extend([[n1, n2, n3], [n1, n3, n4]])
                all_densities.extend([final_densities[er * NELT_SECTOR + et], final_densities[er * NELT_SECTOR + et]])
                
        node_offset += n_nodes_raw

    all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)
    all_triangles, all_densities = np.array(all_triangles), np.array(all_densities)
    
    triangulation = mtri.Triangulation(all_x, all_y, all_triangles)
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121)
    ax1.tripcolor(triangulation, facecolors=all_densities, cmap='Greys', vmin=0, vmax=1)
    ax1.set_aspect('equal')
    ax1.set_title(f'Optimized Bio-Inspired Spider Web ({N_SLICES} Slices)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122, projection='3d')
    mag_factor = (R_OUTER * 0.3) / max(np.max(np.abs(all_z)), 1e-9) 
    z_def_scaled = all_z * mag_factor
    
    surf = ax2.plot_trisurf(triangulation, z_def_scaled, cmap='viridis', edgecolor='none')
    ax2.set_title('Reconstructed 3D Compliant Structure', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=-50)
    ax2.set_zlim(-R_OUTER, R_OUTER)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()