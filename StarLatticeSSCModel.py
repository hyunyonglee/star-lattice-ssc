import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models import lattice

class StarLatticeSSCModel(CouplingMPOModel):
    """
    Star Lattice Model with 3rd Nearest Neighbor Hopping.
    (Basis Re-defined: a1 points to bottom-right, a2 points to top-right)
    """
    default_lattice = "StarLattice"

    def init_sites(self, model_params):
        conserve_str = model_params.get('conserve', 'N,Sz')
        cons_N = 'N' if 'N' in conserve_str else None
        cons_Sz = 'Sz' if 'Sz' in conserve_str else None
        return SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)

    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 2)
        Ly = model_params.get('Ly', 2)
        bc = model_params.get('bc', 'periodic')
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        order = model_params.get('order', 'default')
        
        r = 0.25 
        d = 0.5   
        dist_centers = 2*r + d 
        
        # Basis Vectors
        basis = np.array([
            [np.sqrt(3)/2 * dist_centers, -1.5 * dist_centers], # a1: Bottom-Right
            [np.sqrt(3)/2 * dist_centers,  1.5 * dist_centers]  # a2: Top-Right
        ])

        cA = np.array([0.0, 0.0])
        cB = np.array([0.0, dist_centers])

        # Sublattice Indices: A(0,1,2), B(3,4,5)
        angles_A = np.array([210, 330, 90]) * (np.pi / 180)
        angles_B = np.array([270, 150, 30]) * (np.pi / 180)

        pos_A = [cA + r * np.array([np.cos(a), np.sin(a)]) for a in angles_A]
        pos_B = [cB + r * np.array([np.cos(a), np.sin(a)]) for a in angles_B]
        
        pos = np.array(pos_A + pos_B)
        pos += np.array([basis[0][0]*0.5, basis[1][1]*0.5])

        # 1. Intra-triangle bonds (t_intra)
        nn_intra = [
            (0, 1, [0, 0]), (1, 2, [0, 0]), (2, 0, [0, 0]), 
            (3, 4, [0, 0]), (4, 5, [0, 0]), (5, 3, [0, 0])
        ]
        
        # 2. Inter-triangle bonds (t_inter)
        nn_inter = [
            (2, 3, [0, 0]),   # Vertical
            (0, 5, [0, -1]),  # Left Leg (-a2 direction)
            (1, 4, [1, 0])    # Right Leg (+a1 direction)
        ]

        # 3. Third Nearest Neighbor bonds (t3) - [NEW]
        # Inter 결합이 있는 두 삼각형 사이의 "나머지 평행한 꼭짓점"들을 연결
        nn_t3 = [
            # (A) Vertical Interface [0, 0] (Main bond: 2-3)
            (0, 4, [0, 0]),   # A_Left - B_Left
            (1, 5, [0, 0]),   # A_Right - B_Right
            
            # (B) Left Leg Interface [0, -1] (Main bond: 0-5)
            (2, 4, [0, -1]),  # A_Top - B_TopLeft
            (1, 3, [0, -1]),  # A_Right - B_Bottom
            
            # (C) Right Leg Interface [1, 0] (Main bond: 1-4)
            (2, 5, [1, 0]),   # A_Top - B_TopRight
            (0, 3, [1, 0])    # A_Left - B_Bottom
        ]

        lat = lattice.Lattice(
            Ls=[Lx, Ly],
            unit_cell=[self.init_sites(model_params)] * 6,
            basis=basis,
            positions=pos,
            pairs={
                'nearest_neighbors': nn_intra + nn_inter, 
                'intra': nn_intra, 
                'inter': nn_inter,
                't3': nn_t3  # t3 결합 등록
            },
            bc=bc, bc_MPS=bc_MPS, order=order
        )
        return lat

    def init_terms(self, model_params):
        t_intra = model_params.get('t_intra', 1.0)
        t_inter = model_params.get('t_inter', 1.0)
        t3 = model_params.get('t3', 0.0)  # [NEW] t3 hopping parameter
        J_chi = model_params.get('J_chi', 0.0) 
        J_chi0 = model_params.get('J_chi0', 0.0)
        J_inter = model_params.get('J_inter', 0.0)
        
        # Intra Hopping
        for u1, u2, dx in self.lat.pairs['intra']:
            self.add_coupling(-t_intra, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t_intra, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        
        # Inter Hopping
        for u1, u2, dx in self.lat.pairs['inter']:
            self.add_coupling(-t_inter, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t_inter, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            
        # 3rd Nearest Neighbor Hopping
        if t3 != 0:
            for u1, u2, dx in self.lat.pairs['t3']:
                self.add_coupling(-t3, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(-t3, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        
        # Chirality Interaction
        if J_chi != 0:
            target_offsets = [[0, 0], [0, -1], [1, 0]]
            for offset in target_offsets:
                self.add_chirality_interaction_sz_conserving(-J_chi, offset)

        # On-site Chirality Field
        if J_chi0 != 0:
            self.add_chirality_field(-J_chi0)

        # H = J_inter * (Sz_i Sz_j + 0.5 * (Sp_i Sm_j + Sm_i Sp_j))
        if J_inter != 0:
            for u1, u2, dx in self.lat.pairs['inter']:
                # Sz * Sz
                self.add_coupling(J_inter, u1, 'Sz', u2, 'Sz', dx)
                # 0.5 * (Sp * Sm + h.c.)
                self.add_coupling(J_inter * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)


    def add_chirality_field(self, strength):
        """
        Adds H += strength * sum_{triangles} Chi_triangle
        Triangle A indices: (0, 1, 2)
        Triangle B indices: (3, 5, 4) (Note the order for correct chirality definition)
        """
        # 측정 함수(measure_chirality)와 동일한 prefactor 사용
        prefactor = 2.0 / (np.sqrt(3.0) * 1j)

        # 3-site 항을 생성하는 내부 함수
        def get_single_chirality_terms(indices):
            cyclic_perms = [
                (indices[0], indices[1], indices[2]),
                (indices[1], indices[2], indices[0]),
                (indices[2], indices[0], indices[1])
            ]
            terms = []
            for (idx1, idx2, idx3) in cyclic_perms:
                # Term 1: + prefactor * Sp_1 Sm_2 Sz_3
                terms.append( (prefactor, ['Sp', 'Sm', 'Sz'], [idx1, idx2, idx3]) )
                # Term 2: - prefactor * Sm_1 Sp_2 Sz_3
                terms.append( (-prefactor, ['Sm', 'Sp', 'Sz'], [idx1, idx2, idx3]) )
            return terms

        # 1. Triangle A (0, 1, 2)
        for coeff, ops, idxs in get_single_chirality_terms([0, 1, 2]):
            total_coeff = strength * coeff
            multi_coupling_args = []
            for op, site_idx in zip(ops, idxs):
                # 같은 Unit cell 내의 상호작용이므로 dx는 [0, 0]
                multi_coupling_args.append((op, [0, 0], site_idx))
            self.add_multi_coupling(total_coeff, multi_coupling_args)

        # 2. Triangle B (3, 5, 4) 
        # 주의: measure_chirality 함수와 일관성을 유지하기 위해 인덱스 순서(3, 5, 4) 사용
        for coeff, ops, idxs in get_single_chirality_terms([3, 5, 4]):
            total_coeff = strength * coeff
            multi_coupling_args = []
            for op, site_idx in zip(ops, idxs):
                multi_coupling_args.append((op, [0, 0], site_idx))
            self.add_multi_coupling(total_coeff, multi_coupling_args)


    def add_chirality_interaction_sz_conserving(self, strength, offset_B):
        """
        Adds V = strength * Chi_A(R) * Chi_B(R + offset_B)
        """
        def get_chirality_terms(indices):
            cyclic_perms = [
                (indices[0], indices[1], indices[2]),
                (indices[1], indices[2], indices[0]),
                (indices[2], indices[0], indices[1])
            ]
            terms = []
            prefactor = 2.0 / (np.sqrt(3.0) * 1j) 
        
            for (idx1, idx2, idx3) in cyclic_perms:
                terms.append( (prefactor, ['Sp', 'Sm', 'Sz'], [idx1, idx2, idx3]) )
                terms.append( (-prefactor, ['Sm', 'Sp', 'Sz'], [idx1, idx2, idx3]) )
            return terms

        chi_A_terms = get_chirality_terms([0, 1, 2])
        chi_B_terms = get_chirality_terms([3, 5, 4]) 

        for coeff_A, ops_A, idxs_A in chi_A_terms:
            for coeff_B, ops_B, idxs_B in chi_B_terms:
                total_coeff = strength * coeff_A * coeff_B
                multi_coupling_args = []
                for op, site_idx in zip(ops_A, idxs_A):
                    multi_coupling_args.append((op, [0, 0], site_idx))
                for op, site_idx in zip(ops_B, idxs_B):
                    multi_coupling_args.append((op, offset_B, site_idx))
                self.add_multi_coupling(total_coeff, multi_coupling_args)



# --- Plotting lattice ---
def plot_star_lattice_final():
    model_params = {
        'Lx': 2, 'Ly': 2,
        'bc': ['open', 'periodic'], 
        'bc_MPS': 'finite',
        'conserve': 'N,Sz', 
        'order': 'default',
        't3': 1.0 # 시각화를 위해 t3 활성화
    }

    model = StarLatticeSSCModel(model_params)
    lat = model.lat

    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Grid & Sites
    lat.plot_sites(ax, color='black', marker='o', markersize=13, zorder=10) 
    
    # 2. Bonds Plotting
    lat.plot_coupling(ax, lat.pairs['intra'], color='blue', linewidth=3.0, label='Intra ($t_{intra}$)')
    lat.plot_coupling(ax, lat.pairs['inter'], color='red', linestyle='--', linewidth=2.0, label='Inter ($t_{inter}$)')
    # [NEW] t3 Plotting (녹색 점선)
    lat.plot_coupling(ax, lat.pairs['t3'], color='green', linestyle=':', linewidth=2.0, alpha=0.8, label='3rd NN ($t_3$)')

    # 3. MPS Path
    lat.plot_order(ax, linestyle='-', color='gray', linewidth=4, alpha=0.4, zorder=5)

    # 4. Text Labels
    all_positions = lat.position(lat.order) 
    for i in range(lat.N_sites):
        x, y = all_positions[i]
        lattice_idx = lat.order[i] 
        u = lattice_idx[-1] 
        ax.text(x, y, str(u), color='white', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=15)

    # 5. Basis Vectors Arrow
    origin = all_positions[0] 
    a1, a2 = lat.basis[0], lat.basis[1]
    def plot_arrow(vector, label, color='purple'):
        ax.annotate("", xy=(origin[0] + vector[0], origin[1] + vector[1]), xytext=(origin[0], origin[1]),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=3, headwidth=12, headlength=12, alpha=0.9), zorder=20)
        ax.text(origin[0] + vector[0]*0.6, origin[1] + vector[1]*0.6 - 0.2, label, 
                color=color, fontsize=16, fontweight='bold', zorder=21)

    plot_arrow(a1, r"$\vec{a}_1$")
    plot_arrow(a2, r"$\vec{a}_2$")

    ax.set_aspect('equal')
    ax.set_title(f"Star Lattice with 3rd NN Hopping", fontsize=18)
    
    # Custom Legend
    custom_lines = [
        Line2D([0], [0], color='blue', lw=3),
        Line2D([0], [0], color='red', lw=2, linestyle='--'),
        Line2D([0], [0], color='green', lw=2, linestyle=':'),
        Line2D([0], [0], color='purple', lw=2, label='Basis'),
        Line2D([0], [0], color='gray', lw=4, alpha=0.4, label='MPS Path')
    ]
    ax.legend(custom_lines, ['Intra', 'Inter', '3rd NN (t3)', 'Basis', 'MPS Path'], loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()
