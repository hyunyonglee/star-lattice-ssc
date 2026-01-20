import numpy as np
import argparse
import logging.config
import os
import os.path
import h5py
from tenpy.tools import hdf5_io
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import logging.config
from StarLatticeSSCModel import StarLatticeSSCModel 
    

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d



def run_dmrg_chirality_model(model_params, dmrg_params):
    
    model = StarLatticeSSCModel(model_params)
    mpo_bond_dims = model.H_MPO.chi
    max_mpo_chi = max(mpo_bond_dims)
    print(f"MPO Max Chi (Hamiltonian): {max_mpo_chi}")
    
    # 초기 상태 (Initial State) 설정
    p_state = ["up", "down"] * (model.lat.N_sites // 2)
    psi = MPS.from_product_state(model.lat.mps_sites(), p_state, bc=model.lat.bc_MPS)

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run() 
    psi.canonical_form() 
    
    return psi, E, model

def measure_chirality(psi, model):
    """
    Measure Scalar Spin Chirality <S_i . (S_j x S_k)> for all triangles in the MPS.
    Formula: i/2 * sum_{cyclic} ( <Sp_i Sm_j Sz_k> - <Sm_i Sp_j Sz_k> )
    """
    print("\n" + "="*50)
    print(">>> Scalar Spin Chirality Measurement <Chi>")
    print("="*50)

    # 1. 3-site term을 계산하는 내부 함수
    def get_chi_value_on_triangle(i, j, k):
        # i, j, k: Global site indices in MPS
        
        # Cyclic permutations (i,j,k), (j,k,i), (k,i,j)
        triplet_indices = [(i,j,k), (j,k,i), (k,i,j)]
        
        val = 0.0 + 0.0j
        prefactor = 2.0 / (np.sqrt(3.0) * 1j) # 8.0 * 1.0 / (4.0j) 
        
        for (idx1, idx2, idx3) in triplet_indices:
            # Term 1: + Sp_1 Sm_2 Sz_3
            ops1 = [('Sp', idx1), ('Sm', idx2), ('Sz', idx3)]
            val += prefactor * psi.expectation_value_term(ops1)
            
            # Term 2: - Sm_1 Sp_2 Sz_3
            ops2 = [('Sm', idx1), ('Sp', idx2), ('Sz', idx3)]
            val += -prefactor * psi.expectation_value_term(ops2)
            
        return val.real # 물리량은 실수여야 함 (허수부는 0에 가까워야 정상)

    # 2. 모든 Unit Cell을 순회하며 측정
    # MPS Unit Cell의 총 사이트 수
    N_sites = model.lat.N_sites
    # Unit Cell 하나당 사이트 수 (Star Lattice = 6)
    sites_per_uc = len(model.lat.unit_cell)
    # MPS 내 Unit Cell 개수
    N_cells = N_sites // sites_per_uc

    print(f"{'Cell':^5} | {'Triangle':^10} | {'Indices':^15} | {'<Chi>':^10}")
    print("-" * 50)

    chi_list = []
    
    for u in range(N_cells):
        base = u * sites_per_uc
        
        # --- Triangle A (Sublattice 0, 1, 2) ---
        idx_A = [base + 0, base + 1, base + 2]
        chi_A = get_chi_value_on_triangle(*idx_A)
        chi_list.append(chi_A)
        print(f"{u:^5} | {'A':^10} | {str(idx_A):^15} | {chi_A:^10.5f}")

        # --- Triangle B (Sublattice 3, 4, 5) ---
        idx_B = [base + 3, base + 5, base + 4]
        chi_B = get_chi_value_on_triangle(*idx_B)
        chi_list.append(chi_B)
        print(f"{u:^5} | {'B':^10} | {str(idx_B):^15} | {chi_B:^10.5f}")

    print("-" * 50)
    return chi_list


def measure_chirality_correlation(psi, model):
    """
    Measure Chirality-Chirality Correlation <Chi_A * Chi_B>
    Uses manual coordinate mapping for robustness.
    """
    print("\n" + "="*60)
    print(">>> Chirality-Chirality Correlation <Chi_A * Chi_B>")
    print("="*60)

    # 1. 단일 삼각형 카이랄리티 연산자 (계수 1/4i)
    def get_chirality_terms(indices):
        cyclic_perms = [
            (indices[0], indices[1], indices[2]),
            (indices[1], indices[2], indices[0]),
            (indices[2], indices[0], indices[1])
        ]
        terms = []
        prefactor = 2.0 / (np.sqrt(3.0) * 1j) # 8.0 * 1.0 / (4.0j) 
        for (idx1, idx2, idx3) in cyclic_perms:
            terms.append( (prefactor, [('Sp', idx1), ('Sm', idx2), ('Sz', idx3)]) )
            terms.append( (-prefactor, [('Sm', idx1), ('Sp', idx2), ('Sz', idx3)]) )
        return terms

    # 2. 상관함수 계산
    def calc_correlation(idxs_A, idxs_B):
        terms_A = get_chirality_terms(idxs_A)
        terms_B = get_chirality_terms(idxs_B)
        total_val = 0.0 + 0.0j
        
        for (coeff_A, ops_A) in terms_A:
            for (coeff_B, ops_B) in terms_B:
                combined_coeff = coeff_A * coeff_B
                combined_ops = ops_A + ops_B
                # 기대값 계산 (psi가 InfiniteMPS인 경우, 인덱스가 범위를 벗어나도 처리됨)
                val = psi.expectation_value_term(combined_ops)
                total_val += combined_coeff * val
        return total_val.real

    # 3. 좌표 매핑 테이블 생성 (Manual Lookup)
    # lat2mps_index 메서드 대신 직접 매핑 딕셔너리를 만듭니다.
    lat = model.lat
    Lx, Ly = lat.Ls
    N_sites = lat.N_sites
    
    # (x, y, u) 좌표를 주면 MPS 인덱스를 반환하는 딕셔너리
    site_map = {}
    for i in range(N_sites):
        # lat.order[i]는 [x, y, u] 형태의 배열
        coords = tuple(lat.order[i]) # 딕셔너리 키로 쓰기 위해 튜플 변환
        site_map[coords] = i

    directions = [
        ("Vertical",  [0, 0]),
        ("Left Leg",  [0, -1]),
        ("Right Leg", [1, 0])
    ]

    print(f"{'Cell (x,y)':^12} | {'Direction':^10} | {'<Chi_A Chi_B>':^15}")
    print("-" * 60)

    chi_corr_list = []
    # 격자 순회
    for x in range(Lx):
        for y in range(Ly):
            
            # --- Triangle A (u=0,1,2) 인덱스 찾기 ---
            # A는 현재 유닛셀 (x, y)에 존재
            try:
                idx_A = [site_map[(x, y, u)] for u in [0, 1, 2]]
            except KeyError:
                print(f"Skipping A at ({x},{y}): Index map error")
                continue

            for name, offset in directions:
                dx, dy = offset
                nx, ny = x + dx, y + dy
                
                # --- Triangle B (u=3,4,5) 인덱스 찾기 ---
                # Infinite MPS와 Periodic BC를 고려한 좌표 변환
                
                # 1. Y 방향 (Cylinder 둘레): Periodic
                real_ny = ny % Ly
                
                # 2. X 방향 (Infinite): Shift 계산
                # iDMRG에서 x가 범위를 벗어나면 MPS 인덱스에 N_sites를 더하거나 뺍니다.
                shift_x = nx // Lx  # x 방향으로 몇 번 유닛셀을 건너뛰었는지 (0, 1, -1 등)
                real_nx = nx % Lx   # 기본 유닛셀(0 ~ Lx-1) 내부의 좌표
                
                # 기본 유닛셀 내에서의 MPS 인덱스를 찾음
                try:
                    base_indices_B = [site_map[(real_nx, real_ny, u)] for u in [3, 5, 4]]
                except KeyError:
                    # 매핑 실패 시 건너뜀 (거의 발생하지 않음)
                    continue
                
                # Infinite MPS 인덱스로 변환 (Shift 적용)
                # 예: 오른쪽 셀로 넘어가면 인덱스 + N_sites
                idx_B = [bi + (shift_x * N_sites) for bi in base_indices_B]

                # 상관함수 계산
                corr = calc_correlation(idx_A, idx_B)
                chi_corr_list.append(corr)
                print(f"({x}, {y}): {name:<10} | {corr:.6f}")

    print("-" * 60)

    return chi_corr_list


def write_data( psi, E, EE, Nup, Ndw, chis, chi_corrs, Lx, Ly, J_chi, path, wavefunc=False ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    if wavefunc:
        data = {"psi": psi}
        with h5py.File(path+"/mps/psi_Lx_%d_Ly_%d_Jchi_%.2f.h5" % (Lx, Ly, J_chi), 'w') as f:
            hdf5_io.save_to_hdf5(f, data)

    Sz = 0.5 * (Nup - Ndw)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Nup = open(path+"/observables/Nup.txt","a", 1)
    file_Ndw = open(path+"/observables/Ndw.txt","a", 1)
    file_Sz = open(path+"/observables/Sz.txt","a", 1)
    file_chis = open(path+"/observables/chis.txt","a", 1)
    file_chi_corrs = open(path+"/observables/chi_corrs.txt","a", 1)
        
    file_EE.write(f"{J_chi} {'  '.join(map(str, EE))}\n")
    file_Nup.write(f"{J_chi} {'  '.join(map(str, Nup))}\n")
    file_Ndw.write(f"{J_chi} {'  '.join(map(str, Ndw))}\n")
    file_Sz.write(f"{J_chi} {'  '.join(map(str, Sz))}\n")
    file_chis.write(f"{J_chi} {'  '.join(map(str, chis))}\n")
    file_chi_corrs.write(f"{J_chi} {'  '.join(map(str, chi_corrs))}\n")

    file_EE.close()
    file_Nup.close()
    file_Ndw.close()
    file_Sz.close()
    file_chis.close()
    file_chi_corrs.close()

    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(f"{J_chi} {E} {np.max(EE)} {np.mean(Sz)} {np.mean(chis)} {np.mean(chi_corrs)} \n")
    file.close()


# --- 실행 예시 ---
if __name__ == "__main__":

    current_directory = os.getcwd()

    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    # parser for command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--Lx", default='1', help="Length of cylinder")
    parser.add_argument("--Ly", default='2', help="Circumference of cylinder")
    parser.add_argument("--J_chi", default='100.0', help=" Chirality coupling")
    parser.add_argument("--J_chi0", default='0.0', help=" On-site chirality field")
    parser.add_argument("--chi", default='100', help="Bond dimension")
    parser.add_argument("--max_sweep", default='50', help="Maximum number of sweeps")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument("--wavefunc", action='store_true', help="Save wavefunction")
    args=parser.parse_args()

    # parameters
    Lx = int(args.Lx)
    Ly = int(args.Ly)
    J_chi = float(args.J_chi)
    J_chi0 = float(args.J_chi0)
    chi = int(args.chi)
    max_sweep = int(args.max_sweep)
    path = args.path
    
    model_params = {
        'Lx': Lx,                
        'Ly': Ly,                
        'bc': 'periodic',  #  ['open', 'periodic'],  # 
        'bc_MPS': 'infinite', # 'finite',  # 
        'conserve': 'N,Sz',     
        
        # 물리적 파라미터
        't_intra': 1.0,
        't_inter': 1.0,
        't3': 0.25, 
        'J_chi': J_chi,
        'J_chi0': J_chi0
    }

    dmrg_params = {
        'mixer': True,                # t가 작으므로 필수
        'mixer_params': {
            'amplitude': 1.e-3,       
            'decay': 2.0,             
            'disable_after': 10       
        },
        'trunc_params': {
            'chi_max': chi,           
            'svd_min': 1.e-10
        },
        'chi_list': { 0: 16, 5: 32, 10: 64, 15: 150, 25: chi },
        'max_E_err': 1.e-8,           
        'max_sweeps': max_sweep,             
        'combine': True                 
    }

    # 1. DMRG 실행 (이전 단계)
    psi, E, model = run_dmrg_chirality_model(model_params, dmrg_params)

    # 2. 카이랄리티 측정 (psi, model이 메모리에 있다고 가정)
    chis = measure_chirality(psi, model)
    print(chis)

    chi_corrs = measure_chirality_correlation(psi, model)
    print(chi_corrs)

    EE = psi.entanglement_entropy()
    Nup = psi.expectation_value("Nu")
    Ndw = psi.expectation_value("Nd")

    write_data( psi, E, EE, Nup, Ndw, chis, chi_corrs, Lx, Ly, J_chi, path, wavefunc=args.wavefunc )