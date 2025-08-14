import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit import QuantumCircuit
import time # Để đo thời gian chạy

# ==============================================================================
# PHẦN 1: TẠO/LOAD DỮ LIỆU MỤC TIÊU (TENPY)
# ==============================================================================
# (Hàm này đã ổn định)
pauli_I = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_or_create_target_data(L, J, g):
    filename = f"data/gs_ising_L{L}_J{J}_g{g}.h5"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print(f"File '{filename}' không tồn tại. Đang tạo mới bằng DMRG...")
        model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
        M = tenpy.models.tf_ising.TFIChain(model_params)
        psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)
        dmrg_params = {'mixer': None, 'max_E_err': 1.e-10, 'trunc_params': {'chi_max': 100}}
        info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)
        hdf5_io.save({"psi": psi}, filename)
    psi_mps = hdf5_io.load(filename)["psi"]
    psi_mps.canonical_form()
    target_rdms, target_sv = {}, None
    for i in range(L):
        exp_X = 2*psi_mps.expectation_value("Sx",[i])[0]; exp_Y = 2*psi_mps.expectation_value("Sy",[i])[0]; exp_Z = 2*psi_mps.expectation_value("Sz",[i])[0]
        rdm_i = 0.5 * (pauli_I + exp_X*pauli_X + exp_Y*pauli_Y + exp_Z*pauli_Z)
        target_rdms[(i,)] = np.array(rdm_i, requires_grad=False)
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_sv = full_tensor_array.to_ndarray().flatten()
    target_sv = np.array(target_sv, requires_grad=False)
    return target_rdms, target_sv

# ==============================================================================
# PHẦN 2: CÁC HÀM TIỆN ÍCH
# ==============================================================================
def create_ansatz(params, num_qubits, num_layers):
    for i in range(num_qubits): qml.Hadamard(wires=i)
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i); qml.RY(params[l, i, 1], wires=i); qml.RZ(params[l, i, 2], wires=i)
        for i in range(num_qubits - 1): qml.CNOT(wires=[i, i + 1])
        if num_qubits > 1: qml.CNOT(wires=[num_qubits - 1, 0])

def hybrid_cost_function(params, target_rdms, target_sv, alpha, rdm_qnode, state_qnode):
    circuit_rdms_list = rdm_qnode(params)
    local_loss = 0.0
    for i in range(len(circuit_rdms_list)):
        diff = circuit_rdms_list[i] - target_rdms[(i,)]
        local_loss += qml.math.real(qml.math.trace(qml.math.dot(qml.math.T(qml.math.conj(diff)), diff)))
    circuit_state = state_qnode(params)
    overlap = qml.math.sum(qml.math.conj(target_sv) * circuit_state)
    fidelity_sq = qml.math.abs(overlap)**2
    global_loss = 1.0 - fidelity_sq
    return local_loss + alpha * global_loss

def build_qiskit_circuit(params, num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits); qc.h(range(num_qubits)); qc.barrier()
    for l in range(num_layers):
        for i in range(num_qubits):
            rx, ry, rz = float(params[l,i,0]), float(params[l,i,1]), float(params[l,i,2])
            qc.rx(rx, i); qc.ry(ry, i); qc.rz(rz, i)
        qc.barrier()
        for i in range(num_qubits - 1): qc.cx(i, i + 1)
        if num_qubits > 1: qc.cx(num_qubits - 1, 0)
        qc.barrier()
    return qc

# ==============================================================================
# PHẦN 3: CÁC HÀM VẼ ĐỒ THỊ
# ==============================================================================
def plot_convergence(history, L, g, num_layers, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Optimization Step"); plt.ylabel("Hybrid Cost (log scale)"); plt.yscale('log')
    plt.title(f"Convergence History (L={L}, g={g}, layers={num_layers})")
    plt.grid(True, which="both", ls="--"); plt.savefig(filename); plt.close()

def plot_probabilities(target_sv, final_sv, L, filename):
    probs_target = np.abs(target_sv)**2; probs_circuit = np.abs(final_sv)**2
    plt.figure(figsize=(12, 6));
    basis_states = range(2**L)
    width = 0.8
    plt.bar(basis_states, probs_target, width=width, alpha=0.7, label='Target (MPS)')
    plt.bar(basis_states, probs_circuit, width=width*0.5, alpha=0.9, label='Optimized Circuit')
    plt.xlabel("Basis State"); plt.ylabel("Probability"); plt.title(f"Probability Distribution Comparison (L={L})")
    plt.legend()
    if L > 6: plt.xlim(-0.5, 2**6 - 0.5)
    plt.savefig(filename); plt.close()

def plot_rdm_comparison(target_rdm, final_rdm, site_index, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = [f'Target RDM (Site {site_index})', f'Final RDM (Site {site_index})']
    rdms = [target_rdm, final_rdm]
    for i, ax in enumerate(axes):
        # Vẽ phần thực
        im_real = ax.imshow(np.real(rdms[i]), cmap='viridis', vmin=-1, vmax=1)
        fig.colorbar(im_real, ax=ax)
        ax.set_title(f"{titles[i]} - Real Part")
        ax.set_xticks(range(2)); ax.set_yticks(range(2))
    plt.savefig(filename); plt.close()

# ==============================================================================
# PHẦN 4: CÁC HÀM CHẠY THÍ NGHIỆM
# ==============================================================================
def run_variational_experiment(L, J, g, num_layers, num_steps, step_size, alpha):
    exp_name = f"L{L}_g{g}_layers{num_layers}_alpha{alpha}"
    print("\n" + "#"*80 + f"\n# Variational Experiment: {exp_name} #" + "\n" + "#"*80)

    results_dir = f"results/{exp_name}"; os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    target_rdms, target_sv = get_or_create_target_data(L, J, g)

    dev = qml.device("default.qubit", wires=L)
    @qml.qnode(dev)
    def rdm_circuit(params):
        create_ansatz(params, L, num_layers)
        return [qml.density_matrix(wires=i) for i in range(L)]
    @qml.qnode(dev)
    def state_circuit(params):
        create_ansatz(params, L, num_layers)
        return qml.state()

    param_shape = (num_layers, L, 3)
    params = np.random.normal(0, np.pi, size=param_shape, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=step_size)
    cost_history = []

    for step in range(num_steps):
        params, cost = optimizer.step_and_cost(lambda p: hybrid_cost_function(p, target_rdms, target_sv, alpha, rdm_circuit, state_circuit), params)
        cost_history.append(cost)
        if (step + 1) % 50 == 0: print(f"Step {step+1:4d}: Hybrid Cost = {cost:.8f}")
    
    end_time = time.time()
    
    # Analysis
    final_state = state_circuit(params)
    final_fidelity = np.abs(np.sum(np.conj(target_sv) * final_state))**2
    qiskit_qc = build_qiskit_circuit(params, L, num_layers)
    
    metrics = {
        "method": "Variational (Ours)", "L": L, "g": g, "num_layers": num_layers, "alpha": alpha, "num_steps": num_steps,
        "final_hybrid_cost": float(cost_history[-1]),
        "final_global_fidelity": float(final_fidelity),
        "depth": qiskit_qc.depth(),
        "num_cnots": qiskit_qc.count_ops().get('cx', 0),
        "run_time_seconds": end_time - start_time
    }
    with open(f"{results_dir}/metrics.json", 'w') as f: json.dump(metrics, f, indent=4)
    print("\nFinal Metrics:"); print(json.dumps(metrics, indent=4))

    # Figures
    plot_convergence(cost_history, L, g, num_layers, f"{results_dir}/convergence.png")
    plot_probabilities(target_sv, final_state, L, f"{results_dir}/probabilities.png")
    final_rdms = rdm_circuit(params)
    plot_rdm_comparison(target_rdms[(L//2,)], final_rdms[L//2], L//2, f"{results_dir}/rdm_comparison_center.png")
    qiskit_qc.draw('mpl').savefig(f"{results_dir}/circuit_diagram.png", dpi=600); plt.close()

def run_baseline_experiment(L, J, g):
    exp_name = f"L{L}_g{g}_baseline"
    print("\n" + "#"*80 + f"\n# Baseline Experiment: {exp_name} #" + "\n" + "#"*80)

    results_dir = f"results/{exp_name}"; os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    _, target_sv = get_or_create_target_data(L, J, g)
    
    try:
        dim = 2**L
        unitary_matrix = np.zeros((dim, dim), dtype=complex)
        unitary_matrix[:, 0] = target_sv
        q, _ = np.linalg.qr(unitary_matrix)
        
        qc = QuantumCircuit(L)
        qc.unitary(q, range(L))
        decomposed_qc = qc.decompose()
    except Exception as e:
        print(f"ERROR during baseline construction: {e}"); return
        
    end_time = time.time()

    metrics = {
        "method": "Baseline (Qiskit)", "L": L, "g": g,
        "final_global_fidelity": 1.0,
        "depth": decomposed_qc.depth(),
        "num_cnots": decomposed_qc.count_ops().get('cx', 0),
        "run_time_seconds": end_time - start_time
    }
    with open(f"{results_dir}/metrics.json", 'w') as f: json.dump(metrics, f, indent=4)
    print("\nFinal Metrics:"); print(json.dumps(metrics, indent=4))
    
    decomposed_qc.draw('mpl').savefig(f"{results_dir}/circuit_diagram.png", dpi=600); plt.close()

# ==============================================================================
# PHẦN 5: ĐIỀU KHIỂN CHIẾN DỊCH
# ==============================================================================
def main():
    """
    Định nghĩa và chạy tất cả các thí nghiệm cho bài báo.
    """
    # === CHIẾN DỊCH 1: SO SÁNH HIỆU QUẢ TRÊN CÁC CHẾ ĐỘ VẬT LÝ (L=10) ===
    print("\n\n--- STARTING CAMPAIGN 1: PHYSICAL REGIMES (L=10) ---")
    L_fixed = 10
    g_points = [0.5, 1.0, 1.5]
    for g in g_points:
        run_variational_experiment(L=L_fixed, J=1.0, g=g, num_layers=8, num_steps=1000, step_size=0.05, alpha=0.1)
        run_baseline_experiment(L=L_fixed, J=1.0, g=g)

    # === CHIẾN DỊCH 2: KIỂM TRA KHẢ NĂNG MỞ RỘNG (g=0.8) ===
    print("\n\n--- STARTING CAMPAIGN 2: SCALABILITY TEST (g=0.8) ---")
    g_fixed = 0.8
    L_points = [6, 8, 10] # L=12 sẽ rất lâu, chỉ chạy khi có nhiều thời gian
    for L in L_points:
        # Giảm số lớp/bước cho hệ nhỏ hơn
        num_layers = max(4, 2 * (L // 2 - 1))
        num_steps = max(400, 100 * (L-4))
        run_variational_experiment(L=L, J=1.0, g=g_fixed, num_layers=num_layers, num_steps=num_steps, step_size=0.05, alpha=0.1)
        run_baseline_experiment(L=L, J=1.0, g=g_fixed)

if __name__ == '__main__':
    main()