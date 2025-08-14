import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit import QuantumCircuit

# ==============================================================================
# PHẦN 1: TẠO/LOAD DỮ LIỆU MỤC TIÊU (TENPY)
# ==============================================================================
pauli_I = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_or_create_target_data(L, J, g):
    """
    Tạo hoặc đọc file MPS. Trả về cả RDM 1-site và statevector đầy đủ.
    """
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
        print("Đã tạo và lưu file MPS mới.")

    print(f"Đang đọc MPS và tính toán dữ liệu mục tiêu từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
    psi_mps.canonical_form()
    
    # Tính RDM 1-site
    target_rdms = {}
    for i in range(L):
        exp_X = 2 * psi_mps.expectation_value("Sx", [i])[0]
        exp_Y = 2 * psi_mps.expectation_value("Sy", [i])[0]
        exp_Z = 2 * psi_mps.expectation_value("Sz", [i])[0]
        rdm_i = 0.5 * (pauli_I + exp_X*pauli_X + exp_Y*pauli_Y + exp_Z*pauli_Z)
        target_rdms[(i,)] = np.array(rdm_i, requires_grad=False)

    # Lấy statevector
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_sv = full_tensor_array.to_ndarray().flatten()
    target_sv = np.array(target_sv, requires_grad=False)
    
    print("Đã tính toán xong RDM 1-site và statevector.")
    return target_rdms, target_sv

# ==============================================================================
# PHẦN 2: CÁC HÀM TIỆN ÍCH CHO MẠCH VÀ TÍNH TOÁN
# ==============================================================================
def create_ansatz(params, num_qubits, num_layers):
    """Định nghĩa cấu trúc mạch."""
    for i in range(num_qubits): qml.Hadamard(wires=i)
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i); qml.RY(params[l, i, 1], wires=i); qml.RZ(params[l, i, 2], wires=i)
        for i in range(num_qubits - 1): qml.CNOT(wires=[i, i + 1])
        if num_qubits > 1: qml.CNOT(wires=[num_qubits - 1, 0])

def hybrid_cost_function(params, target_rdms, target_sv, alpha, rdm_qnode, state_qnode):
    """Hàm mất mát lai kết hợp local RDM và global fidelity."""
    # Loss cục bộ (phần này đã hoạt động tốt)
    circuit_rdms_list = rdm_qnode(params)
    local_loss = 0.0
    num_qubits = len(circuit_rdms_list)
    for i in range(num_qubits):
        diff = circuit_rdms_list[i] - target_rdms[(i,)]
        diff_dagger = qml.math.T(qml.math.conj(diff))
        local_loss += qml.math.real(qml.math.trace(qml.math.dot(diff_dagger, diff)))
        
    # Loss toàn cục
    circuit_state = state_qnode(params)
    
    # --- THAY ĐỔI QUYẾT ĐỊNH ---
    # Thay thế qml.math.vdot bằng phép tính thủ công để tránh lỗi VJP
    overlap = qml.math.sum(qml.math.conj(target_sv) * circuit_state)
    fidelity_sq = qml.math.abs(overlap)**2
    
    global_loss = 1.0 - fidelity_sq
    
    # Kết hợp
    return local_loss + alpha * global_loss

def build_qiskit_circuit_for_analysis(params, num_qubits, num_layers):
    """Xây dựng mạch Qiskit tương đương để phân tích."""
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
# PHẦN 3: HÀM THÍ NGHIỆM CHÍNH
# ==============================================================================
def run_experiment(L, J, g, num_layers, num_steps, step_size, alpha=0.1):
    print("\n" + "#"*80)
    print(f"# Bắt đầu Thí nghiệm: L={L}, J={J}, g={g}, layers={num_layers}, steps={num_steps}, alpha={alpha} #")
    print("#"*80)

    results_dir = f"results/L{L}_g{g}_layers{num_layers}_alpha{alpha}"
    os.makedirs(results_dir, exist_ok=True)
    
    target_rdms_dict, target_sv = get_or_create_target_data(L, J, g)

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

    print("\nBắt đầu quá trình tối ưu hóa với HÀM MẤT MÁT HYBRID...")
    for step in range(num_steps):
        params, cost = optimizer.step_and_cost(
            lambda p: hybrid_cost_function(p, target_rdms_dict, target_sv, alpha, rdm_circuit, state_circuit), params
        )
        cost_history.append(cost)
        if (step + 1) % 20 == 0:
            print(f"Bước {step+1:4d}:  Hybrid Cost = {cost:.8f}")
    final_cost = cost_history[-1]

    print("\nĐang phân tích và lưu kết quả...")
    final_state_circuit = state_circuit(params)
    
    # Tính lại fidelity bằng numpy tiêu chuẩn cho báo cáo cuối cùng
    final_overlap = np.sum(np.conj(target_sv) * final_state_circuit)
    final_fidelity = np.abs(final_overlap)**2

    analysis_qc = build_qiskit_circuit_for_analysis(params, L, num_layers)
    depth = analysis_qc.depth()
    ops_count_dict = analysis_qc.count_ops()
    num_cnots = ops_count_dict.get('cx', 0)
    total_gates = sum(ops_count_dict.values())

    metrics = {
        "L": L, "g": g, "num_layers": num_layers, "alpha": alpha,
        "final_hybrid_cost": float(final_cost),
        "final_global_fidelity": float(final_fidelity),
        "depth": depth,
        "total_gates": total_gates,
        "num_cnots": num_cnots,
        "ops_count": {k: v for k, v in ops_count_dict.items()}
    }
    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Các chỉ số đã được lưu vào 'metrics.json':")
    print(json.dumps(metrics, indent=4))

    # Vẽ và lưu figures
    plt.figure(figsize=(10, 6)); plt.plot(range(1, num_steps + 1), cost_history)
    plt.xlabel("Bước tối ưu hóa"); plt.ylabel("Hybrid Cost (log scale)"); plt.yscale('log')
    plt.title(f"Lịch sử hội tụ (Hybrid Loss) cho L={L}, g={g}, {num_layers} lớp")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{results_dir}/convergence_plot.png"); plt.close()
    
    probs_target = np.abs(target_sv)**2; probs_circuit = np.abs(final_state_circuit)**2
    plt.figure(figsize=(12, 6)); plt.bar(range(2**L), probs_target, alpha=0.7, label='Trạng thái Mục tiêu (MPS)')
    plt.bar(range(2**L), probs_circuit, alpha=0.7, label='Trạng thái Mạch (Tối ưu)', width=0.5)
    plt.xlabel("Trạng thái cơ sở (basis state)"); plt.ylabel("Xác suất"); plt.title(f"So sánh phân bố xác suất (L={L}, g={g})")
    plt.legend()
    if L > 6: plt.xlim(-0.5, 2**6 - 0.5)
    plt.savefig(f"{results_dir}/probability_comparison.png"); plt.close()

    print(f"Đã lưu tất cả kết quả và figures vào thư mục: '{results_dir}'")
    print("#"*80)

# ==============================================================================
# PHẦN 4: ĐIỀU KHIỂN CÁC THÍ NGHIỆM
# ==============================================================================
if __name__ == '__main__':
        run_experiment(
        L=10, 
        J=1.0, 
        g=0.5, 
        num_layers=8,     # Tăng sức mạnh
        num_steps=1000,   # Tăng thời gian học
        step_size=0.05, 
        alpha=0.1
    )