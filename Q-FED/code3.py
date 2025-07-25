import pennylane as qml
from pennylane import numpy as np
import torch

dev = qml.device("default.qubit", wires=3)

def partial_trace(rho, keep, dims):
    """
    Tính partial trace trên hệ thống nhiều qubit.
    rho: mật độ ma trận toàn phần (shape (dim_total, dim_total))
    keep: list các chỉ số subsystem muốn giữ (vd: [2])
    dims: list kích thước subsystem, vd [2, 2, 2] cho 3 qubit
    """
    N = len(dims)
    traced = [i for i in range(N) if i not in keep]

    # reshape rho thành tensor 2N chiều
    reshaped_dims = dims + dims
    rho_reshaped = rho.reshape(reshaped_dims)

    # trace lần lượt các subsystem cần loại bỏ
    for t in sorted(traced, reverse=True):
        rho_reshaped = np.trace(rho_reshaped, axis1=t, axis2=t+N)

    return rho_reshaped



@qml.qnode(dev, interface="torch")
def teleport_circuit(alpha, beta, m0=None, m1=None):
    qml.StatePrep([alpha.item(), beta.item()], wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    if m0 is None or m1 is None:
        return qml.probs(wires=[0, 1])

    if m1 == 1:
        qml.PauliX(wires=2)
    if m0 == 1:
        qml.PauliZ(wires=2)

    return qml.state()

def fidelity(state1, state2):
    overlap = torch.dot(state1.conj(), state2)
    return torch.abs(overlap) ** 2

if __name__ == "__main__":
    alpha = torch.tensor(0.6, dtype=torch.complex64)
    beta = torch.tensor(0.8, dtype=torch.complex64)

    probs = teleport_circuit(alpha, beta)
    probs_np = probs.detach().numpy()
    probs_np /= probs_np.sum()
    print("Xác suất đo Alice (wire 0,1):", probs_np)

    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    measured_index = np.random.choice(4, p=probs_np)
    m0, m1 = outcomes[measured_index]
    print(f"Kết quả đo Alice: m0={m0}, m1={m1}")
    full_state = teleport_circuit(alpha, beta, m0, m1)
    if isinstance(full_state, torch.Tensor):
        full_state = full_state.detach().cpu().numpy()

    rho = np.outer(full_state, full_state.conj())
    reduced_rho = partial_trace(rho, keep=[2], dims=[2, 2, 2])

    # Tính vector trạng thái nếu reduced_rho là pure state (gần đúng)
    eigvals, eigvecs = np.linalg.eigh(reduced_rho)
    max_idx = np.argmax(eigvals)
    bob_state = eigvecs[:, max_idx]  # vector trạng thái Bob

    # Chuyển sang torch tensor để tính fidelity
    bob_state = torch.tensor(bob_state, dtype=torch.complex64)
    target_state = torch.tensor([alpha, beta])

    fidelity = torch.abs(torch.dot(bob_state.conj(), target_state))**2
    print("Fidelity:", fidelity.item())

