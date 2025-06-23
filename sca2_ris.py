import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

"""
====================================================================
  Two-Hop Cascaded RIS System (GEO → RIS-1 in LEO → RIS-2 near User)
====================================================================
We now model **two independent re-configurable intelligent surfaces**:
    RIS-1 : N₁ elements, deployed on a LEO satellite acting as a passive reflector.
    RIS-2 : N₂ elements, installed on a high building close to the terrestrial user network.

Signal path: GEO satellite  →  RIS-1  →  RIS-2  →  ground mesh network.

Optimisation goals remain:
    maximise  C = log2(1+SNR)  (bit/s/Hz)
    minimise  end-to-end delay through the ground mesh
subject to flow conservation.
We keep the **routing decision matrix** x  (N_nodes × N_nodes) and optimise
**two phase vectors** φ₁, φ₂ for the two RIS.

Unlike the previous single-RIS code, we now compute the **exact SNR**
and use standard gradient ascent (Adam) for both φ₁ and φ₂; the routing
matrix x is still updated in alternating fashion.
"""

# --------------------------------------------------------
#  Reproducibility & Imports
# --------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------
#  System Parameters
# --------------------------------------------------------

# --- RIS sizes ---
N_RIS1 = 16    # LEO RIS
N_RIS2 = 32    # Ground-side RIS

# --- Ground network ---
N_nodes = 5
s, t = 0, 4          # source / destination nodes

# --- Power & noise ---
P_tx   = 10.0         # satellite transmit power (W)
sigma2 = 1e-9         # noise power (W) (≈ −90 dBm)

# --- Objective weights ---
lambda_delay = 1.0     # weight on delay term
mu_flow     = 50.0     # penalty weight on flow conservation

# --- Optimiser hyper-params ---
max_outer = 50
inner_phase = 200
inner_x     = 200
lr_phase = 0.01
lr_x     = 0.01
tol      = 1e-3

# --------------------------------------------------------
#  Link Budget – choose either quick test or FSPL mode
# --------------------------------------------------------

use_fspl = False  # set True for full free-space path loss calculation

if use_fspl:
    # • GEO → LEO (RIS-1)
    # • LEO → ground-RIS (RIS-2)
    # • ground-RIS → user (mesh entry)
    f_MHz = 12_000
    d1_km = 36_000 - 500        # GEO–LEO vertical distance (~35 500 km)
    d2_km = 500                 # LEO–ground vertical distance
    d3_km = 2                   # last hop (RIS-2 to ground node) – short

    def fspl_db(d_km):
        return 20*math.log10(d_km) + 20*math.log10(f_MHz) + 32.44

    path_loss1_dB = fspl_db(d1_km) - 35   # assume 35 dBi sat tx gain
    path_loss2_dB = fspl_db(d2_km) - 24   # RIS-1 array gain ≈ 24 dB
    path_loss3_dB = fspl_db(d3_km) - 24   # RIS-2 array gain (same)
else:
    # quick-test mode – pick moderate values
    path_loss1_dB = 50   # GEO → RIS-1
    path_loss2_dB = 45   # RIS-1 → RIS-2
    path_loss3_dB = 20   # RIS-2 → ground node

atten1 = 10 ** (-path_loss1_dB / 20)
atten2 = 10 ** (-path_loss2_dB / 20)
atten3 = 10 ** (-path_loss3_dB / 20)

# --------------------------------------------------------
#  Channel Generation (Rayleigh small-scale)
# --------------------------------------------------------

# GEO → RIS-1 (vector N₁)
h_d1 = ((torch.randn(N_RIS1, dtype=torch.complex64) +
         1j*torch.randn(N_RIS1, dtype=torch.complex64)) / np.sqrt(2)) * atten1

# RIS-1 → RIS-2 (matrix N₂ × N₁)
H_12 = ((torch.randn(N_RIS2, N_RIS1, dtype=torch.complex64) +
         1j*torch.randn(N_RIS2, N_RIS1, dtype=torch.complex64)) / np.sqrt(2)) * atten2

# RIS-2 → ground entry point (vector N₂)
h_2g = ((torch.randn(N_RIS2, dtype=torch.complex64) +
         1j*torch.randn(N_RIS2, dtype=torch.complex64)) / np.sqrt(2)) * atten3

# --------------------------------------------------------
#  Ground Network Delay Matrix
# --------------------------------------------------------

d = torch.rand(N_nodes, N_nodes) * 9 + 1  # [1,10] ms random
d = (d + d.T) / 2
d.fill_diagonal_(0)

# --------------------------------------------------------
#  Helper Functions
# --------------------------------------------------------


def vec(ph):
    """Convert real phase vector φ → unit-modulus complex vector."""
    return torch.exp(1j * ph)


def cascaded_channel(phi1, phi2):
    """Compute effective scalar channel h_eff(φ₁, φ₂)."""
    v1 = vec(phi1)                     # N₁
    v2 = vec(phi2)                     # N₂
    tmp = H_12 @ (v1 * h_d1)           # N₂ vector (RIS-1 reflection)
    h_eff = h_2g.conj() @ (v2 * tmp)   # scalar after RIS-2
    return h_eff


def capacity_bits(phi1, phi2):
    snr = (P_tx * torch.abs(cascaded_channel(phi1, phi2))**2) / sigma2
    return torch.log2(1 + snr)


def objective(phi1, phi2, x):
    """Full objective to maximise."""
    cap = capacity_bits(phi1, phi2)           # bit/s/Hz
    delay = torch.sum(d * x)                  # ms
    flow  = torch.sum(x, 1) - torch.sum(x, 0)
    b = torch.zeros_like(flow); b[s], b[t] = 1, -1
    penalty = mu_flow * torch.sum((flow - b)**2)
    return cap - lambda_delay * delay - penalty

# --------------------------------------------------------
#  Optimisation Variables
# --------------------------------------------------------

phi1 = nn.Parameter(2 * np.pi * torch.rand(N_RIS1))
phi2 = nn.Parameter(2 * np.pi * torch.rand(N_RIS2))
x    = nn.Parameter(torch.zeros(N_nodes, N_nodes))

# initialise a simple path s → 1 → t
path_init = [s, 1, t]
for i in range(len(path_init)-1):
    x.data[path_init[i], path_init[i+1]] = 1.

opt_phase = torch.optim.Adam([phi1, phi2], lr=lr_phase)
opt_x     = torch.optim.Adam([x], lr=lr_x)

# --------------------------------------------------------
#  Alternating Optimisation Loop
# --------------------------------------------------------

obj_hist = []
for outer in range(max_outer):
    # -- (a) optimise phases φ₁, φ₂ --
    for _ in range(inner_phase):
        opt_phase.zero_grad()
        loss_p = -objective(phi1, phi2, x)
        loss_p.backward()
        opt_phase.step()

    # -- (b) optimise routing matrix x --
    for _ in range(inner_x):
        opt_x.zero_grad()
        loss_x = -objective(phi1, phi2, x)
        loss_x.backward()
        opt_x.step()
        x.data.clamp_(0, 1)

    # -- logging & convergence check --
    obj_val = objective(phi1, phi2, x).item()
    obj_hist.append(obj_val)
    print(f"Iter {outer:02d}  obj = {obj_val:>8.4f}")

    if outer > 0 and abs(obj_hist[-1] - obj_hist[-2]) < tol:
        print(f"Converged at outer iter {outer}\n")
        break

# --------------------------------------------------------
#  Final Metrics & Plot
# --------------------------------------------------------

x_bin = (x.detach() > 0.5).float()
cap_fin = capacity_bits(phi1, phi2).item()
snr_fin = 2**cap_fin - 1

delay_fin = torch.sum(d * x_bin).item()

print("==================== FINAL RESULT ====================")
print(f"SNR            : {snr_fin:>7.2f}  ({10*np.log10(snr_fin):>5.2f} dB)")
print(f"Capacity       : {cap_fin:>7.2f} bit/s/Hz")
print(f"Total delay    : {delay_fin:>7.2f} ms")
print("Routing (binary):\n", x_bin.numpy())
print("======================================================")

plt.figure()
plt.plot(obj_hist, marker='o')
plt.xlabel('Outer iteration')
plt.ylabel('Objective value')
plt.title('Convergence – two-RIS system')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
