import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

"""
===========================================================
   RIS‑Assisted Satellite Link Optimisation
   – Successive Convex Approximation (SCA) with Flow Penalty –
===========================================================
•  Capacity metric : C = log2(1+SNR)   (bit/s/Hz)
•  Delay metric    : Σ d_ij · x_ij      (ms)
•  Objective       :  f = C − λ·delay − μ·‖flow−b‖²

To obtain **realistic** SNR values we explicitly include a link‑budget
section.  Two options are offered:
  1. *Quick test* — set a fixed, moderate path‑loss (≈ 40 dB per hop).
  2. *Realistic FSPL* — compute free‑space path‑loss for each hop and
     subtract antenna gains + RIS array gain.
Enable either mode via the boolean ``use_fspl`` below.
"""

# --------------------------------------------------------
#                      Reproducibility
# --------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------
#                  System / Model Parameters
# --------------------------------------------------------

N_RIS   = 16          # number of RIS elements
N_nodes = 5           # ground nodes in mesh network
P_tx    = 10.0        # satellite transmit power (W)
sigma2  = 1e-9        # noise power (W)  ≈ −90 dBm
lambda_delay = 1.0    # weight on delay term
mu_flow     = 50.0    # penalty weight enforcing flow conservation

# SCA / optimiser hyper‑parameters
max_outer   = 50      # outer SCA iterations
inner_v     = 100     # inner steps for RIS phase update
inner_x     = 100     # inner steps for routing matrix update
tol         = 1e-3    # convergence threshold

# --------------------------------------------------------
#                  Link‑Budget / Path‑Loss
# --------------------------------------------------------

use_fspl = False  # ← Set True for realistic FSPL calculation

if use_fspl:
    # ---- Physical parameters ----
    f_MHz = 12_000          # carrier frequency (MHz)
    d_sat_ris_km = 500      # Sat → RIS distance (km)
    d_ris_ground_km = 500   # RIS → ground distance (km)
    G_tx = 35               # satellite antenna gain (dBi)
    G_rx = 35               # ground antenna gain   (dBi)
    G_ris = 20 * math.log10(N_RIS)  # RIS array gain ≈ 24 dB

    # ---- FSPL helper ----
    def fspl_db(d_km: float) -> float:
        """Free‑space path loss (dB) at distance d_km and f_MHz."""
        return 20 * math.log10(d_km) + 20 * math.log10(f_MHz) + 32.44

    # Path‑loss for each hop after accounting for gains
    pl1 = fspl_db(d_sat_ris_km)   # Sat → RIS
    pl2 = fspl_db(d_ris_ground_km)  # RIS → GND

    # Distribute RIS gain equally to both hops
    path_loss1_dB = pl1 - (G_tx + G_ris / 2)
    path_loss2_dB = pl2 - (G_rx + G_ris / 2)
else:
    # ---- Quick‑test mode ----
    path_loss1_dB = 40   # Sat → RIS path‑loss (dB) — tweak as needed
    path_loss2_dB = 40   # RIS → Ground path‑loss (dB)

# Convert dB loss to amplitude attenuation
atten1 = 10 ** (-path_loss1_dB / 20)
atten2 = 10 ** (-path_loss2_dB / 20)

# --------------------------------------------------------
#                        Channels
# --------------------------------------------------------

h_d = (torch.randn(N_RIS, dtype=torch.complex64) + 1j * torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)
h_r = (torch.randn(N_RIS, dtype=torch.complex64) + 1j * torch.randn(N_RIS, dtype=torch.complex64)) / np.sqrt(2)

h_d *= atten1  # Sat → RIS
h_r *= atten2  # RIS → GND

# --------------------------------------------------------
#                Ground Network: Delay Matrix
# --------------------------------------------------------

d = torch.rand(N_nodes, N_nodes) * 9 + 1  # random delays ∈ [1,10] ms
d = (d + d.T) / 2
d.fill_diagonal_(0)

s, t = 0, 4      # source = node 0, destination = node 4

# --------------------------------------------------------
#                 Helper / Objective Functions
# --------------------------------------------------------


def v_complex(phi: torch.Tensor) -> torch.Tensor:
    """Convert real phase vector φ ∈ ℝᴺ → complex unit‑modulus vector v."""
    return torch.exp(1j * phi)


def exact_SNR(v: torch.Tensor) -> torch.Tensor:
    """Exact end‑to‑end linear SNR (scalar)."""
    h_eff = h_r.conj() @ (v * h_d)
    return (P_tx * torch.abs(h_eff) ** 2) / sigma2


def approx_SNR(v: torch.Tensor, v_k: torch.Tensor) -> torch.Tensor:
    """First‑order affine approximation of SNR around v_k (for SCA)."""
    g = h_r * h_d                 # element‑wise product (N×1)
    A = torch.outer(g, g.conj())  # N×N hermitian matrix
    term1 = 2 * torch.real(v_k.conj() @ (A @ v))
    term2 = torch.real(v_k.conj() @ (A @ v_k))
    return (P_tx / sigma2) * (term1 - term2)


def objective(v: torch.Tensor, v_k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Surrogate objective (to maximise) used in SCA."""
    # ‑‑‑ Capacity surrogate (bit/s/Hz) ‑‑‑
    snr_lin = torch.clamp(approx_SNR(v, v_k), min=0)
    capacity = torch.log2(1 + snr_lin)

    # ‑‑‑ Network delay term (ms) ‑‑‑
    delay = torch.sum(d * x)

    # ‑‑‑ Flow‑conservation penalty ‑‑‑
    flow = torch.sum(x, dim=1) - torch.sum(x, dim=0)
    b = torch.zeros_like(flow)
    b[s], b[t] = 1.0, -1.0
    penalty = mu_flow * torch.sum((flow - b) ** 2)

    return capacity - lambda_delay * delay - penalty

# --------------------------------------------------------
#               Optimisation Variables & Initialisation
# --------------------------------------------------------

phi = nn.Parameter(2 * np.pi * torch.rand(N_RIS))    # RIS phases (real)
x   = nn.Parameter(torch.zeros(N_nodes, N_nodes))    # routing (soft 0–1)

# simple initial path: s → 1 → t
path_init = [s, 1, t]
for i in range(len(path_init) - 1):
    x.data[path_init[i], path_init[i + 1]] = 1.0

opt_phi = torch.optim.Adam([phi], lr=0.01)
opt_x   = torch.optim.Adam([x],   lr=0.01)

# --------------------------------------------------------
#                       SCA Loop
# --------------------------------------------------------

obj_hist = []
for outer in range(max_outer):
    # ---- (a) update RIS phases ----
    phi_prev = phi.detach().clone()
    for _ in range(inner_v):
        opt_phi.zero_grad()
        loss_v = -objective(v_complex(phi), v_complex(phi_prev), x)
        loss_v.backward()
        opt_phi.step()

    # ---- (b) update routing matrix ----
    for _ in range(inner_x):
        opt_x.zero_grad()
        loss_x = -objective(v_complex(phi), v_complex(phi), x)  # v fixed
        loss_x.backward()
        opt_x.step()
        x.data.clamp_(0, 1)

    # ---- convergence & logging ----
    obj_val = objective(v_complex(phi), v_complex(phi), x).item()
    obj_hist.append(obj_val)
    print(f"Iter {outer:02d}  obj = {obj_val:>8.4f}")

    if outer > 0 and abs(obj_hist[-1] - obj_hist[-2]) < tol:
        print(f"Converged at outer iter {outer}\n")
        break

# --------------------------------------------------------
#                   Final Results & Plots
# --------------------------------------------------------

v_opt  = v_complex(phi).detach()
x_soft = x.detach()
x_bin  = (x_soft > 0.5).float()

snr_out = exact_SNR(v_opt).item()
cap_out = np.log2(1 + snr_out)
delay_out = torch.sum(d * x_bin).item()

print("==================== FINAL RESULT ====================")
print(f"SNR            : {snr_out:>7.2f}  ({10 * np.log10(snr_out):>5.2f} dB)")
print(f"Capacity       : {cap_out:>7.2f} bit/s/Hz")
print(f"Total delay    : {delay_out:>7.2f} ms")
print("Routing (binary):\n", x_bin.numpy())
print("======================================================")

# ---- Convergence curve ----
plt.figure()
plt.plot(obj_hist, marker="o")
plt.xlabel("Outer iteration")
plt.ylabel("Objective value")
plt.title("SCA convergence")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
