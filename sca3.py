import math, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt

"""
=====================================================================
   Two-Hop Cascaded RIS + **High-Gain Active Repeaters** (Design Study)
=====================================================================

We push the parameters into a feasible—but still challenging—region by:
•  **Huge RIS arrays**  (RIS-1: 4096, RIS-2: 16 384 elements)
•  **High-gain LNAs**   50 dB each (power), NF 3 dB
•  GEO & ground dishes  50 dBi, Ku-band 12 GHz
With these, SNR targets fall near 10–20 dB so the optimiser has room to
work.  All numbers are tweakable at the top of the script.
"""

# ----------------------------------------------------
# Helper utilities
# ----------------------------------------------------

def db2pow(db): return 10**(db/10)

def db2amp(db): return 10**(db/20)

def fspl_db(d_km, f_GHz):
    return 92.45 + 20*math.log10(d_km) + 20*math.log10(f_GHz)

# ----------------------------------------------------
# Scenario parameters (feel free to edit)
# ----------------------------------------------------

torch.manual_seed(4); np.random.seed(4)

# Distances (km)
D1 = 35486       # GEO → RIS-1 (LEO 300 km)
D2 = 300         # RIS-1 → RIS-2 (vertical)
D3 = 1           # RIS-2 → ground node
f_GHz = 12.0

# Antenna & array gains (dBi)
G_GEO   = 50
G_GND   = 50
N1, N2  = 4096, 16384
G_RIS1  = 10*math.log10(N1)   # 36 dB
G_RIS2  = 10*math.log10(N2)   # 42 dB
G_FEED  = 32                  # horn behind RIS-1

# Active repeater (LNA) specs
LNA_G_dB = 50                 # each stage
LNA_NF_dB = 3
G_LNA_pow = db2pow(LNA_G_dB)
NF_pow    = db2pow(LNA_NF_dB)

# Power & noise
P_tx_W   = 100.0              # 20 dBW ≈ 50 dBm
sigma2_W = db2pow(-90-30)     # −90 dBm thermal

# Objective trade-offs
lambda_delay = 2.0
mu_flow      = 200.0

# Optimiser hyper-params
N_nodes = 5
max_outer = 80
inner_phi = 300
inner_x   = 400
lr_phi, lr_x = 0.003, 0.01
tol = 1e-3

# Device
device = torch.device('cpu')

def cplx_rand(*size):
    return (torch.randn(size, dtype=torch.float64, device=device) +
            1j*torch.randn(size, dtype=torch.float64, device=device)) / math.sqrt(2)

# ----------------------------------------------------
# Link-budget
# ----------------------------------------------------

PL1 = fspl_db(D1, f_GHz) - (G_GEO + G_RIS1)         # GEO→RIS-1
PL2 = fspl_db(D2, f_GHz) - (G_FEED + 0.5*(G_RIS1+G_RIS2))
PL3 = fspl_db(D3, f_GHz) - (G_RIS2 + G_GND)

amp1, amp2, amp3 = db2amp(-PL1), db2amp(-PL2), db2amp(-PL3)
print("----- Link-budget (Ku, active repeaters) -------------")
print(f"Hop1  PL={PL1:6.1f} dB  amp×{amp1:.2e}")
print(f"Hop2  PL={PL2:6.1f} dB  amp×{amp2:.2e}")
print(f"Hop3  PL={PL3:6.1f} dB  amp×{amp3:.2e}")
print(f"LNA gain {LNA_G_dB} dB each, NF {LNA_NF_dB} dB")
print("------------------------------------------------------\n")

# ----------------------------------------------------
# Channel realisations
# ----------------------------------------------------

h_d1 = cplx_rand(N1)          * amp1
H12  = cplx_rand(N2, N1)      * amp2
h_2g = cplx_rand(N2)          * amp3

# ----------------------------------------------------
# Delay matrix
# ----------------------------------------------------

delay = torch.rand(N_nodes, N_nodes, dtype=torch.float64)*9 + 1
delay = (delay + delay.T)/2
delay.fill_diagonal_(0)

s, t = 0, 4

# ----------------------------------------------------
# Optimisation helper functions
# ----------------------------------------------------

def v(phi): return torch.exp(1j*phi)

def h_eff(phi1, phi2):
    tmp = H12 @ (v(phi1)*h_d1)
    return torch.dot(v(phi2).conj(), tmp * h_2g)


def capacity(phi1, phi2):
    sig_pow = torch.abs(h_eff(phi1, phi2))**2 * (G_LNA_pow**2)  # 2 LNAs
    snr = (P_tx_W * sig_pow) / (sigma2_W * NF_pow)
    return torch.log2(1 + snr)


def objective(phi1, phi2, x):
    cap = capacity(phi1, phi2)
    dly = torch.sum(delay * x)
    flow = torch.sum(x,1) - torch.sum(x,0)
    b = torch.zeros_like(flow); b[s], b[t] = 1,-1
    pen = mu_flow * torch.sum((flow-b)**2)
    return cap - lambda_delay*dly - pen

# ----------------------------------------------------
# Variables & optimisers
# ----------------------------------------------------

phi1 = nn.Parameter(2*math.pi*torch.rand(N1, dtype=torch.float64, device=device))
phi2 = nn.Parameter(2*math.pi*torch.rand(N2, dtype=torch.float64, device=device))
x    = nn.Parameter(torch.zeros(N_nodes, N_nodes, dtype=torch.float64, device=device))

x.data[s,1] = 1; x.data[1,t] = 1
opt_phi = torch.optim.Adam([phi1, phi2], lr=lr_phi)
opt_x   = torch.optim.Adam([x], lr=lr_x)

# ----------------------------------------------------
# SCA-like alternating optimisation
# ----------------------------------------------------

obj_hist = []
for it in range(max_outer):
    for _ in range(inner_phi):
        opt_phi.zero_grad(); (-objective(phi1, phi2, x)).backward(); opt_phi.step()
    for _ in range(inner_x):
        opt_x.zero_grad(); (-objective(phi1, phi2, x)).backward(); opt_x.step(); x.data.clamp_(0,1)
    val = objective(phi1, phi2, x).item(); obj_hist.append(val)
    print(f"Iter {it:02d}  obj = {val:8.4f}")
    if it>0 and abs(obj_hist[-1]-obj_hist[-2])<tol: print(f"Converged @ {it}\n"); break

# ----------------------------------------------------
# Final metrics
# ----------------------------------------------------

x_bin = (x.detach()>0.5).double()
C_fin = capacity(phi1, phi2).item()
SNR_lin = 2**C_fin - 1
SNR_dB  = 10*math.log10(SNR_lin) if SNR_lin>0 else -float('inf')

print("=================== FINAL RESULT ====================")
print(f"SNR     : {SNR_lin:.3e}  ({SNR_dB:.2f} dB)")
print(f"Capacity: {C_fin:7.3f} bit/s/Hz")
print(f"Delay   : {torch.sum(delay*x_bin).item():6.2f} ms")
print("Routing :\n", x_bin.cpu().numpy())
print("====================================================")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(); plt.plot(obj_hist, marker='o');
plt.xlabel('Outer iteration'); plt.ylabel('Objective');
plt.title('Convergence – Big RIS + 50 dB LNAs');
plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()
