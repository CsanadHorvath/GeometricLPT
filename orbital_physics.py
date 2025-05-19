import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from matplotlib.patches import Wedge
from astropy import units as u
from math import degrees, radians

G = 6.6743015e-11 #m3kg-1s-2
c = 3e8
Sm = 2e30
Sr = 7e8

# https://en.wikipedia.org/wiki/Roche_lobe
def RocheLobeRadius(A, M1, M2):
    ''' M1 is the star that CAN overflow, i.e. the main sequence companion,
    not the white dwarf '''
    q = M1 / M2
    f1 = 0.38 + 0.2*np.log(q)
    f2 = 0.46224 * ((q/(1+q))**(1/3))
    return A * f2
    return A*np.max([f1, f2], axis=0)

def OrbitalRadius(T, M1, M2):
    return (G*(M1+M2)*(T**2)/(4*np.pi**2))**(1/3)

def OrbitalVelocity(T, M1, M2):
    A = OrbitalRadius(T, M1, M2)
    A1 = A * M1 / (M1 + M2)
    A2 = A * M2 / (M1 + M2)
    V1 = 2*np.pi*A1 / T
    V2 = 2*np.pi*A2 / T
    return V1, V2

def LightCylinderRadius(S):
    # 2 * np.pi * R_lc / S = c
    # R_lc = c * S / (2 * np.pi)
    return c * S / (2*np.pi)

def StellarRadius(M):
    return (M/Sm)**0.8 * Sr

def CorotationRadius(M, S):
    # The radius where the orbital period matches the spin period
    return OrbitalRadius(S, M, 0)

def AlfvenRadius(R, B, M, Mdot):
    return (2*(R**6)*(B**2)/(4*Mdot*np.sqrt(G*M))) ** (2/7)

# Baraffe & Chabrier 1987 and
# Cifuentes et al. 2020
M_3V_lower = 0.14 * Sm
M_3V_upper = 0.5 * Sm
M_3V_arr = np.linspace(M_3V_lower, M_3V_upper, 101)

# Magnetic white dwarf masses from Amorin et al. 2023
# Can span 0.2 to 1.4 solar masses!
M_WD_lower = 0.6 * Sm # 0.2 * Sm
M_WD_upper = 1.2 * Sm # 1.3 * Sm
M_WD_arr = np.linspace(M_WD_lower, M_WD_upper, 101)

# Periods
orbit_idx = 1
spin_idx = 1
p_pulse = 1318.1957
p_orbit = 31482.3 * orbit_idx
p_spin = p_orbit / (p_orbit/p_pulse + spin_idx)

# Calculating all radii
M_3V_mat, M_WD_mat = np.meshgrid(M_3V_arr, M_WD_arr)
R_orb_mat = OrbitalRadius(p_orbit, M_3V_mat, M_WD_mat)
R_co_mat = CorotationRadius(M_WD_mat, p_spin)
R_Roche_mat = RocheLobeRadius(R_orb_mat, M_3V_mat, M_WD_mat)
R_3V_arr = StellarRadius(M_3V_arr)
R_3V_mat = StellarRadius(M_3V_mat)
V1_mat, V2_mat = OrbitalVelocity(p_orbit, M_3V_mat, M_WD_mat)
R_lc = LightCylinderRadius(p_spin)

# Plotting all radii
fig = plt.figure(figsize=(8,7))
fig.suptitle(f'GPM J1839-10     T = {p_orbit:.2f} s')

gspec = fig.add_gridspec(nrows=3, ncols=5)
ax1  = fig.add_subplot(gspec[1:3, 0:3], zorder=1)
cax1 = fig.add_subplot(gspec[1:3,  3 ], aspect=17)
cax2 = fig.add_subplot(gspec[1:3,  4 ], aspect=17)
ax2  = fig.add_subplot(gspec[ 0 , 0:3], zorder=2, sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)

R_orb_pcolor = ax1.pcolor(M_3V_arr/Sm, M_WD_arr/Sm, R_orb_mat/Sr)
plt.colorbar(R_orb_pcolor, cax=cax1, label='Orbital radius (R☉)')
R_Roche_contour = ax1.contour(M_3V_arr/Sm, M_WD_arr/Sm, R_Roche_mat/Sr, cmap='plasma')
plt.colorbar(R_Roche_contour, cax=cax2, label='Roche lobe radius (R☉)')
ax1.contour(M_3V_arr/Sm, M_WD_arr/Sm, (R_3V_mat > R_Roche_mat).astype(np.float64), levels=[0.5], colors=['red'])
ax1.set_xlabel('M-dwarf mass (M☉)')
ax1.set_ylabel('White dwarf mass (M☉)')

ax2.plot(M_3V_arr/Sm, R_3V_arr/Sr)
ax2.set_ylabel('M-dwarf radius (R☉)')

# Calculating orbital phase occluded
OPO_mat = (2*R_3V_mat) / (2*np.pi*R_orb_mat)

# Plotting orbital phase occluded
fig = plt.figure(figsize=(8,7))
gspec = fig.add_gridspec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gspec[0, 0])
OPO_pcolor = ax1.pcolor(M_3V_arr/Sm, M_WD_arr/Sm, OPO_mat)
plt.colorbar(OPO_pcolor, ax=ax1, label='Orbital phase occluded')
ax1.set_xlabel('M-dwarf mass (M☉)')
ax1.set_ylabel('White dwarf mass (M☉)')
ax1.set_title('Occlusion')

# Calculating time delay through orbit
R1_mat = R_orb_mat * M_WD_mat / (M_3V_mat + M_WD_mat)
R2_mat = R_orb_mat * M_3V_mat / (M_3V_mat + M_WD_mat)
T_mat = (2 * R2_mat) / (3e8)

# Plotting time delay through orbit
fig = plt.figure(figsize=(8,7))
gspec = fig.add_gridspec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gspec[0, 0])

T_pcolor = ax1.pcolor(M_3V_arr/Sm, M_WD_arr/Sm, T_mat)
plt.colorbar(T_pcolor, ax=ax1, label='Light travel time (s)')
ax1.set_xlabel('M-dwarf mass (M☉)')
ax1.set_ylabel('White dwarf mass (M☉)')
ax1.set_title('Light travel time through orbit')

# Getting ranges in solar units
M_m_range  = (np.min(M_3V_arr)    / Sm, np.max(M_3V_arr)    / Sm)
M_w_range  = (np.min(M_WD_arr)    / Sm, np.max(M_WD_arr)    / Sm)
A_range    = (np.min(R_orb_mat)   / Sr, np.max(R_orb_mat)   / Sr)
R_c_range  = (np.min(R_co_mat)    / Sr, np.max(R_co_mat)    / Sr)
A_m_range  = (np.min(R1_mat)      / Sr, np.max(R1_mat)      / Sr)
A_w_range  = (np.min(R2_mat)      / Sr, np.max(R2_mat)      / Sr)
R_m_range  = (np.min(R_3V_mat)    / Sr, np.max(R_3V_mat)    / Sr)
R_rl_range = (np.min(R_Roche_mat) / Sr, np.max(R_Roche_mat) / Sr)
dt_range   = (np.min(T_mat)           , np.max(T_mat)           )
R_lc       = R_lc / Sr
uR_lc      = LightCylinderRadius(0.0002) / Sr

# Getting middle values in solar units
i, j = M_3V_mat.shape
i = int(i / 2)
j = int(j / 2)

M_m_mid  = M_3V_mat[i, j]
M_w_mid  = M_WD_mat[i, j]

A_mid = OrbitalRadius(p_orbit, M_m_mid, M_w_mid)
R_c_mid = CorotationRadius(M_w_mid, p_spin)
R_rl_mid = RocheLobeRadius(A_mid, M_m_mid, M_w_mid)
R_m_mid = StellarRadius(M_m_mid)
A_m_mid = A_mid * M_w_mid / (M_m_mid + M_w_mid)
A_w_mid = A_mid * M_m_mid / (M_m_mid + M_w_mid)
dt_mid = (2 * A_w_mid) / c

M_m_mid  /= Sm
M_w_mid  /= Sm
A_mid    /= Sr
R_c_mid  /= Sr
R_rl_mid /= Sr
R_m_mid  /= Sr
A_m_mid  /= Sr
A_w_mid  /= Sr

# Printing results
print(f'M_w  = {M_w_mid :.3f} ({M_w_range [0]:.3f}, {M_w_range [1]:.3f})')
print(f'M_m  = {M_m_mid :.3f} ({M_m_range [0]:.3f}, {M_m_range [1]:.3f})')
print(f'A    = {A_mid   :.3f} ({A_range   [0]:.3f}, {A_range   [1]:.3f})')
print(f'A_w  = {A_w_mid :.3f} ({A_w_range [0]:.3f}, {A_w_range [1]:.3f})')
print(f'A_m  = {A_m_mid :.3f} ({A_m_range [0]:.3f}, {A_m_range [1]:.3f})')
print(f'R_m  = {R_m_mid :.3f} ({R_m_range [0]:.3f}, {R_m_range [1]:.3f})')
print(f'R_c  = {R_c_mid :.3f} ({R_c_range [0]:.3f}, {R_c_range [1]:.3f})')
print(f'R_rl = {R_rl_mid:.3f} ({R_rl_range[0]:.3f}, {R_rl_range[1]:.3f})')
print(f'R_lc = {R_lc    :.5f} +- {uR_lc:.5f}')
print(f'dt   = {dt_mid  :.3f} ({dt_range  [0]:.3f}, {dt_range  [1]:.3f})')

# Plotting diagram

fig, axs = plt.subplots(1, 3, width_ratios=[8, 1, 4.3], figsize=(9.5, 5))

axs[0].spines.right.set_visible(False)
axs[1].spines.left.set_visible(False)
axs[1].yaxis.set_ticks([])
d = 1.0  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axs[0].plot([1, 1], [0, 1], transform=axs[0].transAxes, **kwargs)
axs[1].plot([0, 0], [0, 1], transform=axs[1].transAxes, **kwargs)

axs[0].set_xlabel('Y (Perpendicular to WD spin axis and orbital axis) ($R_\\odot$)')
axs[0].set_ylabel('Z (WD spin axis) ($R_\\odot$)')
axs[2].axis('off')

theta = -1.744
phi = 0.909
alpha = np.pi/2
wbeam = 1.167 #/ 5
wmod = 1.213 #/ 5
h = A_mid * 1.2
arrow = 0.3
ax = axs[0]

# Center of mass
cm = [A_w_mid * np.sin(alpha), A_w_mid * np.cos(alpha)*np.sin(theta)]
# ax.add_patch(pts.Ellipse((0,0), 2*A_w_mid, 2*A_w_mid*np.sin(theta), edgecolor='blue', facecolor='none', label='Orbit of CM $A_{WD}$'))
# ax.scatter([cm[0]], [cm[1]], c='blue', marker='x', label='CM')
ax.errorbar([cm[0]], [cm[1]], xerr=[[A_w_mid-A_w_range[0]], [A_w_range[1]-A_w_mid]], c='blue', marker='x', label='CM position', capsize=5)
# ax.arrow(cm[0], cm[1], 0, arrow*A_w_mid, head_width=0.05, head_length=0.05, fc='blue', ec='blue')

# Orbit
ax.add_patch(pts.Ellipse((0,0), 2*A_mid, 2*A_mid*np.sin(theta), edgecolor='red', ls='--', facecolor='none', label='Orbit of MD at radius $A$'))
# ax.add_patch(pts.Arc((0,0), 2*A_mid, 2*A_mid*np.sin(theta), theta1=90-degrees(alpha-wmod/5), theta2=90-degrees(alpha+wmod/5), edgecolor='red', facecolor='none', linewidth=3))
ax.add_patch(pts.Arc((0,0), 2*A_mid, 2*A_mid*np.sin(theta), theta1=90-degrees(alpha-wmod  ), theta2=90-degrees(alpha+wmod  ), edgecolor='red', facecolor='none', linewidth=1, label='Modulation width $W_\\text{orb}$'))
# ax.plot([0.98*A_mid*np.sin(alpha-wmod), 1.02*A_mid*np.sin(alpha-wmod)], [0.98*A_mid*np.cos(alpha-wmod)*np.sin(theta), 1.02*A_mid*np.cos(alpha-wmod)*np.sin(theta)], color='black')

# M-dwarf
center = (A_mid * np.sin(alpha), A_mid * np.cos(alpha) * np.sin(theta))
ax.add_patch(pts.Wedge(center, R_m_range[1], 0, 360, edgecolor='none', facecolor='red', alpha=0.5, width=R_m_range[1]-R_m_range[0], label='MD radius $R_{MD}$'))
# ax.scatter([center[0]], [center[1]], marker='x', color='red', label='WD')
ax.errorbar([center[0]], [center[1]], xerr=[[A_mid-A_range[0]], [A_range[1]-A_mid]], marker='x', color='red', label='MD position ($\\phi_\\text{orb} = 90^{\\circ}$)', capsize=5)
# ax.arrow(center[0], center[1], 0, -arrow*A_mid, head_width=0.05, head_length=0.05, fc='red', ec='red')
ax.text(center[0]+0.18, center[1]+0.18, 'MD', color='red')

# Roche lobe
ax.add_patch(pts.Wedge(center, R_rl_range[1], 0, 360, edgecolor='none', facecolor='orange', alpha=0.5, width=R_rl_range[1]-R_rl_range[0], label='Roche lobe radius $R_{rl}$'))

# White dwarf
ax.add_patch(pts.Circle((0,0), 0.01, edgecolor='green' , facecolor='green' , linewidth=2)) # White dwarf
ax.text(0.0, 0.22, 'WD', ha='center', color='green')

# Corotation radius
# ax.add_patch(pts.Rectangle((-R_c_range[1], -h), R_c_range[1]-R_c_range[0], 2*h, edgecolor='none', facecolor='aqua', alpha=0.3, label='Corotation radius $R_c$'))
# ax.add_patch(pts.Rectangle(( R_c_range[0], -h), R_c_range[1]-R_c_range[0], 2*h, edgecolor='none', facecolor='aqua', alpha=0.3))
ax.add_patch(pts.Wedge((0,0), R_c_range[1], 0, 360, edgecolor='none', facecolor='aqua', alpha=0.5, width=R_c_range[1]-R_c_range[0], label='Corotation radius $R_c$'))

# Alfven radius
M = np.array([[0.6, 1.2]]) * Sm
R = np.array([[9000, 4000]]) * 1e3
Mdot = np.array([[1e-14], [1e-10]]) * Sm / 3.154e+7
B = 1e8 / 10000
R_A = AlfvenRadius(R, B, M, Mdot) / Sr
linestyles = [['--', '--'], ['-', '-']]
# labels = [['Alfen radius $R_A$ ($\\dot{M} = 10^{-14} M_\\odot yr^{-1}$)', None], ['Alfen radius $R_A$ ($\\dot{M} = 10^{-10} M_\\odot yr^{-1}$)', None]]
labels = [['Alfvén radius $R_A$', None], ['Alfvén radius $R_A$', None]]

for i in range(1):
    for j in range(2):
        ax.add_patch(pts.Circle((0,0), R_A[i,j], edgecolor='black' , facecolor='none', linestyle=linestyles[i][j], label=labels[i][j]))

R /= Sr
print(f'R_w  =       ({R  [0,1]:.3f}, {R  [0,0]:.3f})')
print(f'R_A  =       ({R_A[0,1]:.3f}, {R_A[0,0]:.3f})')

# Beam
r = h
ax.add_patch(pts.Wedge((0,0), r, 90-degrees(phi+wbeam  )    , 90-degrees(phi-wbeam  )    , edgecolor='none', facecolor='green', alpha=0.2, label='Beam width $W_\\text{spin}$', zorder=-100))
ax.add_patch(pts.Wedge((0,0), r, 90-degrees(phi+wbeam  )-180, 90-degrees(phi-wbeam  )-180, edgecolor='none', facecolor='green', alpha=0.2, zorder=-100))
# ax.add_patch(pts.Wedge((0,0), r, 90-degrees(phi+wbeam/5)    , 90-degrees(phi-wbeam/5)    , edgecolor='none', facecolor='green', alpha=0.4, zorder=-100))
# ax.add_patch(pts.Wedge((0,0), r, 90-degrees(phi+wbeam/5)-180, 90-degrees(phi-wbeam/5)-180, edgecolor='none', facecolor='green', alpha=0.4, zorder=-100))
x = (r*np.sin(phi), r*np.cos(phi))
ax.plot([-x[0], 0.95*x[0]], [-x[1], 0.95*x[1]], color='green', label='WD pole pointing ($\\phi = 270^{\\circ}$)')
ax.plot([-x[0], x[0]], [x[1], -x[1]], color='green', ls='--', label='WD pole pointing ($\\phi = 90^{\\circ}$)')
ax.arrow(-x[0], -x[1], 2*x[0],  2*x[1], width=0.015, head_width=0.1, head_length=0.2, overhang=0.2, length_includes_head=True, color='green', ec='none')
ax.text(x[0]+0.1, x[1], '$\\mu$', ha='center', color='green')

# Orbital phases
phases = [0.1, 0.0, -0.1, -0.2, -0.3]
r = 0.9 * A_mid
for phase in phases:
    ang = phase*2*np.pi + np.pi
    ax.text(r * np.sin(ang), r * np.cos(ang) * np.sin(theta), f'{phase}', va='center', ha='center')
    ax.plot([0.98*A_mid*np.sin(ang), 1.02*A_mid*np.sin(ang)], [0.98*A_mid*np.cos(ang)*np.sin(theta), 1.02*A_mid*np.cos(ang)*np.sin(theta)], 'r-')

ax.set_aspect('equal')
ax.set_xlim([-1.5, 3.2])
ax.set_ylim([-1.2, 2.5])

ax = axs[1]
ax.plot([R_lc, R_lc], [-h, h], '-', color='yellow', label='Light cyliner radius $R_{lc}$')
# ax.set_aspect('equal')
ax.set_ylim(axs[0].get_ylim())
ax.set_xticks([R_lc])

handles0, labels0 = axs[0].get_legend_handles_labels()
handles1, labels1 = axs[1].get_legend_handles_labels()
axs[2].legend(handles0+handles1, labels0+labels1, loc='upper right')
# plt.legend(, bbox_to_anchor=(1, 1), )
# plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.95, wspace=0.001, hspace=0.1)

plt.show()