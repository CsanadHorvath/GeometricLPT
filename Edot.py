import numpy as np
import astropy.units as u
import astropy.constants as c
from itertools import product
 
def PrintRow(name, unit, arr):
    indices = [ [0, -1] for _ in range(arr.ndim) ]
    corner_indices = list(product(*indices))
    arr = np.array([arr[tuple(idx)].value for idx in corner_indices]) * arr.unit
    print(f'{name:20s} {unit:10s}', ' '.join([f'{x.to(unit).value:10.2e}' for x in arr]))

def PrintMinMax(name, arr, unit, end='', fmt='.2e'):
    print(f'{name} ∈ [{np.min(arr).to(unit).value:{fmt}}, {np.max(arr).to(unit).value:{fmt}}] {unit}', end)

P = 31482.4 * u.s                                           # Orbital period
Ps = 1265.2197 * u.s                                        # Spin period
Psdot = 3.6e-13 * u.s/u.s                                   # Period derivative limit

Ms = np.array([0.6  ,      1.2  ]).reshape((1,-1,1)) * u.solMass  # WD mass
Rs = np.array([0.006,      0.013]).reshape((1,-1,1)) * u.solRad   # WD radius
Bs = np.array([1e6, 1e7, 1e8, 1e9]).reshape((-1,1,1)) * u.G        # WD surface magnetic field
# Bs = np.array([            1e8  ]).reshape((-1,1,1)) * u.G        # WD surface magnetic field
Mc = np.array([0.14 ,      0.5  ]).reshape((1,1,-1)) * u.solMass  # MD mass
Bc = 1e3 * u.G                                              # MD surface magnetic field

Rc = (Mc/u.solMass)**0.8 * u.solRad                         # MD radius
a = (c.G*(Ms+Mc)*(P**2)/(4*np.pi**2))**(1/3)                # Orbital semi-major axis

# SPIN DOWN ENERGY

I = 2/5*Ms*Rs**2                                            # Moment of inertia of WD
EdotSD = 4*np.pi**2*I*Psdot*Ps**(-3)                        # Energy dissipation rate with P dot limit

Ins = 2/5*(1.4*u.solMass)*(10*u.km)**2                      # Moment of inertia of NS
EdotSDns = 4*np.pi**2*Ins*Psdot*Ps**(-3)                    # Energy dissipation rate with P dot limit

# UNIPOLAR INDUCTION ECME (Qu & Zhang eq 11)

Ω = 2*np.pi/P                                               # Orbital angular velocity
Ωs = 2*np.pi/Ps                                             # WD spin angular velocity
μs = Bs * Rs**3  # * 2*np.pi/c.mu0                          # WD magnetic moment - I don't understand why they can ignore that factor
μc = Bc * Rc**3  # * 2*np.pi/c.mu0                          # MD magnetic moment
ΔΩ = np.abs(Ωs - Ω)
ξ = Ωs / Ω
ζ = np.abs(1 - ξ)

# Rmag = 4*np.pi/c.c
# Φ = 2*μs*Rc/(c.c*a**2) * ΔΩ
# EdotQZ = 2*Φ**2 / Rmag # I couldn't get this version to work, only the approx
EdotQZ = (2e25*u.erg/u.s) * ζ**2 * (Bs/(1e6*u.G))**2 * (Rc/(0.2*u.solRad))**2 * (Ms/(0.8*u.solMass))**(-4/3)  * (P/(125.5*u.min))**(-14/3)

# UNIPOLAR INDUCTION (Yang eq 32,33)

vrel = ΔΩ * a
ζϕ = 4*vrel/(np.pi*c.c)
ΔΩΩcr = 450 * (P/(100*u.min))**(1/3) * ((Ms+Mc)/u.solMass)**(-1/3) # Unsaturated limit eq 31

# EdotUI = ζϕ * ΔΩ * μs**2 * Rc**2 / (2*a**5) # I couldn't get this version to work, only the approx

# Unsaturated
EdotUIu = (3.9e28*u.erg/u.s) * (μs/(1e34*u.G*u.cm**3))**2 * (Rc/(1e10*u.cm))**2 * (P/(100*u.min))**(-14/3) * ((Ms+Mc)/u.solMass)**(-4/3) * (ΔΩ/Ω)**2

# Saturated
EdotUIs = (1.8e31*u.erg/u.s) * (μs/(1e34*u.G*u.cm**3))**2 * (Rc/(1e10*u.cm))**2 * (P/(100*u.min))**(-13/3) * ((Ms+Mc)/u.solMass)**(-5/3) * (ΔΩ/Ω)

# MAGNETOSPHERIC INTERACTION (Yang eq 39)

# EdotMI = μs * μc * Ω**3 / (c.G*(Mc + Ms)) * (ΔΩ / Ω) # I couldn't get this version to work, only the approx
# EdotMI = (EdotMI.decompose().value * u.J/u.s).to(u.erg/u.s)
EdotMI = (8.6e31*u.erg/u.s) * (μs/(1e34*u.G*u.cm**3)) * (μc/(1e33*u.G*u.cm**3)) * (P/(100*u.min))**(-3) * ((Ms+Mc)/u.solMass)**(-1) * (ΔΩ/Ω)

# Geometry

# Axes: Bs, Ms, Mc, L, H, θ
Rrec = ( a / ((μc/μs)**(1/3) + 1) ).to(u.solRad)
L = np.stack([Rs * np.ones(Rrec.shape), Rrec], 3)[:,:,:,:,None,None] # New axes are for θ and H
Rrec = Rrec[:,:,:,None,None,None]
H = np.array([0.01, 0.1]).reshape((1,1,1,1,-1,1)) * L
θ = np.array([0, 15, 45, 61, 90]).reshape(1,1,1,1,1,-1) * u.deg
fθ = 3*np.sin(θ) * (1+np.cos(θ)**2) * (3+np.cos(θ)) / ( (1+3*np.cos(θ)**2) * (1+np.cos(θ))**2 ) * u.rad
Θb = fθ * H / Rrec

# Rrec *= np.ones(Θb.shape)
# L *= np.ones(Θb.shape)
# H *= np.ones(Θb.shape)
# θ *= np.ones(Θb.shape)

# Synchronisation timescale:

τsynMI = 2.6e5*u.yr * (Ms/(0.8*u.solMass)) * (Rs/(1e9*u.cm))**-1 * (μs/(1e34*u.G*u.cm**3))**-1 * (μc/(1e33*u.G*u.cm**3))**-1 * (P/(100*u.min)) * ((Ms+Mc)/u.solMass) * (ΔΩ/Ω)
PdotMI = (P - Ps) / τsynMI

# LUMINOSITY

S1GHz = 47 * u.mJy                         # Peak 1GHz flux of brightest pulse
d = np.array([5.7-2.9, 5.7, 5.7+2.9]) * u.kpc    # Distance from DM
α = -3.17                                   # Spectral index 1
q =  -0.56                                  # Spectral index 2
φmin = 0.1 * 2*np.pi                        # Minimum possible opening angle from duty cycle
φfit = np.radians(65)                       # Opening angle from model
Ωmin = 2*np.pi*(1-np.cos(φmin))             # Minimum solid angle
Ωfit = 2*np.pi*(1-np.cos(φfit))             # Model solid angle

# Numerical integral of spectrum because I coudln't find an analytic solution:
dν = 0.001 * u.GHz
ν = np.arange(0.001, 10, dν/u.GHz) * u.GHz
L4π = 4*np.pi * d**2 * S1GHz * np.sum((ν/u.GHz)**α * np.exp(q*(np.log(ν/u.GHz))**2) * dν) # Isotropic luminosity
LΩmin = Ωmin/(4*np.pi) * L4π                # Minimum possible luminosity
LΩfit = Ωfit/(4*np.pi) * L4π                # Luminosity given model opening angle
print(4*np.pi * d**2 * S1GHz * np.sum((ν/u.GHz)**α * np.exp(q*(np.log(ν/u.GHz))**2) * dν))

# The calculation in the 1839 discovery paper doesn't make sense...
# 1. Multiplied by both 4*pi*d**2 and omega_1GHz? You're doubling up on the solid angle
# 2. That doesn't look like the solution to the integral.
# 3. How are you taking sqrt of q which is negative anyway?

# ACCRETION PHASE PERIOD LIMIT Yang eq 5

# Not useful here because it's just an upper limit as explained by Yang.
# We are below this but that does not mean we are accreting.
Pacc = 147*u.min * (Mc / (0.2*u.solMass))**0.7

# UNIPOLAR INDUCTION PERIOD LIMIT Yang UI eq 7,9

PUI = np.sqrt(4*np.pi**2*μs/(c.G*(Ms+Mc)*Bc))
# PUI = 91*u.min * (μs/(1e34*u.G*u.cm**3))**0.5 * ((Ms+Mc)/u.solMass)**-0.5 * (Bc/(1e2*u.G))**-0.5

# PRINTING RESULTS

PrintRow('Bs', u.G      , Bs)
PrintRow('Ms', u.solMass, Ms)
PrintRow('Rs', u.solRad , Rs)
PrintRow('Mc', u.solMass, Mc)
PrintRow('Rc', u.solRad , Rc)
PrintRow('a' , u.solRad , a )
PrintRow('Spin-down Edot', u.erg/u.s, EdotSD)
PrintRow('Qu & Zhang Edot', u.erg/u.s, EdotQZ)
PrintRow('Yang UI unsat Edot', u.erg/u.s, EdotUIu)
PrintRow('Yang UI sat Edot', u.erg/u.s, EdotUIs)
PrintRow('Yang MI Edot', u.erg/u.s, EdotMI)
print('')
print('Unipolar inductor limits:')
PrintMinMax('    ζϕ', ζϕ.decompose(), '', '<< 1')
print(f'    ΔΩ/Ω = {ΔΩ/Ω:.2f} << ', end='')
PrintMinMax('(ΔΩ/Ω)cr', ΔΩΩcr, '', fmt='.0f')
print('    Therefore, UI would be in the unsaturated regime.')
PrintMinMax('    PUI', PUI, u.hour, fmt='.2f', end=f' < {P.to(u.hour):.2f}')
print('    Therefore, we would be in magnetospheric interaction phase.')
print('')
print(f'Luminosity given d ∈ [{d[0].value:.1f}, {d[2].value:.1f}] {d.unit} and peak S1GHz = {S1GHz}:')
PrintMinMax('    Isotropic emission         L4π', L4π  , u.erg/u.s, end=f'~ {L4π  [1].to(u.erg/u.s):.2e}')
PrintMinMax('    Geometric model beam     LΩfit', LΩfit, u.erg/u.s, end=f'~ {LΩfit[1].to(u.erg/u.s):.2e}')
PrintMinMax('    Narrowest possible beam  LΩmin', LΩmin, u.erg/u.s, end=f'~ {LΩmin[1].to(u.erg/u.s):.2e}')
print('')
print(f'For Pdot = {Psdot}:')
PrintMinMax('    WD spin-down  Edot', EdotSD  , u.erg/u.s)
print(f'    NS spin-down  Edot = {EdotSDns.to(u.erg/u.s):.2e}')

for iBs in range(Bs.shape[0]):
    print('')
    print(f'For MD field Bc = {Bc:.1e} and WD field Bs = {Bs[iBs,0,0]:.1e}:')
    PrintMinMax('    Qu & Zhang    Edot', EdotQZ [iBs,:,:], u.erg/u.s)
    PrintMinMax('    Yang UI unsat Edot', EdotUIu[iBs,:,:], u.erg/u.s)
    PrintMinMax('    Yang UI sat   Edot', EdotUIs[iBs,:,:], u.erg/u.s)
    PrintMinMax('    Yang MI       Edot', EdotMI [iBs,:,:], u.erg/u.s)
    PrintMinMax('                  τsyn', τsynMI [iBs,:,:], u.yr     )
    PrintMinMax('                  Pdot', PdotMI [iBs,:,:], u.s/u.s  )
    print('')
    PrintMinMax('    Rrec', Rrec[iBs,:,:,:,:,0], u.solRad, fmt='.5f')
    PrintMinMax('       L',    L[iBs,:,:,:,:,0], u.solRad, fmt='.5f')
    PrintMinMax('       H',    H[iBs,:,:,:,:,0], u.solRad, fmt='.5f')
    print('')

    for iθ in range(θ.shape[5]):
        print(f'    For θ = {θ[0,0,0,0,0,iθ]:3.0f}: ', end='')
        PrintMinMax('Θb',   Θb[iBs,:,:,:,:,iθ], u.deg, fmt='>7.3f')