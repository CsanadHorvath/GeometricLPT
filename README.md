Author: Csanad Horvath
Affiliation: Curtin University
Date: 19-05-2025

BinaryModel.py:
    This object stores the geometric parameters and lightcurve data, and calculates the predicted flux, modulation and beam widths if required, and scaling coefficient. Running python BunaryModel.py opens an interactive dynamic pulse profile. The geometric parameters are named by a different convention from what is published.
    ---
    published name      code name       description
    ---
    i                   theta           orbital inclination
    alpha               phi             magnetic obliquity
    phi                 beta            angular spin phase
    phi_orb             alpha           angular orbital phase
    phi_0               sigma           viewing angle about WD spin
    zeta                psi             viewing angle from WD spin
    W_spin              w_beam          angular beam width
    W_orb               w_mod           angular modulation width
    phi^(0)             beta_0          initial angular spin phase
    phi_orb^(0)         alpha_0         initial angular orbital phase
    ---
    
BinaryModelMCMC.py:
    This runs the MCMC fit of the model to the data. Much of it is hard-coded, sorry.

DynspecGroup.py:
    Handles reading of the dynamic spectra, performs calculations like dedispersion, Farraday rotation, and barrycentric correction, and generates lightcurves stored in pickle files.
    
plot_pulse_stack.py:
    Plots the colourful dynamic pulse profile for GPM J1839-10 with PA.

orbital_physics.py:
    Does calculations like orbital separation, corrotation radius, etc, and plots them on a cool diagram.
