"""Run the Gubser exact kinetic theory simulation. 

The notation is loosely based on Ashutosh Dash and Victor Roy, arXiv:2001.10756.  

- run(tauHbyR, rho_stop): Run the Gubser flow simulation for a given tauHbyR.
The parameters of the run needed to start the run are determined by
get_initial_state_params(tauHbyR). 

- get_initial_state_params(tauHbyR): Calculate the default values for the
initial condition parameters Q*R, LambdaT*R, fourpinbys, and rho_start based on
the input tauHbyR.

- T_hat(L0): Calculate the effective temperature T_hat from the zeroth moment L0, T_hat = L0**0.25. 

"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy import special
import scienceplots

import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook'])

# ==========================================
# 1. Define Coefficients (Eqs. 10, 11, 12)
# ==========================================

def a(n):
    "a coefficient in Eq. (9)"
    return 2 * (18 * n**2 + 9 * n - 4) / ((4 * n - 1) * (4 * n + 3))

def b(n):
    "b coefficient in Eq. (9)"
    return -4 * n * (n + 1) * (2 * n - 1) / ((4 * n - 1) * (4 * n + 1))

def c_coef(n):
    "c coefficient in Eq. (9)"
    return 2 * (n + 1) * (2 * n - 1) * (2 * n + 1) / ((4 * n + 1) * (4 * n + 3))


def T_hat(L0):
    """
    Temperature T_hat(rho) is related to the zeroth moment (energy density).  

    In a conformal system, energy density = Ce * T^4.  L0  is  e/Ce, so T_hat = L0**(1/4).  
    """
    return np.abs(L0)**0.25 


# Store the values of P_{2n}(0) for n=0 to 120 to avoid repeated calculations
# and roundoff in scipy.special.legendre(2*n)(0)
P2n_data = np.asarray([
0.5, 0.75, 0.9375, 1.09375, 1.23047, 1.35352, 1.46631, 1.57104, 
1.66924, 1.76197, 1.85007, 1.93416, 2.01475, 2.09224, 2.16697, 
2.2392, 2.30917, 2.37709, 2.44312, 2.50741, 2.5701, 2.63129, 2.69109, 
2.7496, 2.80688, 2.86302, 2.91807, 2.97211, 3.02519, 3.07735, 
3.12863, 3.1791, 3.22877, 3.27769, 3.32589, 3.3734, 3.42026, 3.46648, 
3.51209, 3.55712, 3.60158, 3.6455, 3.6889, 3.73179, 3.7742, 3.81614, 
3.85762, 3.89865, 3.93927, 3.97946, 4.01926, 4.05866, 4.09769, 
4.13634, 4.17464, 4.21259, 4.25021, 4.28749, 4.32445, 4.3611, 
4.39744, 4.43349, 4.46924, 4.50471, 4.5399, 4.57483, 4.60948, 
4.64388, 4.67803, 4.71193, 4.74558, 4.779, 4.81219, 4.84515, 4.87789, 
4.91041, 4.94271, 4.97481, 5.0067, 5.03839, 5.06988, 5.10117, 
5.13228, 5.16319, 5.19393, 5.22448, 5.25485, 5.28505, 5.31508, 
5.34494, 5.37464, 5.40417, 5.43354, 5.46275, 5.49181, 5.52071, 
5.54947, 5.57807, 5.60653, 5.63485, 5.66302, 5.69106, 5.71895, 
5.74672, 5.77434, 5.80184, 5.82921, 5.85645, 5.88356, 5.91055, 
5.93742, 5.96416, 5.99079, 6.01729, 6.04369, 6.06996, 6.09613, 
6.12218, 6.14812, 6.17395, 6.19968, 6.2253, 6.25081, 6.27622, 
6.30153, 6.32673, 6.35184, 6.37685, 6.40176, 6.42657, 6.45129, 
6.47591, 6.50044, 6.52488, 6.54922, 6.57348, 6.59765, 6.62173, 
6.64572, 6.66962, 6.69344, 6.71718, 6.74083, 6.7644, 6.78789, 
6.81129, 6.83462, 6.85787, 6.88104, 6.90413, 6.92714, 6.95008, 
6.97294, 6.99573, 7.01844, 7.04108, 7.06365, 7.08614, 7.10857, 
7.13092, 7.15321, 7.17542, 7.19757, 7.21965, 7.24166, 7.2636, 
7.28548, 7.30729, 7.32904, 7.35073, 7.37235, 7.3939, 7.4154, 7.43683, 
7.4582, 7.47951, 7.50076, 7.52194, 7.54307, 7.56414, 7.58515, 
7.60611, 7.627, 7.64784, 7.66862, 7.68935, 7.71002, 7.73064, 7.7512, 
7.7717, 7.79215, 7.81255, 7.8329, 7.85319, 7.87343, 7.89362, 7.91376, 
7.93384, 7.95388, 7.97386])

def get_P2n(n):
    """ Returns the value of the Legendre polynomial P_{2n}(0) """
    k = n-1
    v = P2n_data[k]/(-1)**n/n   # Adjusting for the sign based on n
    return v 

def test_P2n():
    """ Test the get_P2n function against scipy's legendre function """
    for n in range(1, 8):
        v = get_P2n(n)
        v2 = special.legendre(2*n)(0)
        print(v, v2)


# ==========================================
# Define the Differential Equation (Eq. 9)
# ==========================================
def gubser_derivatives(rho, L, nmax, c):
    """Computes the derivatives dL_n/drho for the Gubser flow moments.
    L is an array of moments [L_0, L_1, ..., L_nmax].
    """

    dL = np.zeros_like(L)
    
    # Equation for n = 0
    dL[0] = -np.tanh(rho) * (a(0) * L[0] + c_coef(0) * L[1])
    
    # Equations for n = 1 to nmax
    for n in range(1, nmax + 1):
        # Apply truncation: L_{n+1} is 0 if n == nmax
        L_next = L[n + 1] if n < nmax else 0.0
        
        # Calculate Eq. (9) for n >= 1
        term1 = -np.tanh(rho) * (a(n) * L[n] + b(n) * L[n - 1] + c_coef(n) * L_next)
        term2 = -(T_hat(L[0]) / c) * L[n]
        dL[n] = term1 + term2
        
    return dL

# The initial energy density is kappa * Q**3. kappa is inserted so that 
# Lambda = Q * (4 pi eta/s)**(1/6) 
kappa_constant = 1.156225998382431

def get_initial_state_params(tauHbyR,  verbose=False):
    """Calculate the parameters Q*R, LambdaT*R, fourpinbys, and rho_start based
    on the input tauHbyR.

    Q*R is the dimensionless Q parameter for the Gubser flow, LambdaT*R is the
    dimensionless temperature scale, fourpinbys the shear viscosity by entropy
    times four pi, and rho_start is the initial de Sitter time coordinate for
    starting the simulation.

    The initial energy density is tau*e = Ce * kappa_constant * QR**3.  The
    thermal energy density is e = Ce * T^4.  The code always works with  e(t,
    r)/Ce

    The value of kappa (or Cinfinity**(-8/9)) is taken to be kappa_constant =
    1.156225998382431, which is the value that gives the best match to the Bjorken attractor solution. 
    """

    # we always start in the free streaming regime, but the starting time is always much smaller than R
    rho_start = np.log(min(tauHbyR/1000.,1./1000.))
    if rho_start < -20:
        print("Warning: rho_start is very negative, which may lead to numerical instability. Consider increasing tauHbyR or adjusting the initial conditions.")

    # Pick rho0 to large negative value to ensure we start in the free-streaming regime, but not too negative to avoid numerical issues.
    rho0 = -30. 
    T0hat = 4.0 * kappa_constant / np.pi 
    QR = T0hat * np.cosh(rho0)
    fourpinbys = (QR * tauHbyR) ** (3/4) 
    LambdaTR = QR * (fourpinbys)**(1/6)

    # Check that LambdaTR is consistent with the expected scaling
    if not np.isclose(LambdaTR**(2/3), QR**(3/4)* tauHbyR**(1/12), rtol=1.e-6):
        print("LambdaTR does not match expected scaling with QR and tauHbyR. Check calculations.")

    x = np.cosh(rho0)/np.cosh(rho_start)

    if verbose:
        print(f"Calculated parameters for tauHbyR={tauHbyR}:")
        print( f"\ttau^4 * e /C_e  ={kappa_constant*QR**3/np.cosh(rho_start)**3} and { T0hat**4  * np.pi/4 * x**3}")
        print(f"\trho_start: {rho_start}")
        print(f"\tQR: {QR}") 
        print(f"\tfourpinbys: {fourpinbys}") 
        print(f"\ttauHbyR: {tauHbyR}")
        print(f"\tlog(tauHbyR): {np.log(tauHbyR)}")

    
    return QR, LambdaTR, fourpinbys, rho_start
    
def run(tauHbyR, rho_stop = 2):
    """Run the Gubser flow simulation for a given tauHbyR and return the
    solution and parameters of the run.  The solution runs from up to a
    specified rho_stop, which defaults to 2. 
    
    The code solves for L_n(rho), the moments of the distribution function,
    as a function of the de Sitter time coordinate rho.  See  Ashutosh Dash,
    Victor Roy, arXiv:2001.10756 for details. The (hatted) energy density is
    proportional to L0.  More precisely the energy density is Ce * T**4, and the
    code consistently works with L0 = e/Ce.   The effective temperature is
    L0**(1/4) . L1/L0 is related to the hatted pressure anisotropy, L1/L0 = (P_L
    - P_T)/e.  The code solves for the evolution of these moments as a
    function of rho, starting from an initial condition that corresponds to free
    streaming at a very early time (large negative rho).

    Returns:
        - solution: The result of the ODE solver, containing the evolution of
        the moments L_n as a function of rho.  The solution object can be
        interogated by calling solution.sol(rho) to get the values of L_n at any
        rho within the integration range.  The energy density (divided by C_e)
        is solution.sol(rho)[0] while (P_L - P_T)/e is
        solution.sol(rho)[1]/solution.sol(rho)[0].
        
    """

    QR, LambdaTR, fourpinbys, rho_start = get_initial_state_params(tauHbyR)

    nmax = 80

    # De Sitter time span (\rho_0 to \rho_final)
    # Note: Initializing exactly at rho=0 with shear can cause numerical instability.
    rho_span = (rho_start, rho_stop)

    # Initial conditions set to the free-streaming solution 
    L_initial = np.zeros(nmax + 1)
    L_initial[0] = kappa_constant * QR**3 / (np.cosh(rho_start))**3  
    for n in range(1, nmax + 1):
        L_initial[n] = L_initial[0] * get_P2n(n)  # Using P_{2n}(0) for initial anisotropy

    # Solve the coupled ODEs with continous evaluation points for better resolution in plots
    c_param = 5.0 * fourpinbys / (4 * np.pi) 
    solution = solve_ivp(
        fun=lambda rho, L: gubser_derivatives(rho, L, nmax, c_param),
        t_span=rho_span,
        y0=L_initial,
        dense_output=True,
        method='RK45'
    )

    return solution

