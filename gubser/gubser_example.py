import numpy as np
import gubser_kinetics as gk
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'notebook'])
# =============================================================u

def run_sample():
    """Run a sample Gubser flow simulation and plot the results.  

    This function demonstrates how to run the Gubser flow simulation using the gubser_kinetics module. It sets a sample value for tauHbyR, retrieves the initial state parameters, runs the simulation, and plots the temperature T as a function of rho. You can adjust the value of tauHbyR and the range of rho to explore different scenarios.
    """ 

    tauHbyR = 1.0

    # get the intial state parameters for given tauHbyR and print them out
    QR, LambdaTR, fourpinbys, rho_start = gk.get_initial_state_params(tauHbyR, verbose=True)

    # run the code and get the solution for rho from -6 to 2
    rho_stop = 4
    solution = gk.run(tauHbyR=tauHbyR, rho_stop=rho_stop)

    # Choose where to plot the solution.  We will plot T(rho) for rho from -6 to 2.
    rho_eval = np.linspace(max(-6, rho_start), rho_stop, 500)
    That =  gk.T_hat(solution.sol(rho_eval)[0])/LambdaTR**(2/3)

    # Plot the results. 
    plt.plot(rho_eval, That)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\hat{T}(\rho)/\Lambda_T^{2/3}$')
    plt.title('Sample Gubser Flow Simulation')
    plt.show()

if __name__ == "__main__":
    run_sample()
