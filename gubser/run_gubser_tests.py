
def run_bjorken():
    """ Run gubser solution in the bjorken limit  where tauHbyR=0.0001 is very small. The goal here is to reproduce Fig. 1 of 1908.02866 

    We are using the gubser kinetic theory code to run a simulation of the Bjorken flow, which the system approaches at early times.
    """
    tauHbyR = 0.0001
    solution = gk.run(tauHbyR=tauHbyR, rho_stop=np.log(100*tauHbyR))

    plot_bjorken_solution(tauHbyR, solution, (solution.t[0], solution.t[-1])) 
    
def plot_bjorken_solution(tauHbyR, solution, rho_span):
    """ Plot the canonical bjorken attractor plot of Fig. 1 of 1908.02866, which
    is a plot of e(tau)/e_hydro(tau) as a function of w = T_eff(tau) tau / (4 pi
    eta/s). """  

    QR, LambdaTR, fourpinbys, rho_start  = gk.get_initial_state_params(tauHbyR, verbose=True)
    
    rho_eval = np.linspace(rho_span[0], rho_span[1], 500)

    tau = 1./(-np.sinh(solution.t)) 
    T =  gk.T_hat(solution.y[0])/tau
    w = T * tau / (fourpinbys)
    wb = T * tau / (fourpinbys) * 4*np.pi/5
    e = T**4 

    y = tau**(4/3)*e
    enorm = y[-1]* (1. + 2./(3*np.pi*w[-1]))   

    # Set the size of the plot to be square
    fig, ax = plt.subplots(figsize=(5*1.1, 5))
    # make the x axis log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # set the x limits to be from 0.01 to 10
    ax.set_xlim(0.05, 15)
    ax.set_ylim(0.3, 1.05)
    # put horizontal line black dotted line at y = 1
    ax.axhline(1, color='black', linestyle=':')
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([f'{x:.1f}' for x in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    
    navier_stokes = 1 - 2./(3*np.pi*w)
    ax.plot(w, y/enorm, label='Gubser Kinetics', linewidth=2)
    ax.plot(w[navier_stokes > 0.5], navier_stokes[navier_stokes > 0.5], label='Navier-Stokes', linestyle='--', color='C1')

    free = w**(4/9)
    Cinf = free[0] * enorm/y[0] 
    k = Cinf**(-9/8)
    print("Parameters for Bjorken plot:")
    print("k-computed = ", k, "kappa-default = ", gk.kappa_constant)
    print("Cinf = ", Cinf, "Cinf-default = ", gk.kappa_constant**(-8/9) )
    free = (gk.kappa_constant**2*w)**(4/9)

    # plot the freee where free is less than 0.7
    ax.plot(w[free < 0.7], free[free < 0.7], label='Free Streaming', linestyle='-.', color='C2')
    ax.set_xlabel(r'$w = T_{\rm eff}(\tau) \tau / (4\pi \eta/s)$')
    ax.set_ylabel(r'$e(\tau)  / (e_{\rm hydro}(\tau))$')
    ax.legend(loc="lower right")
    plt.show()


# =============================================================u
    
def temperature_at_rhozero(tauHbyR_list):
    """ Calculate the temperature at rho=0 for a list of tauHbyR values.  This is used to make the plot of T(rho=0) as a function of tauHbyR.  The temperature is calculated by running the Gubser flow simulation for each tauHbyR value and extracting the temperature at rho=0 from the solution.  The temperature is then normalized by LambdaTR**(2/3) to make it dimensionless. """

    # Loop over the 
    T = []
    for tauHbyR in tauHbyR_list:
        QR, LambdaTR, fourpinbys, rho_start = gk.get_initial_state_params(tauHbyR, verbose=True)
        solution = gk.run(tauHbyR=tauHbyR, rho_stop=0)
        T.append(gk.T_hat(solution.y[0, -1])/LambdaTR**(2/3))

    return np.array(T)

def run_temperature_at_rhozero():
    """Run the Gubser flow simulation for a range of tauHbyR values and plot the temperature at rho=0 as a function of tauHbyR. """
    # chiinv = (tauHbyR)**(2/3)) 

    # Choose chiinv to be linearly spaced between 0.05 and 20 with 50 points
    #nchi = 10
    nchi = 300
    chinv = np.logspace(np.log10(0.01), np.log10(100), nchi)
    tauHbyR = chinv**(3/2)

    T = temperature_at_rhozero(tauHbyR)
    np.savetxt("temperature_at_rhozero.txt", np.column_stack((chinv, tauHbyR, T)), header="chinv tauHbyR T")

def pretty_plot_rhozero_temperature():
    """ Makes a plot of previously saved data for the temperature at rho=0 as a
    function of tauHbyR.  The data is saved in a file called
    temperature_at_zero.txt and has the following columns: chinv, tauHbyR, T,
    vischydro, free.  
    """

    # Make a figure with four panels, one rows and two columns
    fig, axs = plt.subplots(1, 2, figsize=(9.5,4.25))
    axs = axs.flatten()

    # Load the data from the file
    data = np.loadtxt("temperature_at_rhozero.txt", skiprows=1)
    chiinv = data[:, 0]
    chiinv_by2 = chiinv / 2
    tauHbyR = data[:, 1]
    Tvals = data[:, 2]

    # Find the viscous hydro and free streaming limits for the same tauHbyR values
    C1 = 0.646777
    vischydro = 1 - C1*chiinv/(3*np.pi)
    free = gk.kappa_constant**(1/4) * tauHbyR **(-1/12)

    # plot black dotted horizontal line at T = 1
    axs[0].axhline(1, color='black', linestyle=':')
    axs[0].plot(chiinv_by2, Tvals, label='Gubser Kinetics', linewidth=2)
    axs[0].plot(chiinv_by2[vischydro > 0.8], vischydro[vischydro > 0.8], label='Navier-Stokes', linestyle='--', color='C1')
    axs[0].plot(chiinv_by2[free < 0.9], free[free < 0.9], label='Free Streaming', linestyle='-.', color='C2')
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r'$(2\chi)^{-1}$')
    axs[0].set_ylabel(r'$T(\rho=0)\;  / \; \Lambda^{2/3}$')
    axs[0].legend()

    axs[1].axhline(1, color='black', linestyle=':')
    axs[1].plot(1./chiinv_by2, Tvals, label='Gubser Kinetics', linewidth=2)
    axs[1].plot(1./chiinv_by2[vischydro > 0.8], vischydro[vischydro > 0.8], label='Navier-Stokes', linestyle='--', color='C1')

    axs[1].plot(1./chiinv_by2[free < 0.9], free[free < 0.9], label='Free Streaming', linestyle='-.', color='C2')
    axs[1].set_xlim(0, 5)
    axs[1].set_xscale('linear')
    axs[1].set_xlabel(r'$2\chi$')
    axs[1].set_ylabel(r'$T(\rho=0) \; / \; \Lambda^{2/3}$')
    axs[1].legend(loc='lower right')

    fig.tight_layout()
    fig.savefig("temperature_at_rhozero.pdf", dpi=300)
    plt.show()    
    
# =============================================================u

def viscous_gubser_solution(rho, chinv) :
    """Returns the viscous Gubser solution for a given rho and H0. This is a known analytical solution that can be used to test the numerical solver."""
    g = -np.sinh(rho)

    G, Chinv = np.meshgrid(g, chinv)

    T0 = 1- 1/(3*np.pi*Chinv)

    ideal = T0 / (1 + G**2)**(1/3)

    return ideal, 1 - 1/(3*np.pi*Chinv) * G / np.sqrt(1 + G**2) * (1 - (1 + G**2)**(1/6) * scipy.special.hyp2f1(1/2, 1/6, 3/2, -G**2))

def load_temperature_data_allrho():
    """Load the temperature data from temperature_data.npz for all rho from the file and print the shapes of the arrays. """
    data = np.load("temperature_data.npz")
    chinv = data['chinv']
    tauHbyR = data['tauHbyR']
    rho = data['rho']
    T = data['T']
    L1 = data['L1']
    print("Loaded temperature data from {}:".format("temperature_data.npz"))
    print("\t chinv.shape:", chinv.shape, "tauHbyR.shape:", tauHbyR.shape, "rho.shape:", rho.shape, "T.shape:", T.shape, "L1.shape:", L1.shape)
    return chinv, tauHbyR, rho, T, L1
    
    
    
def run_temperature_allrho():
    """Run the Gubser flow simulation for a range of tauHbyR values and plot the
    temperature at rho=0 as a function of tauHbyR. 

    The data is saved in a file
    called temperature_data.npz and has the following arrays: chinv, tauHbyR,
    rho, T, L1.  

    The temperature T is calculated for all values of rho and all
    values of tauHbyR.  The temperature is normalized by LambdaTR**(2/3) to make
    it dimensionless.  The array L1 contains the values of L1 for all rho and
    tauHbyR, which can be used to calculate the pressure anisotropy (P_L -
    P_T)/e = L1/L0.  The code runs the Gubser flow simulation for each value of
    tauHbyR and extracts the temperature and L1 for all values of rho, and saves
    the data in a .npz file for later analysis and plotting. """
    # chiinv = (tauHbyR)**(2/3)) 

    # Choose chiinv to be linearly spaced between 0.05 and 20 with 50 points
    #nchi = 10
    nchi = 300
    chinv = np.logspace(np.log10(0.01), np.log10(100), nchi)
    tauHbyR = chinv**(3/2)
    
    rho_stop = 2
    #nrho = 6
    nrho = 401
    rho = np.linspace(-6, rho_stop, nrho)

    # Loop over the values of tauHbyR

    T = np.zeros((nchi, nrho))
    L1 = np.zeros((nchi, nrho))
    for i, tH in enumerate(tauHbyR):
        QR, LambdaTR, fourpinbys, rho_start = gk.get_initial_state_params(tH)
        solution  = run(tauHbyR=tH, rho_stop=rho_stop)
        L0 = solution.sol(rho)[0, :]
        data = solution.sol(rho)[0:2]/(LambdaTR ** (2/3))
        print(data.shape)
        T[i, :] = data[0, :]
        L1[i, :] = data[1, :]
        
    np.savez("temperature_data.npz", chinv=chinv,  tauHbyR=tauHbyR, rho=rho,T=T, L1=L1)   

    load_temperature_data() 

if __name__ == "__main__":
    run_sample()
    #run_bjorken()
    # run_temperature_at_rhozero()
    #pretty_plot_rhozero_temperature()
    #run_temperature() 
    #load_temperature_data_allrho()
    