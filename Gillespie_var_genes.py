#!/usr/bin/env python3

'''
Parallel Gillespie algorithms for var gene switch implementation using as much
Boolean algebra as possible, just for the lulz :D the actual Gillespie sequence
runs only 4 ifs per timestep!

Pablo Cardenas R.
2.18
'''

### Imports ###
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots
from scipy import integrate # numerical integration
import numpy as np # handle arrays
import pandas as pd # data wrangling
import joblib as jl
    # parallel coffee beans get stupid things done faster with more energy

### Setup ###
sns.set_style("darkgrid") # make pwetty plots

### Constants ###
cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/

# Global param
N_GEN = 3 # number of var genes
N_MOL = 5 # number of molecular species per var gene

# Event ID constants
GEN_IDS = [ i for i in range(N_GEN) ] # one event per var gene

R_F = 0 # aslncRNA synthesis
R_B = 1 # aslncRNA degradation
E_F = 2 # Euchromatin conversion from heterochromatin
E_B = 3 # Heterochromatin conversion from euchromatin
N_F = 4 # Nuclear binding protein-aslncRNA complex formation
N_B = 5 # Nuclear binding protein-aslncRNA complex dissociation
P_F = 6 # PfEMP1 protein synthesis
P_B = 7 # PfEMP1 protein degradation (this event is equivalent to lysis)
S_F = 8 # slncRNA synthesis
S_B = 9 # slncRNA degradation

# State var ID constants
R = 0 # aslncRNA
E = 1 # Euchromatin
N = 2 # Nuclear binding protein-aslncRNA complex
P = 3 # PfEMP1 protein
S = 4 # slncRNA

### Methods ###
# User-defined methods #
def drawFigures():

    '''
    Main method containing all simulation and plotter calls.
    Writes figures to file.
    '''

    # Figure 3 - Trajectories
    print('\n*** Starting Fig 1 Sims ***')
    idc = 30
    # create, simulate model objects with the given time series and [ATP] in M
    t = np.linspace( 0, 48*idc, int(48*idc*100) ) # time vector to evaluate
    #ms = multiSim(20,t)
    m_4  = simModel( t )
        # shift 400 microM [ATP] series by 0.5 s as in paper for readability

    # # Set up model conditions
    # p = params() # get parameter values, store in dictionary p
    # y_0 = initCond() # get initial conditions
    # det_sol = odeSolver(odeFun,t,y_0,p,solver="LSODA");

    fig1([m_4])#,det_sol) # plot

    # # Figure 4 - Effect of volume
    # print('\n*** Starting Fig 2 Sims ***')
    # vol_vals = np.array( [1e-1,1e0,1e1,1e2,1e3] )
    # n_iter = 100 # number of iterations per [ATP]
    # t = np.linspace( 0, 1, int(1e5) ) # time vector to evaluate
    # vol_sims = [ multiSim(n_iter,t,vol=vol) for vol in vol_vals ]
    #     # run multiple sims for every vol, save in list
    # dat_vol = toDataTable(vol_sims,['vol','C_end']) # wrangle data into df
    # det_sols = jl.Parallel(n_jobs=jl.cpu_count(), verbose=6) \
    #     ( jl.delayed(detSol)(v) for v in vol_vals )
    # #dat_atp = dat_atp.groupby('vol')['C_end'].agg([np.mean,np.std]).reset_index()
    #     # Calculate mean and standard deviation of load, grouping by [ATP]
    # fig2(dat_vol,det_sols) # plot

    print('\n*** Done :) ***\n')


# def detSol(vol):
#     # Set up model conditions
#     p = params() # get parameter values, store in dictionary p
#     y_0 = initCond() # get initial conditions
#     t = np.linspace(p['t_0'],p['t_f'], int( (p['t_f']-p['t_0'])/p['t_den'] ) + 1)
#         # time vector based on minimum, maximum, and time step values
#
#     # Solve model
#     sol = odeSolver(odeFun,t,y_0,p,solver="LSODA");
#
#     return sol.y[2,-1]


def multiSim(n_iter,t_vec):

    '''
    Performs multiple simulations using embarrassing parallelization.
    Arguments:
        n_iter  : int - total number of iterations of simulation to run
        t_vec   : s - time vector for simulations
        atp     : M - concentration of ATP in M
        load    : 'trap' or pN - load scheme to use, either optical trap
                  function ('trap'), or constant value
        stall_t : s - time without movement after which simulation stops
    Returns:
        list of StocModel objects after simulation
    '''

    r = jl.Parallel(n_jobs=jl.cpu_count(), verbose=6) \
        ( jl.delayed(simModel)(t_vec) for _ in range(n_iter) )
        # parallelizes simulations across all available cores
        # verbose 10 shows progress bar

    return r


def simModel(t_vec):

    '''
    Creates and simulates stochastic model 'a la Gillespie.
    Arguments:
        t_vec   : s - time vector for simulations
        atp     : M - concentration of ATP in M
        load    : 'trap' or pN - load scheme to use, either optical trap
                  function ('trap'), or constant value
        stall_t : s - time without movement after which simulation stops
    Returns:
        StocModel object after simulation
    '''

    m = StocModel(t_vec) # create model object
    # m.vol = vol # set ATP
    # m.x_var = m.x_var * vol
    m.gillespie() # simulate
    m.gene_end = np.argmax(m.x[P,:,-1])
        # returns index ID of maximally expressed var gene at end

    return m


# def toDataTable(sim_arr,props):
#
#     '''
#     Reformats list of lists of simulations into a long-formatted Pandas
#     dataframe with specific endpoint property information.
#     Arguments:
#         sim_arr : list - list of lists of simulations produced by multiSim
#         props   : list - list of strings with names of properties to be
#                   extracted from StocModel simulation objects
#     Return:
#         Pandas dataframe with columns for Condition (sim_arr dimension 1),
#             Replicate (sim_arr dimension 2), and each property to be extracted.
#     '''
#
#     dat = pd.DataFrame( # init df
#         np.nan, # fill with NaNs
#         index = range( len(sim_arr) * len(sim_arr[0]) ), # 1 row for every sim
#         columns = ['Condition','Replicate'] + props
#             # columns for both original array indexes, 1 col for every prop
#     )
#
#     for i,cond in enumerate(sim_arr): # iterate across conditions
#         for j,sim in enumerate(cond): # iterate within condition across reps
#             dat.iloc[j+i*len(sim_arr[0]),0] = i # store condition number
#             dat.iloc[j+i*len(sim_arr[0]),1] = j # store replicate number
#             for k,prop in enumerate(props): # for every property to be saved,
#                 dat.iloc[j+i*len(sim_arr[0]),k+2] = getattr(sim,prop) # save it
#
#
#
#     return dat


def fig1(models):

    """
    This function makes a plot for Figure 1 by taking all the simulation objects
    as arguments, and prints out the plot to a file.
    Arguments:
        m_1000 : StocModel object for 1 mM [ATP]
        m_400 : StocModel object for 0.4 mM [ATP]
    """

    t = models[0].t[:] # get time values

    plt.figure(figsize=(6, 12), dpi=200) # make new figure
    ax = plt.subplot(6, 1, 1) # get axis
    for m in models:
        plt.plot(m.t, m.x[P,0,:], color=cb_palette[2], alpha=0.2, linewidth=1) # plot
        plt.plot(m.t, m.x[P,1,:], color=cb_palette[1], alpha=0.2, linewidth=1) # plot

    plt.plot(m.t, m.x[P,0,:], label=r'$var$ 1', color=cb_palette[2], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[P,1,:], label=r'$var$ 2', color=cb_palette[1], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[P,2,:], label=r'$var$ 3', color=cb_palette[7], alpha=0.8, linewidth=1) # plot
    # plt.plot(t, sol.y[0,:], label=r'$A,B$', color=cb_palette[2])
    # plt.plot(t, sol.y[2,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)') # labels
    plt.ylabel('PfEMP1 proteins (AU)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    ax = plt.subplot(6, 1, 2) # get axis
    for m in models:
        plt.plot(m.t, m.x[N,0,:], color=cb_palette[2], alpha=0.2, linewidth=1) # plot
        plt.plot(m.t, m.x[N,1,:], color=cb_palette[1], alpha=0.2, linewidth=1) # plot

    plt.plot(m.t, m.x[N,0,:], label=r'$var$ 1', color=cb_palette[2], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[N,1,:], label=r'$var$ 2', color=cb_palette[1], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[N,2,:], label=r'$var$ 3', color=cb_palette[7], alpha=0.8, linewidth=1) # plot
    # plt.plot(t, sol.y[0,:], label=r'$A,B$', color=cb_palette[2])
    # plt.plot(t, sol.y[2,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)') # labels
    plt.ylabel('NBP complex (count)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    ax = plt.subplot(6, 1, 3) # get axis
    for m in models:
        plt.plot(m.t, m.x[R,0,:], color=cb_palette[2], alpha=0.2, linewidth=1) # plot
        plt.plot(m.t, m.x[R,1,:], color=cb_palette[1], alpha=0.2, linewidth=1) # plot

    plt.plot(m.t, m.x[R,0,:], label=r'$var$ 1', color=cb_palette[2], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[R,1,:], label=r'$var$ 2', color=cb_palette[1], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[R,2,:], label=r'$var$ 3', color=cb_palette[7], alpha=0.8, linewidth=1) # plot
    # plt.plot(t, sol.y[0,:], label=r'$A,B$', color=cb_palette[2])
    # plt.plot(t, sol.y[2,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)') # labels
    plt.ylabel('aslncRNA (count)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    ax = plt.subplot(6, 1, 4) # get axis
    for m in models:
        plt.plot(m.t, m.x[S,0,:], color=cb_palette[2], alpha=0.2, linewidth=1) # plot
        plt.plot(m.t, m.x[S,1,:], color=cb_palette[1], alpha=0.2, linewidth=1) # plot

    plt.plot(m.t, m.x[S,0,:], label=r'$var$ 1', color=cb_palette[2], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[S,1,:], label=r'$var$ 2', color=cb_palette[1], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[S,2,:], label=r'$var$ 3', color=cb_palette[7], alpha=0.8, linewidth=1) # plot
    # plt.plot(t, sol.y[0,:], label=r'$A,B$', color=cb_palette[2])
    # plt.plot(t, sol.y[2,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)') # labels
    plt.ylabel('slncRNA (count)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    ax = plt.subplot(6, 1, 5) # get axis
    for m in models:
        plt.plot(m.t, m.x[E,0,:], color=cb_palette[2], alpha=0.2, linewidth=1) # plot
        plt.plot(m.t, m.x[E,1,:], color=cb_palette[1], alpha=0.2, linewidth=1) # plot

    plt.plot(m.t, m.x[E,0,:], label=r'$var$ 1', color=cb_palette[2], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[E,1,:], label=r'$var$ 2', color=cb_palette[1], alpha=0.8, linewidth=1) # plot
    plt.plot(m.t, m.x[E,2,:], label=r'$var$ 3', color=cb_palette[7], alpha=0.8, linewidth=1) # plot
    # plt.plot(t, sol.y[0,:], label=r'$A,B$', color=cb_palette[2])
    # plt.plot(t, sol.y[2,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)') # labels
    plt.ylabel('Euchromatin (count)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    ax = plt.subplot(6, 1, 6) # get axis
    T = m.mu * np.power( np.cos( (m.t-m.xi)*np.pi/m.lam ), 2*m.nu )
    plt.plot(m.t, T, color=cb_palette[3], label=r'Cell cycle TF', alpha=1, linewidth=1) # plot
    plt.xlabel('Time (h)') # labels
    plt.ylabel('Transcription factor (count)')
    handles, labels = ax.get_legend_handles_labels() # get legend
    plt.legend(handles, labels, loc='upper right') # show it

    plt.savefig('Fig1.png', bbox_inches='tight') # save


# def fig2(dat,det_sols):
#
#     """
#     This function makes a plot for Figure 4 by taking the solution dataframe as
#     an argument, and prints out the plot to a file.
#     Arguments:
#         dat : solution dataframe taken from simulation outputs
#     """
#     plt.figure(figsize=(6, 4), dpi=200) # make new figure
#     ax = plt.subplot(1, 1, 1) # get axis
#     ax = sns.violinplot(x="vol", y="C_end", data=dat, width=1.4)
#     ax.scatter(range(len(det_sols)), det_sols, marker='x', s=100, color=cb_palette[1],label='Deterministic \nsteady state')
#     # plt.errorbar(x=dat['vol'], y=dat['mean'], yerr=0, color=cb_palette[6],
#     #     alpha=0.5) # plot lines (don't know why regular plot wasn't working)
#     # plt.errorbar(x=dat['vol'], y=dat['mean'], yerr=dat['std'],
#     #     capsize=5, color=cb_palette[6], fmt='o') # plot points and error bars
#     plt.xlabel(r'Volume (AU)') # labels
#     plt.ylabel('Steady state C concentration (AU)')
#     handles, labels = ax.get_legend_handles_labels()
#     plt.legend(handles, labels, loc='upper right')
#
#     plt.savefig('Fig2.png', bbox_inches='tight') # save


class StocModel(object):

    """
    Class defines a model's parameters and methods for changing system state
    according to the possible events and simulating a timecourse using the
    Gillespie algorithm.
    """

    def __init__(self, t_vec):

        '''
        Class constructor defines parameters and declares state variables.
        Arguments:
            t_vec    : s - array with times at which to evaluate simulation
        '''

        super(StocModel, self).__init__() # initialize as parent class object

        # Event modifier constants
        self.k_T = 1e0
        self.k_R = 5e1
        self.k_E = 1e2
        self.k_H = 5e1
        self.k_N = 2e1

        self.h_T = 3
        self.h_R = 3
        self.h_E = 1
        self.h_H = 1
        self.h_N = 1

        self.alp = 2e0
        self.bet = 1e0
        self.gam = np.array( [1e1]*N_GEN )
        self.dlt = 1e0 # .del appears to be reserved
        self.eps = 1e-3
        self.zet = 1e-3
        self.eta = 1e1
        self.the = 2e-1
        self.iot = 1e0
        self.kap = 1e0

        self.lam = 48
        self.mu = 2
        self.nu = 1
        self.xi = 24

        self.d = np.array( [100]*N_GEN )
            # total DNA positions for modification available for each gene
        self.n_t = 100 # total nuclear binding proteins available

        # Event IDs
        self.evt_IDs = [ R_F, R_B, E_F, E_B, N_F, N_B, P_F, P_B, S_F, S_B ]
            # event IDs in specific order

        # State variables
        self.t_var = 0.0 # s - time elapsed in simulation
        self.x_var = np.zeros( [ N_MOL,N_GEN ] )
            # state vectors (matrix due to multiple var genes)
            # state variable order: R, E, N, P, S
        self.x_var[E,:] = self.d[0]*0.25 # start at full euchromatin
        self.x_var[E,0] = self.d[0]*0.5 # start at full euchromatin
        self.x_var[N,0] = self.n_t # start at full NBP
        self.x_var[R,0] = 80 # start at full aslncRNA
        self.t_l = 0 # counter tracks time since previous lysis
        self.max_dt = self.lam / 48 # maximum time step size, forces re-evaluation of probability vector

        # Array trackers
        self.t = t_vec
            # s - array with time values at which to evaluate simulation
        self.x = np.empty( [ *self.x_var.shape, t_vec.shape[0] ] )


    def getRates(self):

        '''
        Calculates event rates according to current system state.
        Returns:
            dictionary with event ID constants as keys and rates as values.
                Includes total rate under 'tot' key.
        '''

        T = self.mu * np.power( np.cos( (self.t_var-self.xi)*np.pi/self.lam ), 2*self.nu )
            # TF concentration at this time

        rates = np.zeros( [ len(self.evt_IDs),N_GEN ] )
            # rate array size of event space

        rates[R_F,:] = (
            ( self.alp * np.power(T,self.h_T) / ( np.power(T,self.h_T) + np.power(self.k_T,self.h_T) ) ) * ( ( self.x_var[E,:] * np.power(self.x_var[N,:],self.h_R) / ( np.power(self.x_var[N,:],self.h_R) + np.power(self.k_R,self.h_R) ) ) + self.kap )
        )

        rates[R_B,:] = (
            self.dlt * self.x_var[R,:]
        )

        rates[E_F,:] = (
            self.eps * self.x_var[N,:] * (self.d - self.x_var[E,:]) * np.power( (self.d - self.x_var[E,:]),self.h_E ) / ( ( np.power( (self.d - self.x_var[E,:]),self.h_E ) + np.power(self.k_E,self.h_E) ) )
        )

        rates[E_B,:] = (
            self.zet * np.sum(self.x_var[S,:]) * self.x_var[E,:] * np.power(self.x_var[E,:],self.h_H) / ( ( np.power(self.x_var[E,:],self.h_H) + np.power(self.k_H,self.h_H) ) )
        )

        rates[N_F,:] = (
            self.gam * self.x_var[R,:] * np.power( (self.n_t - np.sum(self.x_var[N,:]) ),self.h_N ) / ( ( np.power( (self.n_t - np.sum(self.x_var[N,:]) ),self.h_N ) + np.power(self.k_N,self.h_N) ) )
        )

        rates[N_B,:] = (
            self.bet * self.x_var[N,:]
        )

        rates[P_F,:] = (
            self.eta * self.x_var[E,:] * np.power(T,self.h_T) * np.power(self.x_var[N,:],self.h_R) / ( ( np.power(T,self.h_T) + np.power(self.k_T,self.h_T) ) * ( np.power(self.x_var[N,:],self.h_R) + np.power(self.k_R,self.h_R) ) )
        )

        rates[P_B,:] = (
            0
        )

        rates[S_F,:] = (
            self.the * (self.d - self.x_var[E,:]) * np.power(T,self.h_T) / ( np.power(T,self.h_T) + np.power(self.k_T,self.h_T) )
        )

        rates[S_B,:] = (
            self.iot * self.x_var[S,:]
        )

        #print(np.sum(rates,1), T)

        # rate_dict = dict(zip( self.evt_IDs, rates )) # save IDs, rates in dict
        # rate_dict['tot'] = rates.sum() # save total sum of rates

        return rates


    def doAction(self,act,gen):

        '''
        Changes system state variables according to act argument passed (must be
        one of the event ID constants)
        Arguments:
            act : int event ID constant - defines action to be taken
        '''

        if act == R_F:
            self.x_var[R,gen] += 1
        elif act == R_B:
            self.x_var[R,gen] -= 1
        elif act == E_F:
            self.x_var[E,gen] += 1
        elif act == E_B:
            self.x_var[E,gen] -= 1
        elif act == N_F:
            self.x_var[N,gen] += 1
            #self.x_var[R,gen] -= 1
        elif act == N_B:
            self.x_var[N,gen] -= 1
            #self.x_var[R,gen] += 1
        elif act == P_F:
            self.x_var[P,gen] += 1
        elif act == P_B:
            self.x_var[P,:] = 0 # RBC lysis!
        elif act == S_F:
            self.x_var[S,gen] += 1
        elif act == S_B:
            self.x_var[S,gen] -= 1


    def gillespie(self):

        '''
        Simulates a time series with time values specified in argument t_vec
        using the Gillespie algorithm. Stops simulation at maximum time or if no
        change in distance has occurred after the time specified in
        max_no_change. Records position values for the given time values in
        x_vec.
        '''

        # Simulation variables
        self.t_var = self.t[0] # keeps track of time
        self.x[:,:,0] = self.x_var # initialize distance at current distance
        i = 0 # keeps track of index within x and t_vec lists

        while self.t_var < self.t[-1]:
                # repeat until t reaches end of timecourse
            r = self.getRates() # get event rates in this state
            r_tot = np.sum(r) # sum of all rates
            print(self.t_var)
            if 1/self.max_dt < r_tot: # if average time to next event is less than maximum permitted time step,
                # allow it, calculate probability
                # Time handling
                dt = np.random.exponential( 1/r_tot ) # time until next event
                self.t_var += dt # add time step to main timer
                while i < len(self.t)-1 and self.t[i+1] < self.t_var:
                    # while still within bounds and the new time exceeds the next
                    # time in list,
                    i += 1 # move to next time frame in list
                    self.x[:,:,i] = self.x[:,:,i-1]
                        # fill in the state with the previous frame
                    self.t_l += self.t[i] - self.t[i-1] # advance life cycle timer
                    if self.t_l > self.lam: # if over life cycle time,
                        self.t_l = 0 # restart life cycle timer
                        self.doAction(P_B,-1) # lysis

                # Event handling
                if self.t_var < self.t[-1]: # if still within max time
                    u = np.random.random() * r_tot
                        # random uniform number between 0 (inc) and total rate (exc)
                    r_cum = 0 # cumulative rate
                    for e in range(r.shape[0]): # for every possible event,
                        for g in range(r.shape[1]): # for every possible gene,
                            r_cum += r[e,g] # add this event's rate to cumulative rate
                            if u < r_cum: # if random number is under cumulative rate
                                self.doAction(e,g) # do corresponding action
                                self.x[:,:,i] = self.x_var # record distance in list
                                break # exit event loop


                        else: # if the inner loop wasn't broken,
                            continue # continue outer loop

                        break # otherwise, break outer loop



            else: # if no event happening probabilistically in this max permitted time step,
                self.t_var += self.max_dt # add time step to main timer
                while i < len(self.t)-1 and self.t[i+1] < self.t_var:
                    # while still within bounds and the new time exceeds the next
                    # time in list,
                    i += 1 # move to next time frame in list
                    self.x[:,:,i] = self.x[:,:,i-1]
                        # fill in the distance with the previous frame
                    self.t_l += self.t[i] - self.t[i-1] # advance life cycle timer
                    if self.t_l > self.lam: # if over life cycle time,
                        self.t_l = 0 # restart life cycle timer
                        self.doAction(P_B,-1) # lysis

        self.t = self.t[0:i+1] # keep only time points that have been simulated
        self.x = self.x[:,:,0:i+1] # keep only distances that have been simulated
        #self.x = self.x / self.vol # make concentration


# Run all if code is called as a script
if __name__ == '__main__':
    drawFigures()
