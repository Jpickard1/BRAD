import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
# from TwinCell import TwinCell
from copy import deepcopy
import warnings

# This code comes from the Biomarker Selection for Adaptive Systems project
# scr/model.py+
class Model:
    '''
    Model class:
        - type: 'LTI', 'LTV'
        - D: data used to build model
        - states: variables in the model
    '''
    def __init__(self, D=None, modelType=None, states=None):
        self.data = D
        self.type = modelType
        self.states = states
    
    def evaluate(self, x):
        '''
        Evaluate: evaluates the model update and output at a state x.
        
        Params:
        --------------
        x (np.array):
            a model state
        '''
        return self.evaluateState(x), self.evaluateOutput(x)

    def evaluateState(self, x):
        '''
        Evaluates the model update at a state x
        
        Params:
        --------------
        x (np.array):
            a model state
        '''
        return None
    

    def evaluateOutput(self, x):
        '''
        Evaluate: evaluates the model output at a state x.
        
        Params:
        --------------
        x (np.array):
            a model state
        '''
        return None
    
    def obsv(self, T=None):
        """
        Constructs the observability matrix for a given time horizon.

        Args:
        - T (int): Time horizon of the observability matrix. By default, T equals the number of states.

        Returns:
        - np.array: The observability matrix.

        The observability matrix is defined as:

        .. math::

            \\mathcal{O} = \\begin{bmatrix}
            C \\
            CA \\
            CA^2 \\
            \\vdots \\
            CA^{n-1}
            \\end{bmatrix}


        where $C$ is the output matrix and $A$ is the state transition matrix.

        """

        return None

    def observabilityGramian(self):
        '''
        Constructs the observability Gramian matrix.

        Returns:
        - np.array: The observability Gramian matrix.

        Observability Gramian (G):
        The observability Gramian is defined as the integral of the observability matrix transpose multiplied by the observability matrix over an infinite time horizon. It represents the ability of the system to be observed based on its outputs.
        '''
        return None

    def setMeasurements(self, states):
        '''
        Set the measurements for the model output.
        '''
        return None
    
class LinearTimeInvariant(Model):
    """
    Linear Time-Invariant (LTI) model class.

    This class extends the base `Model` class and represents a specific type of model
    with additional parameters specific to LTI systems.

    Params:
    --------------
    data (np.ndarray):
        3 dimensional array (measurements/genes by time by replicates) from which the model will be built

    Attributes:
    --------------
    dmd_res:
        Dictionary containing Dynamic Mode Decomposition (DMD) results for the LTI model.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 29, 2024

    # Order of parameter importance
    #    1. data
    #    2. A
    #    2. DMD
    
    def __init__(self, data=None, A=None, DMD=True, dmdRank=None, states=None, XpXf=None):
        super().__init__(data, 'LTI', states=states)
        self.f = {}
        if data is None:
            self.f['A'] = None
            self.dmd_res = {}
        elif A is not None:
            self.f['A'] = A
            self.dmd_res = None
        elif DMD is True:
            self.dmd_res = dmd(data, rank=dmdRank, XpXf=XpXf)
            self.f['A'] = self.dmd_res['A']
        else:
            print(data)
            print(A)
            print(A is None)
            raise ValueError("Invalid parameters are passed to build the model")

        self.output = {
            'measurements': None,
            'C': np.zeros((1,len(self.states))), # modified so that we can call .evaluate without any ``real" measurements
        }

    def evaluateState(self, x):
        """
        Evaluate state method specific to LinearTimeInvariant model.

        Params:
        --------------
        x (np.array):
            a model state

        Returns:
        --------------
        result: The evaluation result specific to LTI model.
        """
        # Provide your specific evaluation logic here
        # Example: return the dot product of A matrix and the model state x
        return np.dot(self.f['A'], x)

    def evaluateOutput(self, x, t=None):
        """
        Evaluate state method specific to LinearTimeInvariant model.

        Params:
        --------------
        x (np.array):
            a model state(s)

        Returns:
        --------------
        result: The evaluation result specific to LTI model.
        """
        # Provide your specific evaluation logic here
        # Example: return the dot product of A matrix and the model state x
        return self.output['C'] @ x

    def setMeasurments(self, states):
        """
        Set model output and measurement matrix C

        Params:
        --------------
        states:
            a list of model states to measure
        """
        # print('func called')
        if isinstance(states, dict):
            states = states[list(states.keys())[0]]
            warnings.warn("Trying to set time variant sensors on a LinearTimeInvariant model.")
        self.output['measurments'] = states
        self.output['C'] = sp.sparse.csr_matrix((len(states), len(self.states)))
        for i, state in enumerate(states):
            if state not in self.states:
                raise ValueError("Unsupported model state. State: " + str(state) + " is not found in your model.")
            loc = self.states.index(state)
            self.output['C'][i, loc] = 1
            
    def observabilityGramian(self, T=10, reduced=False):
        '''
        Constructs the observability Gramian
        
        Params:
        --------------
        T (int):
            number of time points to use in the finite horizon summation
        reduced (bool, optional):
            if model reduction should be used to compute this
            
        Notes:
        --------------
        currently this is only implemented where reduced is True and needs to be implemented where reduced is False
        '''
        if reduced == False:
            raise ValueError('the non reduced Gramian is not implemented yet. Please add that code')
            return
        C = self.output['C']
        Ar = self.dmd_res['Atilde']
        phiT = np.eye(Ar.shape[0])
        G = np.zeros(Ar.shape)
        for t in range(T):
            half = Ar.T @ self.dmd_res['u_r'].T @ C.T
            G += (half @ half.T)
            phiT = Ar @ phiT
        G = self.dmd_res['u_r'] @ G
        return G

    def gram_matrix(self, T=50, reduced=True, randomIC=False):
        '''
        Compute the Gram matrix for the Linear Time-Invariant (LTI) model.

        Params:
        --------------
        A (np.array):
            Matrix representation of the Koopman operator from DMD
        x0 (np.array):
            Initial conditions from measurements from self.data
        nT (int, optional):
            Number of timepoints over which to compute the Gram matrix (default is 50).
        reduced (bool, optional):
            If True, will compute reduced G from reduced data and KO and will also return full G after inverse projection (default is True).
        projection_matrix (np.array, optional):
            The matrix used to project data and KO to low-dimensional space (first r eigenvectors of Data.T @ Data) (default is an empty array).

        Returns:
        --------------
        G (np.array):
            The Gram matrix for the LTI model.
        Gfull (np.array, optional):
            The full Gram matrix after inverse projection (returned only if reduced is True).
            
        References:
        --------------
        Hasnain, A., Balakrishnan, S., Joshy, D. M., Smith, J., Haase, S. B., & Yeung, E. (2023).
        Learning perturbation-inducible cell states from observability analysis of transcriptome dynamics.
        Nature Communications, 14(1), 3148. [Nature Publishing Group UK London]
        '''
        if self.dmd_res is None:
            x0 = self.data[:, 0, :]
            A = self.f['A']
            A_pow = {
                0: np.eye(N=A.shape[0])
            }
            for t in range(1, T):
                A_pow[t] = A @ A_pow[t-1]
            G = np.zeros_like(A)
            for t in range(T):
                # G += np.matmul(np.matmul(A_pow[t], x0), np.matmul(x0.T, A_pow[t].T))
                G = np.add(G, np.matmul(np.matmul(A_pow[t], x0), np.matmul(x0.T, A_pow[t].T)), casting="unsafe")
            return G
        # otherwise use the DMD reduction
        else:
            x0 = self.dmd_res['data_embedded'][:, 0, :]
            A = self.dmd_res['Atilde']
            if randomIC:
                # Generate artificial initial conditions for robust optimization
                # this code is taken from Aqib's code, and I don't trust it yet, hence the optional arg
                x0min = np.min(x0, axis=1)
                x0max = np.max(x0, axis=1)
                numICs = x0.shape[0]
                x0uni = np.zeros((len(x0min), numICs))
                x0uni[:, 0:x0.shape[1]] = deepcopy(x0)

                for ii in range(x0.shape[1], x0uni.shape[1]):
                    x0tmp = np.random.uniform(x0min, x0max)
                    x0uni[:, ii] = x0tmp
            A_pow = {
                0: np.eye(N=A.shape[0])
            }
            for t in range(1, T):
                A_pow[t] = A @ A_pow[t-1]
            G = np.zeros_like(A)
            for t in range(T):
                G += np.matmul(np.matmul(A_pow[t], x0), np.matmul(x0.T, A_pow[t].T))

            if reduced:
                projection_matrix = self.dmd_res['u_r']
                Gfull = np.matmul(np.matmul(projection_matrix, G), projection_matrix.T)
                return Gfull
            else:
                return G
        
    def gram_matrix_TV(self,T=0,x0=None):
        if self.dmd_res is None:
            if x0 is None:
                x0 = self.data[:,0,:]
            A = self.f['A']
            At = np.linalg.matrix_power(t)
            G = At @ x0 @ x0.T @ At.T
            return G
        else:
            x0 = self.dmd_res['data_embedded'][:, 0, :]
            A = self.dmd_res['Atilde']
            A_pow = {
                0: np.eye(N=A.shape[0])
            }
            for t in range(1, T):
                A_pow[t] = A @ A_pow[t-1]
            G = np.zeros_like(A)
            for t in range(T):
                G += np.matmul(np.matmul(A_pow[t], x0), np.matmul(x0.T, A_pow[t].T))
            projection_matrix = self.dmd_res['u_r']
            Gfull = np.matmul(np.matmul(projection_matrix, G), projection_matrix.T)
            return Gfull
            
        
    def obsv(self, t=None, reduced=False):
        """
        Compute the observability matrix.

        Parameters
        ----------
        t : int, optional
            Number of terms to compute in the observability matrix. Defaults to the number of states (n).

        Returns
        -------
        ob : np.ndarray
            Observability matrix [c; ca; ca^2; ...].
        """
        if (self.output['C'].shape[1] != len(self.states)):
            raise ValueError("Input dimensions are not compatible")
            
        if t is None:
            t = len(self.states)

        c = self.output['C']            
        ny = c.shape[0]
        
        if reduced:
            a = self.dmd_res['Atilde']
            u = self.dmd_res['u_r']
            c = c @ u
        else:
            a = self.f['A']

        # Allocate ob and compute each C A^k term
        n = c.shape[1]
        ob = np.zeros((t * ny, n))
        if isinstance(c, csr_matrix):
            c = c.toarray()

        ob[:ny, :] = c

        for k in range(1, t):
            ob[k * ny:(k + 1) * ny, :] = np.dot(ob[(k - 1) * ny:k * ny, :], a)

        if reduced:
            ob = ob @ u.T

        return ob

class LinearTimeVariant(Model):
    """Linear Time-Variant (LTV) model class

    Args:
        Model (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, times=None, data=None, states=None, dmdRanks=None, LTIs=None, debug=False, reduced=True, u_r=None):
        """_summary_

        Args:
            data (ndarray, optional): time series data. Defaults to None.
            models (dict, optional): dictionary of models at each time. Defaults to None.
            states (list, optional): list of states. Defaults to None.
            times (list, optional): list of time ranges where a single LTI is applied
        """
        # make data vars x time x replicates
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        super().__init__(data, 'LTI', states=states)
            
        if times is None:
            times = range(data.shape[1]+1)
            
        self.reduced = reduced
        if reduced is True and LTIs is None and data is not None:
            dataReshaped = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
            # SVD for DMD
            u, s, vh = np.linalg.svd(dataReshaped)
            if dmdRanks is None:
                rank = getOHT(u, s, vh)
            else:
                rank = dmdRanks[0]
            if rank == -1:
                # print(type(s))
                # print(s.ndim)
                rank = s.shape[0]
            self.u_r = u[:, 0:rank] # truncate to rank-r
            self.phiR = {}
            self.phiR[0] = np.eye(rank)
            
        # make a LTI for each time range
        self.LTI = {} # each time range has a single LTI model
        self.phi = {} # set of state transition matrices from 0
        if data is not None:
            self.phi[0]=np.eye(data.shape[0])
        if LTIs is not None:
            self.LTI=LTIs
            self.states = self.LTI[list(self.LTI.keys())[0]].states
            self.phiR = {}
            if reduced:
                self.u_r = LTIs[0].dmd_res['u_r']
                self.phiR[0] = np.eye(LTIs[0].dmd_res['Atilde'].shape[0])
            else:
                self.phi[0] = np.eye(len(self.states))
        else:
            for i, t0 in enumerate(times):
                if i+1 >= len(times):
                    break
                t1 = times[i+1]
                # determine if do DMD or DGC (DMD if distance is greater than 1)
                if t1-t0>1:
                    if debug:
                        print(t0)
                        print(t1)
                        print(data[:, t0:t1+1, :])
                        print(data[:, t0:t1+1, :].shape)
                    self.LTI[t0] = LinearTimeInvariant(data=data[:, t0:t1+1, :], states=states, DMD=True, dmdRank=dmdRanks[i])
                    self.phiR[0] = np.eye(self.LTI[t0].dmd_res['Atilde'].shape[0])
                    if self.u_r.shape[1] > self.LTI[t0].dmd_res['u_r'].shape[1]:
                        self.u_r = self.u_r[:, :self.LTI[t0].dmd_res['u_r'].shape[1]]
                else:
                    dmd_res = None
                    A = DGC(data[:, t0:t1+1, :], debug=debug)
                    if len(list(A.keys()))==0:
                        break
                    else:
                        A = A[list(A.keys())[0]]
                    if debug:
                        print('Data=')
                        print(data[:, t0:t1+1, 0])
                        print('A(t)=')
                        print(A)
                    self.LTI[t0] = LinearTimeInvariant(data=data[:, t0:t1+1, :], A=A, states=states, DMD=False)
                    if reduced is True:
                        self.LTI[t0].dmd_res = {}
                        self.LTI[t0].dmd_res['Atilde'] = self.u_r.T @ A @ self.u_r

    def getLTImodelKey(self, t):
        """get ket to Linear Time Invariant model at time t

        Args:
            t (int): time

        Returns:
            _type_: a key mapping to the LTI model at time t
        """
        largest_key = None
        for key in self.LTI.keys():
            if key <= t and (largest_key is None or key > largest_key):
                largest_key = key
        return largest_key

    def evaluateState(self, x, t):
        tModel = self.getLTImodelKey(t)
        # print('t',t)
        # print('tModel',tModel)
        return self.LTI[tModel].evaluateState(x)
    
    def evaluateOutput(self, x, t=0):
        tModel = self.getLTImodelKey(t)
        return self.LTI[tModel].evaluateOutput(x)

    def setMeasurments(self, states):
        for t, sensors in states.items():
            # print('call func')
            self.LTI[t].setMeasurments(sensors)
            
    def observabilityGramian(self, T=10, reduced=False):
        if reduced == False:
            raise ValueError('the non reduced Gramian is not implemented yet. Please add that code')
            return
        G = np.zeros((len(self.states), len(self.states)))
        for t in range(T):
            tModel = self.getLTImodelKey(t)
            phiT = self.getPhi(t1=0, t2=t, reduced=reduced)
            C = self.LTI[tModel].output['C']
            half = self.u_r @ phiT.T @ self.u_r.T @ C.T
            G += (half @ half.T)
        return G
            
    def gram_matrix(self, T=0, x0=None, reduced=True):
        if x0 is None:
            x0 = self.data[:, 0, :]
        xt = x0
        tModel = self.getLTImodelKey(T)
        G = np.zeros_like(self.LTI[tModel].f['A'])            
        for t in range(T):
            tModel = self.getLTImodelKey(t)
            phiT = self.getPhi(t1=0, t2=t, reduced=reduced)
            if reduced:
                half = self.u_r @ phiT @ self.u_r.T @ xt
            else:
                half = phiT @ xt
            print(half.shape)
            G += (half @ half.T)
            # xt = self.LTI[tModel].f['A'] @ xt
            # G += (xt @ xt.T)
        return G
        
    def getPhi(self, t1=0, t2=1, reduced=False):
        """
        phi(t1,t2) returns the state transition matrix from time point t1 to t2
        """
        if t1 != 0:
            raise ValueError('LTV model storage of phi is implemented for t1=0. Please modify the Model.LTV class to change this')
        largest_key_smaller_than_t2 = 0
        for key in self.phi.keys():
            if key < t2 and (largest_key_smaller_than_t2 is None or key > largest_key_smaller_than_t2):
                largest_key_smaller_than_t2 = key
        # print('t2=',t2)
        # print('largest_key_smaller_than_t2=',largest_key_smaller_than_t2)
        if not reduced:
            for t in range(largest_key_smaller_than_t2+1, t2+1):
                tModel = self.getLTImodelKey(t-1)
                # print('tModel:',tModel)
                A = self.LTI[tModel].f['A']
                self.phi[t] = A @ self.phi[t-1]
            return self.phi[t2]
        else:
            for t in range(largest_key_smaller_than_t2+1, t2+1):
                tModel = self.getLTImodelKey(t-1)
                A = self.LTI[tModel].dmd_res['Atilde']
                self.phiR[t] = A @ self.phiR[t-1]
            return self.phiR[t2]
        
    def gram_matrix_TV(self,T=0,x0=None):
        if x0 is None:
            x0 = self.data[:,0,:]
        if self.reduced is True:
            x0 = self.u_r.T @ x0
        phi = self.getPhi(t1=0, t2=T, reduced=self.reduced)
        G = phi @ x0 @ x0.T @ phi.T
        if self.reduced:
            G = self.u_r @ G @ self.u_r.T
        return G


    def obsv(self, t=None, reduced=False, debug=False):
        """
        Compute the observability matrix.

        Parameters
        ----------
        t : int, optional
            Number of terms to compute in the observability matrix. Defaults to the number of states (n).

        Returns
        -------
        ob : np.ndarray
            Observability matrix [c; ca; ca^2; ...].
        """
        # set time points
        if t is None:
            t = len(self.states)
            
        # determine the number of measurements
        sumMeasurements = 0
        numMeasurements = {}
        for i in range(t):
            sumMeasurements += self.LTI[i].output['C'].shape[0]
            numMeasurements[i] = self.LTI[i].output['C'].shape[0]

        # get number of states and reduction map
        if reduced:
            u = self.u_r
            n = self.u_r.shape[1] # model reduced states
        else:
            n = len(self.states)  # number of true states
            
        # allocate space to store Observability matrix
        ob = np.zeros((sumMeasurements, n))

        # save the first time point
        if reduced:
            ob[:numMeasurements[0], :] = self.LTI[0].output['C'] @ u
        else:
            ob[:numMeasurements[0], :] = self.LTI[0].output['C']
            
        if debug:
            print(ob)
        
        # save all other time points
        start = numMeasurements[0]
        for k in range(1, t):
            tModel = self.getLTImodelKey(k)
            if debug:
                print('kth iter of obsv: ', k)
                print(tModel)
            c = self.LTI[tModel].output['C']
            if reduced:
                c = c @ u
            phi = self.getPhi(t1=0, t2=k, reduced=reduced)
            out = np.dot(c, phi)
            ob[start:start+c.shape[0] ,:] = out
            start = start+c.shape[0]
            if debug:
                print('k=',k)
                print('c=')
                print(c)
                print('phi(0, ' + str(k) + ') =')
                print(phi)
                print('ob=')
                print(ob[:start,:])
                print('==================')
                print(' ')
                print('==================')

        if reduced:
            ob = ob @ u.T
            
        return ob
    
def DGC(data, debug=False):
    """Data Guided Control Time Varying Linear Model

    Notes: Need to modify how the number of replicates is accounted for. 
    Currently the denominator is a sum over all replicates, but we may want 
    to change this.

    Args:
        data (ndarray): n x T x replicates data array
    """
    if debug:
        print('DGC data.shape=' + str(data.shape))
    n, T, replicates = data.shape
    A = {}
    for t in range(T-1):
        xt = data[:, t, :]
        xtp1 = data[:, t + 1, :]
        A[t] = np.eye(n) + (xtp1 - xt) @ xt.T / np.sum(xt.T @ xt)
    return A

def tvDMD(data, dmdArgs):
    """Time Varying Dynamic Mode Decomposition

    Args:
        data (ndarray): n x T x replicates data array
        dmdArgs (dict): arguments for DMD commands
            DMDranges (list): list of time points separating where a linear model should be trained on
                ex. [0, 4, 10, 12] will fit 3 linear models (one predicting flow from time t=0 until t=4,
                on for the flow from t=4 to t=10, and one for the flow from t=10 to t=12)
    """
    n, T, replicates = data.shape
    A = {}
    DMD_RES = {}
    for f in range(len(dmdArgs['ranges']) - 1):
        t0 = dmdArgs['ranges'][f]
        t1 = dmdArgs['ranges'][f + 1]
        fData = data[:, t0:t1, :]
        DMD_RES[t0] = dmd(fData, rank=dmdArgs['rank'])
        A[t0] = DMD_RES[t0]['A']
    return A, DMD_RES


class KalmanFilter(Model):
    def __init__(self, model):
        if not isinstance(model, TwinCell.Model.LinearTimeInvariant) and not isinstance(model, TwinCell.Model.LinearTimeVariant):
            raise ValueError("Unsupported model type. Supported types: LinearTimeInvariant, LinearTimeVariant")
        super().__init__(D=model.D, type=model.type, states=model.states)

################################
#### Helper functions below ####
################################

def dmd_reshape(data):
    """
    Utility function to reshape the data for Dynamic Mode Decomposition (DMD) as described by Hasnain et al.

    Params:
    --------------
    data (np.array):
        An array of shape (genes, timepoints, replicates).

    Returns:
    --------------
    Xp (np.array):
        The first m-1 timepoints for all replicates.

    Xf (np.array):
        The last m-1 timepoints for all replicates.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    n, m, r = data.shape

    Xp = data[:,:-1].reshape(n, (m-1)*r, order='F')
    Xf = data[:,1:].reshape(n, (m-1)*r, order='F')

    return Xp, Xf

def getOHT(u, s, vh):
    """
    Compute the optimal hard threshold from the Singular Value Decomposition (SVD) of the data.
    NOTE: This function assumes a tall, skinny matrix.

    Params:
    --------------
    u (np.array):
        Left singular vectors.
    s (np.array):
        Diagonal matrix of singular values.
    vh (np.array):
        Right singular vectors transposed.

    Returns:
    --------------
    oht (int):
        The index of the optimal hard threshold for a non-square matrix with unknown noise level.
    """
    n = u.shape[0]
    m = vh.shape[0] 
    
    beta = m / n
    omega = (0.56*beta**3) - (0.95 * beta**2) + (1.82 * beta) + 1.43
    y_med = np.median(s)
    tau = omega * y_med
    s_ind = np.argwhere(s >= tau)
    oht = np.max(s_ind) 
    return oht

def embed_data(data, u, rank):
    """A utility function to embed the data based on the 
    low-rank approximation of Xp 
    
    Params:
    --------------
    data (np.array):
        An array of shape (genes, timepoints, replicates)
    u (np.array):
        The left singular vectors of Xp
    rank (int):
        The rank truncation for u
    
    Returns:
    --------------  
    data_embedded (np.array):
        The embedded data
    """
    u_r = u[:, 0:rank] # truncate to rank-r
    n, m, r = data.shape
    
    data_embedded = np.zeros((rank, m, r))

    for i in range(r):
        data_embedded[:,:,i] = np.dot(u_r.T, data[:,:,i])
    return data_embedded
    
def dmd(data, rank=None, XpXf=None):
    """
    Compute Dynamic Mode Decomposition (DMD) of the data based on Hasnain et al. 2023.

    This function calculates the DMD using the given data and optionally truncates the Singular Value Decomposition (SVD)
    using the optimal hard threshold if `rank` is set to `None`. If `rank` is -1, it computes the exact DMD.

    Params:
    --------------
    data (np.array):
        An array of shape (genes, timepoints, replicates).
    rank (int or None):
        If `None`, the truncated SVD will be computed using the optimal hard threshold.
        If -1, the exact DMD will be computed.

    Returns:
    --------------
    dmd_res (dict):
        A dictionary containing DMD results, including:
        - 'A': Full model matrix,
        - 'Atilde': Low-rank dynamics matrix,
        - 'rank': Truncated rank,
        - 'u_r': Left singular vectors (truncated),
        - 'SVD': List containing [u, s, vh] from SVD,
        - 'L': Eigenvalues of Atilde,
        - 'W': Eigenvectors of Atilde,
        - 'data_embedded': Embedded data in the eigenbasis,
        - 'Phi': DMD modes,
        - 'Phi_hat': DMD modes (alternative computation),
        - 'amplitudes': DMD amplitudes,
        - 'n': Number of genes,
        - 'm': Number of timepoints,
        - 'r': Number of replicates.
    """

    n, m, r = data.shape
    # RESHAPE DATA
    if XpXf is None:
        Xp, Xf = dmd_reshape(data)
    else:
        Xp, Xf = XpXf[0], XpXf[1]

    # SVD for DMD
    u, s, vh = np.linalg.svd(Xp)

    if rank == -1:
        rank = s.shape[0]
    elif rank == None: 
        rank = getOHT(u, s, vh)

    # perform DMD
    u_r = u[:, 0:rank] # truncate to rank-r
    s_r = s[0:rank]
    vh_r = vh[0:rank, :]
    Atilde = u_r.T @ Xf @ vh_r.T @ np.diag(1/s_r) # low-rank dynamics
    A = u_r@Atilde@u_r.T

    # DMD EIGENVALUES AND EIGENVECTORS
    L, W = np.linalg.eig(Atilde)

    # DMD MODES
    Phi = Xf @ vh_r.T @ np.diag(1/s_r) @ W
    Phi_hat = np.dot(u_r, W)

    # DMD AMPLITUDES: X0 in the eigenvector (of A) basis
    data_embedded = embed_data(data, u, rank)
    amps = []
    for i in range(r):
        b_ri = np.linalg.inv(np.dot(W, np.diag(L))) @ data_embedded[:,:,i]
        amps.append(b_ri)

    return {
        'A' : A,
        'Atilde' : Atilde,
        'rank' : rank,
        'u_r' : u_r,
        'SVD' : [u, s, vh],
        'L' : L,
        'W' : W,
        'data_embedded' : data_embedded,
        'Phi' : Phi,
        'Phi_hat' : Phi_hat,
        'amplitudes' : amps,
        'n' : n,
        'm' :  m,
        'r' : r,
    }

def fit_sparse(data, S):
    """
    Estimates the sparse matrix A that generated data

    Parameters:
    data (ndarray): The input data matrix of shape (n, t), where n is the dimensionality
                    and t is the number of time steps.
    S (ndarray): The sparse pattern matrix indicating the known entries of A.
                 Nonzero elements indicate known entries, while zero elements indicate unknown entries.
                 Should have the same shape as A.

    Returns:
    A (ndarray): The estimated sparse matrix A.
    fval (float): The value of the objective function at the minimum (optimal) solution.
    """
    # Get dimensions of the data
    n, t = data.shape
    
    # Extract Xp and Xm from data
    Xp = data[:, 1:]
    Xm = data[:, :-1]
    
    # Calculate v, Q, and b
    v = np.dot(Xm, Xp.T)
    Q = np.kron(np.eye(n), np.dot(Xm, Xm.T))
    b = -2 * v.flatten()
    
    # Build E
    E_indices = np.where(S == 0)
    E = csr_matrix((np.ones_like(E_indices[0]), E_indices), shape=(np.count_nonzero(S), n**2))
    
    # Define the objective function for optimization
    def objective(a):
        return 0.5 * np.dot(a, np.dot(Q, a)) + np.dot(b, a)
    
    # Initial guess for optimization
    x0 = np.zeros(n**2)
    
    # Perform quadratic programming
    result = minimize(objective, x0, constraints={'type': 'eq', 'fun': lambda a: E.dot(a)}, options={'disp': True})
    
    # Extract optimized solution
    a_opt = result.x
    
    # Reshape a_opt to get A
    A = np.reshape(a_opt, (n, n))
    
    return A, result.fun

def exact_dmd(data):
    """
    Compute the exact Dynamic Mode Decomposition (DMD) of the data.

    This method computes the DMD without truncating the Singular Value Decomposition (SVD).
    It directly computes the full model A, eigenvalues L, and eigenvectors W.

    Params:
    --------------
    data (np.array):
        An array of shape (genes, timepoints, replicates).

    Returns:
    --------------
    dmd_res (dict):
        A dictionary containing exact DMD results, including:
        - 'A': Full model matrix,
        - 'L': Eigenvalues,
        - 'W': Eigenvectors,
        - 'n': Number of genes,
        - 'm': Number of timepoints,
        - 'r': Number of replicates.
    """

    n, m, r = data.shape
    Xp, Xf = dmd_reshape(data)

    A = Xf @ np.linalg.pinv(Xp)  # full model A
    L, W = np.linalg.eig(A)
    return {
        'A' : A,
        'L' : L,
        'W' : W,
        'n' : n,
        'm' :  m,
        'r' : r,
    }
    
