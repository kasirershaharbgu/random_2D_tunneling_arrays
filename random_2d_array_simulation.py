__author__ = 'Shahar Kasirer'

# Distributed under GNU GENERAL PUBLIC LICENSE version 2.0, see LICENSE

# Environment imports
import os
os.environ["OPENBLAS_NUM_THREADS"] = "5"  # Number of threads used for EACH simulation instance.
from time import sleep
from multiprocessing import Pool
from copy import copy

# Mathematical tools
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy.integrate import cumtrapz
from mpmath import quad, exp, sqrt, fabs, re, inf, ninf, mp
from scipy.interpolate import interp1d

# Graphical tools
import matplotlib
matplotlib.use("Agg")  # To avoid showing plots during a run on server.
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Parsing tools
from optparse import OptionParser
import re as regex
from ast import literal_eval

#  Constants  #
EPS = 0.0001
#  Gillespie Constants  #
MIN_STEPS = 10
STEADY_STATE_VAR = 1e-4
ALLOWED_ERR = 1e-3
STEADY_STATE_REP = 100
INV_DOS = 0.01
#  Tau Leaping Constants #
TAU_EPS = 0.03
#  Graph Constants  #
DQ = 0.1
Q_SHIFT = 10
GRAD_REP = 100
INI_LR = 0.001


#  Helping Methods  #
def flattenToColumn(a):
    """
    Returns the given array, reshaped into a column array.
    :param a: (N,M) numpy array.
    :return: (N*M,1) numpy array.
    """
    return a.reshape((a.size, 1))


def flattenToRow(a):
    """
    Returns the given array, reshaped into a row array.
    :param a: (N,M) numpy array.
    :return: (1,N*M) numpy array.
    """
    return a.reshape((1, a.size))


def detect_local_minima(arr):
    # From https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local minimum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood minimum, 0 otherwise)
    """
    def local_min_func(input):
        mid = len(input) // 2
        return (input[mid] <= input).all()
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    return np.where(filters.generic_filter(arr, local_min_func, footprint=neighborhood,
                                           mode='constant', cval=np.min(arr)-1))


def simple_gradient_descent(grad, x0, eps=1e-4, lr=1e-3, max_iter=1000000, plot_lc=True):
    """
    Performs a gradient descent for a given function to detect the nearest local minima.
    :param grad: A method that gets an array with the same shape as x0 and calculates the gradient in this point.
    :param x0: Initial point.
    :param eps: A point will be considered a local minima if the gradient in this point is smaller than eps.
    :param lr: Learning rate. At each step x -> x - lr*grad(x)
    :param max_iter: Maximum iterations
    :param plot_lc: If true, the learning curve will be plotted (gradient vs. iteration number).
    :return: x, success_flag. x - Detected local minima (or location in last iteration if max_iteration has
     been reached)
     success_flag - True if run terminated successfully.
    """
    x = x0.flatten()
    curr_grad = grad(x)
    if plot_lc:
        gradvec = []
    iter=0
    while np.max(np.abs(curr_grad)) > eps and iter < max_iter:
        if plot_lc:
            gradvec.append(curr_grad)
        x = x - curr_grad * lr
        curr_grad = grad(x)
        iter += 1
    if plot_lc:
        plt.figure()
        plt.plot(gradvec)
        plt.show()
    return x, iter < max_iter


#  Methods for calculating superconducting related tunneling rates  #
#  see https://doi.org/10.1007/978-1-4757-2166-9_2  #
def high_impedance_p(x, Ec, T, kappa):
    """
    P- function for high impedance.
    :param x: function input (energy).
    :param Ec: Electrostatic energy of environment.
    :param T: Temperature.
    :param kappa: Carrier charge in units of e.
    :return: P(x)
    """
    sigma = 2*(kappa**2)*Ec*T
    mu = kappa**2*Ec
    return exp(-(x-mu)**2/(2*sigma))/sqrt(2*np.pi*sigma)


def fermi_dirac_dist(x, T):
    """
    Fermi-dirac distribution function.
    :param x: Energy.
    :param T: Temperature (energy units).
    :return: f(x).
    """
    exp_arg = x/T
    if exp_arg > 20:
        return 0
    else:
        return 1/(1 + exp(x/T))


def qp_density_of_states(x, energy_gap):
    """
    Normalized Quasi-particles density of states (BCS)
    :param x: energy.
    :param energy_gap: Superconducting gap.
    :return: Ns(x)/N0(x).
    """
    arg = x**2-energy_gap**2
    return fabs(x) / sqrt(arg)


def cp_tunneling(x, Ec, Ej, T):
    """
    Cooper-pairs tunneling rates for high impedance.
    :param x: tunneling energy differences array (before - after).
    :param Ec: Electrostatic energy of environment.
    :param Ej: Josephson energy.
    :param T: Temperature.
    :return: Array with same shape as x.
    """
    res = np.zeros(x.shape)
    for ind, val in enumerate(x):
        res[ind] = (np.pi*Ej)**2 * high_impedance_p(val, Ec, T, 2)
    return res


def qp_tunneling(deltaE, Ec, gap, T):
    """
    Quasi-particles tunneling rates for high impedance.
    :param deltaE: tunneling energy differences array (before - after).
    :param Ec: Electrostatic energy of environment.
    :param gap: Superconducting gap.
    :param T: Temperature.
    :return: Array with same shape as deltaE.
    """
    def qp_integrand(dE):
        def f(x1, x2):
            return qp_density_of_states(x1, gap) * qp_density_of_states(x2, gap) * fermi_dirac_dist(x1, T) \
                   * (1 - fermi_dirac_dist(x2, T)) * high_impedance_p(x1 - x2 + dE, Ec, T, 1)
        return f

    def qp_tunneling_single(dE):
        if fabs(deltaE) < 10 * gap:
            mp.dps = 50
            part1 = quad(qp_integrand(dE), [ninf, -gap], [ninf, -gap])
            part2 = quad(qp_integrand(dE), [ninf, -gap], [gap, inf])
            part3 = quad(qp_integrand(dE), [gap, inf], [ninf, -gap])
            part4 = quad(qp_integrand(dE), [gap, inf], [gap, inf])
            mp.dps = 15
            return re(part1 + part2 + part3 + part4)
        elif deltaE > 0:
            return deltaE - Ec
        else:
            return 0

    res = np.zeros(deltaE.shape)
    for ind, val in enumerate(deltaE):
        res[ind] = qp_tunneling_single(val)
    res[res < 0] = 0
    return res


class TunnelingRateCalculator:
    """
    Calculates tunneling rates and saves them for later use (or loads existing rates).
    """
    def __init__(self, deltaEmin, deltaEmax, deltaEstep, rateFunc, Ec, T, otherParam, dirPath):
        """
        :param deltaEmin, deltaEmax, deltaEstep: Rates are calculated for energy differences between given minimum
         and maximum with the given steps (and interpolated in between).
        :param rateFunc: Function for calculating tunneling rates, with signature f(deltaE, Ec, otherParameter, T).
        :param Ec: Electrostatic energy of environment.
        :param T: Temperature.
        :param otherParam: Superconducting gap (for quasi-particles) or Josephson energy (for Cooper-pairs).
        :param dirPath: Path to directory where rates will be saved (or loaded from).
        """
        self.T = T
        self.otherParam = otherParam
        self.Ec = Ec
        self.dirName = os.path.join(dirPath, "tunneling_rate_" + str(self.T) + "_" + str(self.otherParam))
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)
        self.deltaEmin = deltaEmin
        self.deltaEmax = deltaEmax
        self.deltaEstep = deltaEstep
        self.rateFunc = rateFunc
        self.deltaEvals = None
        self.vals = None
        self.set_results()
        self.set_approx()

    def isWriting(self):
        """
        Checking of writing "mutex" is occupied
        """
        if os.path.exists(os.path.join(self.dirName, "writing.txt")):
            writing_process_still_alive = False
            try:
                with open(os.path.join(self.dirName, "writing.txt"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        pid = int(line)
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            continue
                        else:
                            writing_process_still_alive = True
                return writing_process_still_alive
            except FileNotFoundError:
                return False
        else:
            return False

    def getWritingLock(self):
        """
        Getting writing "mutex".
        """
        got_it = False
        while not got_it:
            while self.isWriting():
                sleep(60)
            with open(os.path.join(self.dirName, "writing.txt"), "a") as f:
                f.write(str(os.getpid()) + "\n")
            got_it = True
            try:
                with open(os.path.join(self.dirName, "writing.txt"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        pid = int(line)
                        if pid < os.getpid():
                            got_it = False
                            sleep(60)
            except FileNotFoundError:
                got_it = False

    def freeWritingLock(self):
        """
        Freeing writing "mutex".
        :return:
        """
        if self.isWriting():
            os.remove(os.path.join(self.dirName, "writing.txt"))

    def set_results(self):
        """
        Setting initial state
        """
        if os.path.isdir(self.dirName):
            desireddeltaEmin = self.deltaEmin
            desireddeltaEmax = self.deltaEmax
            self.getWritingLock()
            self.deltaEmin = np.load(os.path.join(self.dirName, "deltaEmin.npy"))
            self.deltaEmax = np.load(os.path.join(self.dirName, "deltaEmax.npy"))
            self.deltaEstep = np.load(os.path.join(self.dirName, "deltaEstep.npy"))
            self.deltaEvals = np.arange(self.deltaEmin, self.deltaEmax, self.deltaEstep)
            self.vals = np.load(os.path.join(self.dirName, "vals.npy"))
            self.freeWritingLock()
            while self.deltaEmin - desireddeltaEmin >= -self.deltaEstep:
                self.getWritingLock()
                self.decrease_low_limit()
                self.freeWritingLock()
            while self.deltaEmax - desireddeltaEmax <= self.deltaEstep:
                self.getWritingLock()
                self.increase_high_limit()
                self.freeWritingLock()
        else:
            os.mkdir(self.dirName)
            self.getWritingLock()
            self.deltaEvals = np.arange(self.deltaEmin, self.deltaEmax, self.deltaEstep)
            self.vals = self.rateFunc(self.deltaEvals, self.Ec, self.otherParam, self.T)
            self.saveVals()
            self.freeWritingLock()
        return True

    def saveVals(self):
        """
        Saving calculated rates
        """
        np.save(os.path.join(self.dirName, "deltaEmin"), self.deltaEmin)
        np.save(os.path.join(self.dirName, "deltaEmax"), self.deltaEmax)
        np.save(os.path.join(self.dirName, "deltaEstep"), self.deltaEstep)
        np.save(os.path.join(self.dirName, "vals"), self.vals)

    def set_approx(self):
        """
        Setting interpolation for calculated rates.
        """
        self.approx = interp1d(self.deltaEvals, self.vals, assume_sorted=True)

    def increase_high_limit(self):
        """
        Calculating rates for higher energy differences than the ones currently exist.
        """
        print("Increasing high limit", flush=True)
        new_deltaEmax = self.deltaEmax + np.abs(self.deltaEmax)
        new_inputs = np.arange(self.deltaEmax, new_deltaEmax, self.deltaEstep)
        new_vals = self.rateFunc(new_inputs, self.Ec, self.otherParam, self.T)
        self.deltaEmax = new_deltaEmax
        self.deltaEvals = np.hstack((self.deltaEvals, new_inputs))
        self.vals = np.hstack((self.vals, new_vals))
        self.saveVals()
        self.set_approx()
        print("High limit increased, Emax= " + str(self.deltaEmax), flush=True)

    def decrease_low_limit(self):
        """
        Calculating rates for lower energy differences than the ones currently exist.
        """
        print("Decreasing low limit")
        new_deltaEmin = self.deltaEmin - np.abs(self.deltaEmin)
        new_inputs = np.arange(new_deltaEmin, self.deltaEmin, self.deltaEstep)
        new_vals = self.rateFunc(new_inputs, self.Ec, self.otherParam, self.T)
        self.deltaEmin = new_deltaEmin
        self.deltaEvals = np.hstack((new_inputs, self.deltaEvals))
        self.vals = np.hstack((new_vals, self.vals))
        self.saveVals()
        self.set_approx()
        print("Low limit decreased, Emin= " + str(self.deltaEmin), flush=True)

    def update_rates(self):
        """
        Updating rates by loading existing results.
        """
        self.getWritingLock()
        self.deltaEmin = np.load(os.path.join(self.dirName, "deltaEmin.npy"))
        self.deltaEmax = np.load(os.path.join(self.dirName, "deltaEmax.npy"))
        self.deltaEstep = np.load(os.path.join(self.dirName, "deltaEstep.npy"))
        self.deltaEvals = np.arange(self.deltaEmin, self.deltaEmax, self.deltaEstep)
        self.vals = np.load(os.path.join(self.dirName, "vals.npy"))
        self.set_approx()
        self.freeWritingLock()

    def get_tunneling_rates(self, deltaE):
        """
        Returns tunneling rates for the given energy differences (before - after).
        :param deltaE: 1D numpy array.
        :return: Numpy array with same size as deltaE.
        """
        if self.deltaEmin - np.min(deltaE) >= -self.deltaEstep or self.deltaEmax - np.max(deltaE) <= self.deltaEstep:
            self.update_rates()
        while self.deltaEmin - np.min(deltaE) >= -self.deltaEstep:
            self.getWritingLock()
            self.decrease_low_limit()
            self.freeWritingLock()
        while self.deltaEmax - np.max(deltaE) <= self.deltaEstep:
            self.getWritingLock()
            self.increase_high_limit()
            self.freeWritingLock()
        return self.approx(deltaE)

    def plot_rate(self, fig, ax, fmt):
        """
        Plotting the calculated rates (for dbg).
        :return:
        """
        ax.plot(self.deltaEvals, self.vals, fmt)
        return fig, ax


class DotArray:
    """
    An array of quantum dots connected to external voltage from left to
    right and to gate voltage
    """

    def __init__(self, rows, columns, VL, VR, VU, VD, VG, Q0, n0, CG, RG, Ch, Cv, Rh, Rv, temperature,
                 temperature_gradient, leftElectrode, rightElectrode, upElectrode, downElectrode,
                 fastRelaxation=False, tauLeaping=False, modifyR=False):
        """
        Creates new array of quantum dots
        :param rows: number of rows (int)
        :param columns: number of columns (int)
        :param VL: left electrode voltage (double)
        :param VR: right electrode voltage (double)
        :param VU: top electrode voltage (double)
        :param VD: bottom electrode voltage (double)
        :param Q0: np array of initial charges (rowsXcolumns double array)
        :param n0: np array of initial electrons (rowsXcolumns double array)
        :param VG: voltage of the gates (rowsXcolumns double array)
        :param Ch: horizontal capacitance ((rowsXcolumns+1 double array)
        :param Cv: vertical capacitance (rows-1Xcolumns double array)
        :param Rh: vertical tunnelling resistance (rowsXcolumns+1 double array)
        :param Rv: horizontal tunnelling resistance (rowsXcolumns+1 double array)
        :param RG: gate resistances (rowsXcolumns double array)
        :param temperature: temperature on the left side of the array (double)
        :param temperature_gradient: constant temperature gradient, from left to right (double)
        :param left/right/up/downElectrode: location of electrodes
         ((rows,) size array with 1 where electrode is connected and 0 otherwise)
        :param fastRelaxation: True for working in the fast relaxation limit.
        :param tauLeaping: True to use tau leaping approximation.
        :param modifyR: True to modify tunneling resistance according to the occupation (finite DOF for electrons).
        """
        self.fast_relaxation = fastRelaxation
        self.rows = rows
        self.columns = columns
        self.VL = VL
        self.VR = VR
        self.VU = VU
        self.VD = VD
        self.leftElectrode = np.array(leftElectrode, dtype=np.bool)
        self.rightElectrode = np.array(rightElectrode, dtype=np.bool)
        self.upElectrode = np.array(upElectrode, dtype=np.bool)
        self.downElectrode = np.array(downElectrode, dtype=np.bool)
        self.VG = flattenToColumn(VG)
        self.Q = flattenToColumn(Q0)
        self.n = flattenToColumn(n0)
        self.CG = flattenToColumn(CG)
        self.RG = flattenToColumn(RG)
        self.Ch = Ch
        self.Cv = Cv
        self.Rh = Rh
        self.Rv = Rv if Rv.size else np.zeros((0, 0))
        self.R = np.hstack((Rh.flatten(), Rh.flatten(), Rv.flatten(), Rv.flatten()))
        self.temperature = temperature if temperature_gradient == 0 else \
            self.getTemperatureArray(temperature, temperature_gradient)
        self.rightWorkLen = (self.columns + 1) * self.rows
        self.downWorkLen = self.columns * (self.rows + 1)
        self.no_tunneling_next_time = False
        self._Q_eigenbasis = None
        self.createCapacitanceMatrix()
        self.setDiagonalizedInvTau()
        self.setConstWork()
        self.setConstMatrix()
        self.setConstNprimePart()
        self.use_modifyR = modifyR

        # for variable R calculation
        if self.use_modifyR:
            self._leftnExponent = np.ones((self.rows, self.columns+1))
            self._rightnExponent = np.ones((self.rows, self.columns + 1))
        # Memory initialization for tau leaping
        self.tauLeaping = tauLeaping
        if tauLeaping:
            self.totalAction = np.zeros((self.rows, self.columns))
            self.right = np.zeros((self.rows, self.columns + 1))
            self.left = np.zeros((self.rows, self.columns + 1))
            self.down = np.zeros((self.rows + 1, self.columns))
            self.up = np.zeros((self.rows + 1, self.columns))

    def __copy__(self):
        """
        Copy constructor. Shallow copy for static constant variables. Deep copy for dynamical variables.
        :return: A copy of the current DotArray instance.
        """
        copy_array = object.__new__(type(self))
        copy_array.fast_relaxation = self.fast_relaxation
        copy_array.rows = self.rows
        copy_array.columns = self.columns
        copy_array.VL = self.VL
        copy_array.VR = self.VR
        copy_array.VU = self.VU
        copy_array.VD = self.VD
        copy_array.VG = self.VG
        copy_array.leftElectrode = self.leftElectrode
        copy_array.rightElectrode = self.rightElectrode
        copy_array.upElectrode = self.upElectrode
        copy_array.downElectrode = self.downElectrode
        copy_array.no_connections = self.no_connections
        copy_array.Q = np.copy(self.Q)
        copy_array.n = np.copy(self.n)
        copy_array.CG = self.CG
        copy_array.RG = self.RG
        copy_array.Ch = self.Ch
        copy_array.Cv = self.Cv
        copy_array.Rh = self.Rh
        copy_array.Rv = self.Rv
        copy_array.R = np.copy(self.R)
        copy_array.temperature = self.temperature
        copy_array.rightWorkLen = self.rightWorkLen
        copy_array.downWorkLen = self.downWorkLen
        copy_array.no_tunneling_next_time = self.no_tunneling_next_time
        copy_array.invC = self.invC
        copy_array.invCeq = self.invCeq
        copy_array._InvTauEigenVectors = self._InvTauEigenVectors
        copy_array._InvTauEigenValues = self._InvTauEigenValues
        copy_array._InvTauEigenVectorsInv = self._InvTauEigenVectorsInv
        copy_array.timeStep = self.timeStep
        copy_array.default_dt = self.default_dt
        copy_array._constQnPart = self._constQnPart
        copy_array._matrixQnPart = self._matrixQnPart
        copy_array.commonHorz = self.commonHorz
        copy_array.commonVert = self.commonVert
        copy_array.additionalLeft = self.additionalLeft
        copy_array.additionalUp = self.additionalUp
        copy_array.leftConstWork = self.leftConstWork
        copy_array.rightConstWork = self.rightConstWork
        copy_array.upConstWork = self.upConstWork
        copy_array.downConstWork = self.downConstWork
        copy_array.work = np.copy(self.work)
        copy_array.horizontalMatrix = self.horizontalMatrix
        copy_array.verticalMatrix = self.verticalMatrix
        copy_array._left_part_n_prime = self._left_part_n_prime
        copy_array._right_part_n_prime = self._right_part_n_prime
        copy_array._up_part_n_prime = self._up_part_n_prime
        copy_array._down_part_n_prime = self._down_part_n_prime
        copy_array.tauLeaping = self.tauLeaping
        copy_array.use_modifyR = self.use_modifyR
        copy_array._Q_eigenbasis = np.copy(self._Q_eigenbasis)
        # Modified R variables
        if self.use_modifyR:
            copy_array._leftnExponent = np.copy(self._leftnExponent)
            copy_array._rightnExponent = np.copy(self._rightnExponent)
        copy_array.rates = np.copy(self.rates)
        copy_array.cumRates = np.copy(self.cumRates)
        # tau leaping variables
        if copy_array.tauLeaping:
            copy_array.totalAction = np.copy(self.totalAction)
            copy_array.left = np.copy(self.left)
            copy_array.right = np.copy(self.right)
            copy_array.down = np.copy(self.down)
            copy_array.up = np.copy(self.up)
        return copy_array

    def getRows(self):
        """
        :return: The number of rows in the array.
        """
        return self.rows

    def getColumns(self):
        """
        :return: The number of columns in the array.
        """
        return self.columns

    def getTemperatureArray(self, temperature, temperatureGradient):
        """
        Calculates the temperature on each junction with a constant temperature gradient.
        :param temperature: Temperature on the left side of the array.
        :param temperatureGradient: Gradient of temperature (left to right direction).
        :return: A vector holding the temperature of each junction in the same order as
         tunneling rates, [horizontal, horizontal, vertical, vertical].
        """
        line = temperature + temperatureGradient * np.arange(self.getColumns() + 1)
        Th = np.tile(line, (self.getRows(), 1))
        Tv = (Th[1:, :-1] + Th[1:, 1:])/2
        return np.hstack((Th.flatten(), Th.flatten(), Tv.flatten(), Tv.flatten()))

    def is_vertical_current(self):
        """
        :return: True if vertical current is possible, False otherwise
        """
        return self.upElectrode.any() or self.downElectrode.any()

    def setTemperature(self, temperature):
        """
        Sets the temperatures in the array
        :param temperature: Array temperature, can be number if it is uniform, or a vector with
         the same size and order as self.Rates if a temperature gradient exists.
        """
        self.temperature = temperature

    def getTemperature(self):
        """
        :return: Array temperature, can be number if it is uniform, or a vector with
         the same size and order as self.Rates if a temperature gradient exists.
        """
        return self.temperature

    def changeVext(self, newVL, newVR, newVU=0, newVD=0):
        """
        Updating the voltages on electrodes
        :param newVL: left electrode voltage.
        :param newVR: right electrode voltage.
        :param newVU: top electrode voltage.
        :param newVD: bottom electrode voltage.
        :return:
        """
        self.VL = newVL
        self.VR = newVR
        self.VU = newVU
        self.VD = newVD
        self.setConstWork()
        return True

    def getOccupation(self):
        """
        :return: the occupation (number of excess electrons) on each island.
         Numpy array with the shape of the islands array.
        """
        return np.copy(self.n).reshape((self.rows, self.columns))

    def getGateCharge(self):
        """
        :return: the charge on gate capacitors of each island.
         Numpy array with the shape of the islands array.
        """
        return np.copy(self.Q).reshape((self.rows, self.columns))

    def setOccupation(self, n):
        """
        Setting array occupation (number of excess electrons on each island)
        :param n: Numpy array with the same size as the islands array.
        """
        self.n = flattenToColumn(n)

    def setGateCharge(self, Q):
        """
        Setting charges on gate capacitors.
        :param Q: Numpy array with the same size as the islands array.
        """
        self.Q = flattenToColumn(Q)
        self._Q_eigenbasis = self._InvTauEigenVectorsInv.dot(self.Q)

    def getTimeStep(self):
        """
        :return: The approximated time until steady state (after this time,
        the state would be evaluated to see if the system already reached a steady state).
        """
        return self.timeStep

    def getCurrentMap(self):
        """
        :return: An array with the local current on each junction.
        """
        Ih = self.rates[:self.rightWorkLen] - \
             self.rates[self.rightWorkLen:2 * self.rightWorkLen]
        Iv = self.rates[2 * self.rightWorkLen:(2 * self.rightWorkLen + self.downWorkLen)] - \
             self.rates[(2 * self.rightWorkLen + self.downWorkLen):]
        map = np.zeros((self.rows*2 + 1, self.columns+1))
        map[1::2, :] = Ih.reshape((self.rows, self.columns+1))
        if Iv.size:
            map[::2, :-1] = Iv.reshape((self.rows + 1, self.columns))
        return map

    def createCapacitanceMatrix(self):
        """
        Creates the inverse capacitance matrix (called only once for each simulation,
        matrix is shallow-copied between instances)
        """
        diagonal = self.Ch[:, :-1] + self.Ch[:, 1:] + self.Cv[:-1, :] + self.Cv[1:, :]
        second_diagonal = np.copy(self.Ch[:, 1:])
        second_diagonal[:, -1] = 0
        second_diagonal = second_diagonal.flatten()
        second_diagonal = second_diagonal[:-1]
        n_diagonal = np.copy(self.Cv[1:-1, :])
        C_mat = np.diagflat(diagonal) - np.diagflat(second_diagonal, k=1) - np.diagflat(second_diagonal, k=-1) - \
                np.diagflat(n_diagonal, k=self.columns) - np.diagflat(n_diagonal, k=-self.columns)
        self.invC = np.linalg.inv(C_mat)
        self.invCeq = np.linalg.inv(C_mat + np.diagflat(self.CG))
        return True

    def setConstNprimePart(self):
        """
        Creating the boundary terms (external voltage dependent terms) for energy calculation
        """
        self._left_part_n_prime = np.copy(self.Ch[:, :-1])
        self._left_part_n_prime[:, 1:] = 0
        self._left_part_n_prime[:, 0] = self._left_part_n_prime[:, 0]*self.leftElectrode
        self._right_part_n_prime = np.copy(self.Ch[:, 1:])
        self._right_part_n_prime[:, :-1] = 0
        self._right_part_n_prime[:, -1] = self._right_part_n_prime[:, -1]*self.rightElectrode
        self._up_part_n_prime = np.copy(self.Cv[:-1, :])
        self._up_part_n_prime[1:, :] = 0
        self._up_part_n_prime[0, :] = self._up_part_n_prime[0, :]*self.upElectrode
        self._down_part_n_prime = np.copy(self.Cv[1:, :])
        self._down_part_n_prime[:-1, :] = 0
        self._down_part_n_prime[-1, :] = self._down_part_n_prime[-1, :] * self.downElectrode
        self._right_part_n_prime = flattenToColumn(self._right_part_n_prime)
        self._left_part_n_prime = flattenToColumn(self._left_part_n_prime)
        self._up_part_n_prime = flattenToColumn(self._up_part_n_prime)
        self._down_part_n_prime = flattenToColumn(self._down_part_n_prime)
        return True

    def getNprime(self):
        """
        Adding the boundary terms to array occupation.
        :return:  Array occupation including boundary terms (marked as n' in the thesis).
        """
        return self.n + self._left_part_n_prime*self.VL + \
               self._right_part_n_prime*self.VR + \
               self._up_part_n_prime*self.VU + \
               self._down_part_n_prime*self.VD

    def getNprimeForGivenN(self, n):
        """
        Adding the boundary terms to the given array occupation.
        :param n: Numpy array with the same shape as the islands array.
        :return:  Array occupation including boundary terms (marked as n' in the thesis).
        """
        return flattenToColumn(n) + self._left_part_n_prime*self.VL + \
               self._right_part_n_prime*self.VR + \
               self._up_part_n_prime * self.VU + \
               self._down_part_n_prime * self.VD

    def getbVector(self):
        """
        For gate charge relaxation, given by dQ/dt=inv_tau*Q +b
        """
        res = -self.invC.dot(self.getNprime()) + self.VG
        return res / self.RG

    def getInvTauMatrix(self):
        """
        For development of Q by dQ/dt=inv_tau*Q +b
        """
        res = self.invC + np.diagflat(1/self.CG)
        return -res / np.repeat(flattenToColumn(self.RG), res.shape[1], axis=1)

    def setDiagonalizedInvTau(self):
        """
        Diagonalizing inverse tau matrix, for updating gate charges between tunnelings.
        :return:
        """
        self._InvTauEigenValues, self._InvTauEigenVectors = np.linalg.eig(self.getInvTauMatrix())
        self._InvTauEigenValues = flattenToColumn(self._InvTauEigenValues)
        self._InvTauEigenVectorsInv = np.linalg.inv(self._InvTauEigenVectors)
        self.timeStep = -2/np.max(self._InvTauEigenValues)  # Time to reach steady state
        self.default_dt = -0.1/np.min(self._InvTauEigenValues)  # Time in which QG doesn't change much
        tauMat = np.linalg.inv(self.invC + np.diagflat(1/self.CG))
        self._constQnPart = tauMat.dot(self.VG)
        CGMat = np.repeat(1/flattenToRow(self.CG), tauMat.shape[0], axis=0)
        self._matrixQnPart = tauMat * CGMat - np.eye(tauMat.shape[0])
        self._Q_eigenbasis = self._InvTauEigenVectorsInv.dot(self.Q)
        return True

    def developQ(self, dt):
        """
        Updating gate charges according to charge relaxation.
         Calculating the gate charges for the current time + dt.
        :param dt: Determines for what time gate charges would be calculated.
        """
        b = self._InvTauEigenVectorsInv.dot(self.getbVector())
        exponent = np.exp(self._InvTauEigenValues*dt)
        self._Q_eigenbasis = self._Q_eigenbasis*exponent + (b/self._InvTauEigenValues) * (exponent - 1)
        self.Q = self._InvTauEigenVectors.dot(self._Q_eigenbasis)
        return True

    def get_steady_Q_for_n(self):
        """
        :return: Gate charges steady-state value, for the current occupation.
        """
        return self._constQnPart + self._matrixQnPart.dot(self.getNprime())

    def get_steady_Q_for_given_n(self, n):
        """
        :param n: A given occupation (column numpy vector or array with the same shape as islands array)
        :return: Gate charges steady-state value, for the given occupation.
        """
        n = flattenToColumn(n)
        return self._constQnPart + self._matrixQnPart.dot(self.getNprimeForGivenN(n))

    def get_dist_from_steady(self, n, Q):
        """
        Calculating the distance from a steady state, which is the maximum difference
         between a given gate charge and its desired equilibrium value (for the given n).
        :param n: Given occupation (column numpy vector or array with the same shape as islands array).
        :param Q: Given gate charge (column numpy vector or array with the same shape as islands array).
        :return: Distance from equilibrium.
        """
        return np.max(np.abs(flattenToColumn(Q) - self.get_steady_Q_for_given_n(n)))

    def setConstWork(self):
        """
        Calculating the constant part of the energy differences for tunneling. Shallow copied between instances.
        """
        # "Energy barrier terms" that are independent of external voltage.
        invCDiagMat = np.diag(np.copy(self.invC)).reshape((self.rows, self.columns))
        lowerCDiag = np.pad(np.diag(np.copy(self.invC), k=-1), ((0, 1),), mode='constant').reshape((self.rows,
                                                                                                   self.columns))
        lowerCDiag[:, -1] = 0
        lowerCDiag = np.pad(lowerCDiag, ((0, 0), (1, 0)), mode='constant')
        upperCDiag = np.pad(np.diag(np.copy(self.invC), k=1), ((0, 1),), mode='constant').reshape((self.rows,
                                                                                                   self.columns))
        upperCDiag[:, -1] = 0
        upperCDiag = np.pad(upperCDiag, ((0, 0), (1, 0)), mode='constant')
        self.commonHorz = np.pad(invCDiagMat, ((0, 0), (1, 0)), mode='constant') + np.pad(invCDiagMat, ((0, 0), (0, 1)),
                                                                                          mode='constant') - \
                         lowerCDiag - upperCDiag

        lowerNCDiag = np.diag(np.copy(self.invC), k=-self.columns).reshape((self.rows-1, self.columns))
        upperNCDiag = np.diag(np.copy(self.invC), k=self.columns).reshape((self.rows-1, self.columns))
        self.commonVert = np.pad(invCDiagMat, ((1, 0), (0, 0)), mode='constant') +\
                          np.pad(invCDiagMat, ((0, 1), (0, 0)), mode='constant') -\
                          np.pad(lowerNCDiag, ((1, 1), (0, 0)), mode='constant') -\
                          np.pad(upperNCDiag, ((1, 1), (0, 0)), mode='constant')

        # External voltage dependent terms
        self.additionalLeft = np.zeros((self.rows, self.columns+1))
        self.additionalLeft[self.leftElectrode, 0] = self.VL
        self.additionalLeft[self.rightElectrode, -1] = -self.VR
        self.leftConstWork = (0.5*self.commonHorz + self.additionalLeft).flatten()

        additionalRight = -self.additionalLeft
        self.rightConstWork = (0.5*self.commonHorz + additionalRight).flatten()
        
        self.additionalUp = np.zeros((self.rows + 1, self.columns))
        self.additionalUp[0, self.upElectrode] = self.VU
        self.additionalUp[-1, self.downElectrode] = -self.VD
        self.upConstWork = (0.5*self.commonVert + self.additionalUp).flatten()

        additionalDown = -self.additionalUp
        self.downConstWork = (0.5*self.commonVert + additionalDown).flatten()

        # Initializing memory for rates
        self.work = np.zeros((4*self.rows*self.columns + 2*(self.rows + self.columns),))
        self.rates = np.zeros(self.work.shape)
        self.cumRates = np.zeros(self.work.shape)

        # Vector for setting up electrodes in the right places
        horz_no_connections = np.zeros((self.rows, self.columns+1), dtype=np.bool)
        horz_no_connections[:, 0] = np.logical_not(self.leftElectrode)
        horz_no_connections[:, -1] = np.logical_not(self.rightElectrode)
        horz_no_connections = horz_no_connections.flatten()
        vert_no_connections = np.zeros((self.rows+1, self.columns), dtype=bool)
        vert_no_connections[0, :] = np.logical_not(self.upElectrode)
        vert_no_connections[-1, :] = np.logical_not(self.downElectrode)
        vert_no_connections = vert_no_connections.flatten()
        self.no_connections = np.hstack((horz_no_connections, horz_no_connections,
                                         vert_no_connections, vert_no_connections))

        return True

    def setConstMatrix(self):
        """
        Setting up the matrix that multiplies charge dynamical terms (n and QG) in energy difference calculation.
        Shallow copied between instances.
        :return:
        """
        firstHorzMat = np.zeros(((self.columns + 1) * self.rows, self.columns * self.rows))
        secondHorzMat = np.zeros(firstHorzMat.shape)
        firstLocations = np.ones(((self.columns + 1) * self.rows,))
        secondLocations = np.ones(firstLocations.shape)
        firstLocations[self.columns::self.columns + 1] = 0
        secondLocations[0::self.columns + 1] = 0
        firstHorzMat[firstLocations.astype(np.bool), :] = np.copy(self.invC)
        secondHorzMat[secondLocations.astype(np.bool), :] = np.copy(self.invC)
        self.horizontalMatrix = firstHorzMat - secondHorzMat
        firstVertMat = np.pad(np.copy(self.invC), ((0, self.columns), (0, 0)), mode='constant')
        secondVertMat = np.pad(np.copy(self.invC), ((self.columns, 0), (0, 0)), mode='constant')
        self.verticalMatrix = firstVertMat - secondVertMat
        return True

    def getWork(self):
        """
        Calculating the work (energy difference) needed for each possible tunneling.
        :return: The work for each tunneling in the following order
         [right to left, left to right, top down, bottom up] each of terms is ordered in the same order as the
          islands array, i.e. (0,0), (0,1), .... (0,N), (1,0), (1,1), .....(M,0), (M,1),... (M,N)
        """
        q = self.getNprime() + self.Q
        variableRightWork = (self.horizontalMatrix.dot(q)).flatten()
        variableDownWork = (self.verticalMatrix.dot(q)).flatten()
        variableRightWorkLen = variableRightWork.size
        variableDownWorkLen = variableDownWork.size
        self.work[:variableRightWorkLen] = variableRightWork + self.rightConstWork
        self.work[variableRightWorkLen:2*variableRightWorkLen] = -variableRightWork + self.leftConstWork
        self.work[2 * variableRightWorkLen:2 * variableRightWorkLen + variableDownWorkLen] = variableDownWork +\
                                                                                             self.downConstWork
        self.work[2 * variableRightWorkLen + variableDownWorkLen:] = -variableDownWork + self.upConstWork
        return self.work

    def getRates(self):
        """
        Returns the tunnelling rate between neighboring dots.
        :return: The rate for each tunneling in the following order
         [right to left, left to right, top down, bottom up] each of terms is ordered in the same order as the
          islands array, i.e. (0,0), (0,1), .... (0,N), (1,0), (1,1), .....(M,0), (M-1,1),... (M-1,N)
        """
        self.getWork()
        work = np.copy(self.work)
        # are we in zero temperature?
        if hasattr(self.temperature, "__len__"):
            zero_temp = (self.temperature == 0).all()
        else:
            zero_temp = (self.temperature == 0)
        if zero_temp:  # If zero temperature, only energy reducing tunnelings are allowed
            work[work > -EPS] = 0
        else:
            exp_arg = work/self.temperature
            work[exp_arg > 20] = 0  # Very low temperature and positive work
            work[np.abs(exp_arg) < EPS] = -self.temperature[np.abs(exp_arg) < EPS] if \
                hasattr(self.temperature, "__len__") else -self.temperature  # high temperatures
            rest = np.logical_and(EPS <= np.abs(exp_arg), np.abs(exp_arg) <= 20)
            work[rest] = work[rest]/(1-np.exp(exp_arg[rest]))  # medium temperatures
        if self.use_modifyR:
            self.modifyR()
        self.rates = -work / self.R
        self.rates[self.no_connections] = 0  # No tunneling to or from electrodes where there is no connection
        return self.rates

    def getCurrentFromRates(self):
        """
        Calculates the current using tunneling rates (for e=1)
        :return: horizontal current (positive direction - left->right),
        vertical current (positive direction - top-> bottom)
        """
        if self.is_vertical_current():
            rightCurrent = np.sum(self.rates[0:self.rightWorkLen:self.columns+1] +
                                  self.rates[self.columns:self.rightWorkLen:self.columns+1]) -\
                          np.sum(self.rates[self.rightWorkLen:2*self.rightWorkLen:self.columns+1] +
                                 self.rates[self.rightWorkLen + self.columns:
                                            2*self.rightWorkLen:self.columns+1])
            downCurrent = np.sum(self.rates[2*self.rightWorkLen:2*self.rightWorkLen + self.columns] +
                                 self.rates[2*self.rightWorkLen + self.downWorkLen - self.columns:
                                            2*self.rightWorkLen + self.downWorkLen]) -\
                          np.sum(self.rates[2*self.rightWorkLen + self.downWorkLen:
                                            2*self.rightWorkLen + self.downWorkLen + self.columns] +
                                 self.rates[2*self.rightWorkLen + 2*self.downWorkLen - self.columns:])
            return rightCurrent/2, downCurrent/2
        else:
            current = np.sum(self.rates[:self.rightWorkLen]) - \
                      np.sum(self.rates[self.rightWorkLen:2 * self.rightWorkLen])
            return current / (self.columns + 1), 0

    def modifyR(self):
        """
        Modifies tunneling resistance according to the given finite, constant, density of states.
        :return:
        """
        nExponent = np.exp(-INV_DOS*self.n).reshape((self.rows, self.columns))
        self._rightnExponent[:, 1:] = nExponent
        self._leftnExponent[:, :-1] = nExponent
        horzSize = self.Rh.size
        vertSize = self.Rv.size
        self.R[:horzSize] = (self.Rh*self._rightnExponent).flatten()
        self.R[horzSize:2*horzSize] = (self.Rh * self._leftnExponent).flatten()
        if vertSize > 0:
            self.R[2*horzSize:2*horzSize + vertSize] = (self.Rv*nExponent[:-1, :]).flatten()
            self.R[2 * horzSize + vertSize:] = (self.Rv * nExponent[1:, :]).flatten()

    def getVoltages(self):
        """
        :return: Voltage on each island (numpy array with same shape as islands array)
        """
        return self.invC.dot(self.getNprime() + self.Q).reshape(self.rows, self.columns)

    def getVoltagesFromGate(self):
        """
        :return: Voltage on each island, assuming no current through gate resistor
         (numpy array with same shape as islands array)
        """
        return (self.VG - self.Q/self.CG).reshape(self.rows, self.columns)

    def getTimeInterval(self, randomNumber):
        """
        Calculates time until next tunneling using Gillespie's algorithm.
        :param randomNumber: Uniformly distributing random number between 0 and 1.
        :return: (dt, intervals). dt -  time until next tunneling.
         intervals - how many different QG updating intervals were needed.
        """
        rates = self.getRates()
        np.cumsum(rates, out=self.cumRates)
        sum_rates = self.cumRates[-1]
        if sum_rates < 1e-10:
            self.no_tunneling_next_time = True
            return self.default_dt, 0
        dt = np.log(1 / randomNumber) / sum_rates
        return self.update_Q_for_long_steps(dt)

    def update_Q_for_long_steps(self, dt):
        """
        Updating Q in small steps for long time step.
        :param dt: total step time.
        :return: (dt, intervals). dt -  time until next tunneling.
         intervals - how many different QG updating intervals were needed.
        """
        sum_rates = self.cumRates[-1]
        intervals = 0
        out_of_interval = dt > self.default_dt
        while out_of_interval:
            intervals += 1
            self.developQ(self.default_dt)
            rates = self.getRates()
            np.cumsum(rates, out=self.cumRates)
            new_sum_rates = self.cumRates[-1]
            if new_sum_rates == 0:
                self.no_tunneling_next_time = True
                return self.default_dt, intervals
            dt = (dt - self.default_dt) * (new_sum_rates / sum_rates)
            sum_rates = new_sum_rates
            out_of_interval = dt > self.default_dt
            if np.max(self.get_dist_from_steady(self.n, self.Q)) < ALLOWED_ERR:
                break
            return dt, intervals

    def getLeapingTimeInterval(self):
        """
        Time until next tunneling batch, according to tau-leaping approximation
        :return: (tau, intervals). tau -  time until next tunneling batch.
         intervals - how many different QG updating intervals were needed.
        """
        rates = self.getRates()
        if (rates <= EPS).all():
            self.no_tunneling_next_time = True
            return self.default_dt
        horzSize = self.rows * (self.columns + 1)
        vertSize = (self.rows + 1) * self.columns
        self.right[:, :] = rates[:horzSize].reshape(self.right.shape)
        self.left[:, :] = rates[horzSize:2*horzSize].reshape(self.left.shape)
        self.down[:, :] = rates[2*horzSize:2*horzSize + vertSize].reshape(self.up.shape)
        self.up[:, :] = rates[2*horzSize + vertSize:].reshape(self.down.shape)
        tunnelingTo = self.right[:, :-1] + self.left[:, 1:]
        tunnelingTo[1:, :] += self.down
        tunnelingTo[:-1, :] += self.up
        tunnelingFrom = self.right[:, 1:] + self.left[:, :-1]
        tunnelingFrom[1:, :] += self.up
        tunnelingFrom[:-1, :] += self.down
        changeAvg = np.abs(tunnelingTo - tunnelingFrom)
        changeVar = tunnelingTo + tunnelingFrom
        smallestChange = TAU_EPS*np.abs(self.getOccupation())
        smallestChange[smallestChange < 1] = 1
        tau = 0
        if (changeAvg > 0).any():
            tau = np.min(smallestChange[changeAvg > 0] / changeAvg[changeAvg > 0])
        if (changeVar > 0).any():
            tau2 = np.min(smallestChange[changeVar > 0]**2/changeVar[changeVar > 0])
            if tau > 0:
                tau = min(tau, tau2)
            else:
                tau = tau2
        if tau <= 1e-10:
            tau = self.default_dt
            self.no_tunneling_next_time = True
            return tau, 0
        else:
            return self.update_Q_for_long_steps(tau)

    def nextStep(self, dt, randomNumber):
        """
        Executing the next tunneling event according to Gillespie's algorithm
        :param dt: time for updating QG
        :param randomNumber: random uniformly distributing number between 0 and 1
        """
        self.developQ(dt)
        if self.no_tunneling_next_time:
            self.no_tunneling_next_time = False
            return True
        if (self.rates == 0).all():
            return True
        actionInd = np.searchsorted(self.cumRates, randomNumber*self.cumRates[-1])
        self.executeAction(actionInd)
        return True

    def nextLeapingSteps(self, dt):
        """
        Executing the next tunneling batch according to Gillespie's tau-leaping approximation/
        :param dt: - time for updating QG.
        """
        self.developQ(dt)
        if self.no_tunneling_next_time:
            self.no_tunneling_next_time = False
            return True
        actionVec = np.random.poisson(lam=self.rates*dt, size=self.rates.size)
        self.executeMultipleActions(actionVec)

    def tunnel(self, fromDot, toDot, charge=1):
        """
        Executing a tunnel.
        :param fromDot: origin island index (i,j)
        :param toDot: destination island index (i',j')
        :param charge: How much charge is tunneling (in units of e)
        """
        if (0 <= fromDot[1] < self.columns) and (0 <= fromDot[0] < self.rows):
            self.n[fromDot[0]*self.columns + fromDot[1], 0] -= charge
        if (0 <= toDot[1] < self.columns) and (0 <= toDot[0] < self.rows):
            self.n[toDot[0]*self.columns + toDot[1], 0] += charge
        return True

    def printState(self):
        """
        Printing the state of the array. For debug.
        """
        for i in range(self.rows):
            for j in range(self.columns):
                print("At dot (" + str(i) + ", " + str(j) + ") the state "
                      "is: n= " + str(self.n[i, j]) + " and QG = " + str(
                    self.Q[i, j]))

    def __str__(self):
        """
        :return: Array parameters string representation.
        """
        rows = "Rows: " + str(self.rows)
        columns = "Columns: " + str(self.columns)
        VG = "VG: " + str(self.VG)
        CG = "CG: " + str(self.CG)
        RG = "RG: " + str(self.RG)
        Ch = "Ch: " + str(self.Ch)
        Cv = "Cv: " + str(self.Cv)
        Rh = "Rh: " + str(self.Rh)
        Rv = "Rv: " + str(self.Rv)
        res = "----Array Parameters-----"
        for param in [rows, columns, VG, CG, RG, Ch, Cv, Rh, Rv]:
            res += "\n" + param
        return res

    def executeMultipleActions(self, actionVec, charge=1):
        """
        Executing multiple tunnelings (for tau-leaping)
        :param actionVec: Number of tunnelings from each action,
        [right tunnelings, left tunnelings, down tunnelings, up tunnelings].
        :param charge: How much charge transfers at each tunneling (in e units).
        :return:
        """
        horzSize = self.rows * (self.columns + 1)
        vertSize = (self.rows + 1) * self.columns
        self.right[:, :] = actionVec[:horzSize].reshape(self.right.shape)*charge
        self.left[:, :] = actionVec[horzSize:2 * horzSize].reshape(self.left.shape)*charge
        self.down[:, :] = actionVec[2 * horzSize:2 * horzSize + vertSize].reshape(self.down.shape)*charge
        self.up[:, :] = actionVec[2 * horzSize + vertSize:].reshape(self.up.shape)*charge
        self.totalAction[:, :] = 0
        self.totalAction += self.left[:, 1:] - self.left[:, :-1] + self.right[:, :-1] - self.right[:, 1:]
        self.totalAction[1:, :] += self.down - self.up
        self.totalAction[:-1, :] += self.up - self.down
        self.n += flattenToColumn(self.totalAction)

    def executeAction(self, ind, charge=1):
        """
        Execute a single tunneling
        :param ind: action ind, according to self.rates order
        :param charge: how much charge tunnels (in units of e).
        :return: (origin island index (i,j), destination island index (i',j')).
        """
        horzSize = self.rows*(self.columns+1)
        vertSize = (self.rows + 1)*self.columns
        if ind < horzSize:  # tunnel right:
            fromDot = (ind//(self.columns+1), (ind % (self.columns+1))-1)
            toDot = (fromDot[0], fromDot[1] + 1)
        elif ind < horzSize*2:  # tunnel left
            ind -= horzSize
            fromDot = (ind//(self.columns+1), ind % (self.columns+1))
            toDot = (fromDot[0], fromDot[1] - 1)
        elif ind < horzSize*2 + vertSize:  # tunnel down
            ind -= horzSize*2
            fromDot = (ind//self.columns-1, ind % self.columns)
            toDot = (fromDot[0] + 1, fromDot[1])
        else:  # tunnel up
            ind -= (horzSize*2 + vertSize)
            fromDot = ((ind//self.columns), ind % self.columns)
            toDot = (fromDot[0] - 1, fromDot[1])
        self.tunnel(fromDot, toDot, charge=charge)
        return fromDot, toDot


class JJArray(DotArray):
    """
    Superconducting array version
    """
    def __init__(self, rows, columns, VL, VR, VU, VD,  VG, Q0, n0, CG, RG, Ch, Cv, Rh, Rv,
                 temperature, temperature_gradient, scGap, leftElectrode, rightElectrode, upElectrode, downElectrode,
                 fastRelaxation=False, tauLeaping=False, modifyR=False):
        """
        Creates a new Josephson junctions array.
        :param rows: number of rows (int)
        :param columns: number of columns (int)
        :param VL: left electrode voltage (double)
        :param VR: right electrode voltage (double)
        :param VU: top electrode voltage (double)
        :param VD: bottom electrode voltage (double)
        :param Q0: np array of initial charges (rowsXcolumns double array)
        :param n0: np array of initial electrons (rowsXcolumns double array)
        :param VG: voltage of the gates (rowsXcolumns double array)
        :param Ch: horizontal capacitance ((rowsXcolumns+1 double array)
        :param Cv: vertical capacitance (rows-1Xcolumns double array)
        :param Rh: vertical tunnelling ressistance (rowsXcolumns+1 double array)
        :param Rv: horizontal tunnelling ressistance (rowsXcolumns+1 double array)
        :param RG: gate resistances (rowsXcolumns double array)
        :param scGap: superconducting gap.
        :param temperature: temperature on the left side of the array (double)
        :param temperature_gradient: constant temperature gradient, from left to right (double)
        :param left/right/up/downElectrode: location of electrodes
         ((rows,) size array with 1 where electrode is connected and 0 otherwise)
        :param fastRelaxation: True for working in the fast relaxation limit.
        :param tauLeaping: True to use tau leaping approximation.
        :param modifyR: True to modify tunneling resistance according to the occupation (finite DOF for electrons).
        """
        DotArray.__init__(self, rows, columns, VL, VR, VU, VD, VG, Q0, n0, CG, RG, Ch, Cv, Rh, Rv,
                          temperature, temperature_gradient, leftElectrode, rightElectrode, upElectrode, downElectrode,
                          fastRelaxation=fastRelaxation, tauLeaping=tauLeaping, modifyR=modifyR)

        self.gap = scGap
        self.Ej = self.getEj()
        self.Ec = 1/(2*np.mean(CG))
        self.qp_rate_calculator = TunnelingRateCalculator(-1, 1, 0.01, qp_tunneling, self.Ec, temperature, scGap,
                                                          "quasi_particles_rate")
        self.cp_rate_calculator = TunnelingRateCalculator(-1, 1, 0.01, cp_tunneling, self.Ec, temperature, self.Ej,
                                                          "cooper_pairs_rate")
        print("Rates were calculated", flush=True)
        if tauLeaping:
            self.right_cp = np.zeros((self.rows, self.columns + 1))
            self.left_cp = np.zeros((self.rows, self.columns + 1))
            self.down_cp = np.zeros((self.rows - 1, self.columns))
            self.up_cp = np.zeros((self.rows - 1, self.columns))

    def __copy__(self):
        """
        Creating a copy of the array. Shallow copy for static constant variables.
        Deep copy for dynamical variables.
        :return: a copy of the current instance.
        """
        copy_array = super().__copy__()
        copy_array.gap = self.gap
        copy_array.Ej = self.Ej
        copy_array.Ec = self.Ec
        copy_array.qp_rate_calculator = self.qp_rate_calculator
        copy_array.cp_rate_calculator = self.cp_rate_calculator
        if self.tauLeaping:
            copy_array.right_cp = np.copy(self.right_cp)
            copy_array.left_cp = np.copy(self.left_cp)
            copy_array.up_cp = np.copy(self.up_cp)
            copy_array.down_cp = np.copy(self.down_cp)
        return copy_array

    def getEj(self):
        """
        Calculates Josephson energy according to the given SC gap.
        :return: Ej
        """
        return (self.gap/8)*np.tanh(self.gap/(2*T))

    def setConstWork(self):
        """
        Setting the constant part of the work (energy differences) for tunnelings
        """
        super().setConstWork()
        leftConstWorkCp = 2*(self.commonHorz + self.additionalLeft).flatten()

        additionalRight = -self.additionalLeft
        rightConstWorkCp = 2*(self.commonHorz + additionalRight).flatten()

        upConstWorkCp = 2*(self.commonVert + self.additionalUp).flatten()
        additionalDown = -self.additionalUp
        downConstWorkCp = 2*(self.commonVert + additionalDown).flatten()

        self.constWorkCp = np.hstack((rightConstWorkCp, leftConstWorkCp, downConstWorkCp, upConstWorkCp))
        self.constWorkQp = np.hstack((self.rightConstWork, self.leftConstWork, self.downConstWork, self.upConstWork))

        self.rates = np.zeros((2*self.work.size,))
        self.cumRates = np.zeros((2*self.work.size,))

    def getWork(self):
        """
        Calculating the required work for tunnelings.
        :return: work for a quasi particle tunneling, work for a Cooper pair tunneling.
        """
        q = self.getNprime() + self.Q
        variableRightWork = (self.horizontalMatrix.dot(q)).flatten()
        variableDownWork = (self.verticalMatrix.dot(q)).flatten()
        variableRightWorkLen = variableRightWork.size
        variableDownWorkLen = variableDownWork.size
        variableWork = self.work
        variableWork[:variableRightWorkLen] = variableRightWork
        variableWork[variableRightWorkLen:2 * variableRightWorkLen] = -variableRightWork
        variableWork[2 * variableRightWorkLen:2 * variableRightWorkLen + variableDownWorkLen] = variableDownWork
        variableWork[2 * variableRightWorkLen + variableDownWorkLen:] = -variableDownWork
        qp_work = variableWork + self.constWorkQp
        cp_work = 2 * variableWork + self.constWorkCp
        return qp_work, cp_work

    def getCurrentFromRates(self):
        mid = self.rates.size // 2
        if self.is_vertical_current():
            rightCurrent = np.sum(self.rates[0:self.rightWorkLen:self.columns + 1] +
                                  self.rates[self.columns:self.rightWorkLen:self.columns + 1]) - \
                           np.sum(self.rates[self.rightWorkLen:2 * self.rightWorkLen:self.columns + 1] +
                                  self.rates[self.rightWorkLen + self.columns:
                                             2 * self.rightWorkLen:self.columns + 1]) + \
                           2*np.sum(self.rates[mid:mid+self.rightWorkLen:self.columns + 1] +
                                  self.rates[mid + self.columns:mid+self.rightWorkLen:self.columns + 1]) - \
                           2*np.sum(self.rates[mid + self.rightWorkLen:mid+2 * self.rightWorkLen:self.columns + 1] +
                                  self.rates[mid + self.rightWorkLen + self.columns:
                                             mid + 2 * self.rightWorkLen:self.columns + 1])
            downCurrent = np.sum(self.rates[2 * self.rightWorkLen:2 * self.rightWorkLen + self.columns] +
                                 self.rates[2 * self.rightWorkLen + self.downWorkLen - self.columns:
                                            2 * self.rightWorkLen + self.downWorkLen]) - \
                          np.sum(self.rates[2 * self.rightWorkLen + self.downWorkLen:
                                            2 * self.rightWorkLen + self.downWorkLen + self.columns] +
                                 self.rates[2 * self.rightWorkLen + 2 * self.downWorkLen - self.columns:
                                            2 * self.rightWorkLen + 2 * self.downWorkLen]) + \
                          2*np.sum(self.rates[mid + 2 * self.rightWorkLen:mid+2 * self.rightWorkLen + self.columns] +
                                 self.rates[mid + 2 * self.rightWorkLen + self.downWorkLen - self.columns:
                                            mid + 2 * self.rightWorkLen + self.downWorkLen]) - \
                          2*np.sum(self.rates[mid + 2 * self.rightWorkLen + self.downWorkLen:
                                            mid + 2 * self.rightWorkLen + self.downWorkLen + self.columns] +
                                 self.rates[mid + 2 * self.rightWorkLen + 2 * self.downWorkLen - self.columns:
                                            mid + 2 * self.rightWorkLen + 2 * self.downWorkLen])
            return rightCurrent / 2, downCurrent / 2
        else:
            current = np.sum(self.rates[:self.rightWorkLen]) -\
                  np.sum(self.rates[self.rightWorkLen:2*self.rightWorkLen]) + \
                  2 * np.sum(self.rates[mid:mid+self.rightWorkLen]) - \
                  2 * np.sum(self.rates[mid+self.rightWorkLen:mid+2 * self.rightWorkLen])
            return current/(self.columns+1), 0

    def getRates(self):
        """
        :return: Tunnelling rates for all possible tunnelings,
         [rates for quasi particles, rates for Cooper pairs].
        """
        qp_work, cp_work = self.getWork()
        if self.temperature == 0:
            raise NotImplementedError
        else:
            mid = self.rates.size // 2
            self.rates[:mid] = self.qp_rate_calculator.get_tunneling_rates(-qp_work) / self.R
            self.rates[mid:] = self.cp_rate_calculator.get_tunneling_rates(-cp_work) / (self.R**2)
            self.rates[np.abs(self.rates) < EPS] = 0
        return self.rates

    def getLeapingTimeInterval(self):
        """
        Time until next tunneling batch, according to tau-leaping approximation
        :return: (tau, intervals). tau -  time until next tunneling batch.
         intervals - how many different QG updating intervals were needed.
        """
        mid = self.rates.size // 2
        rates = self.getRates()
        if (rates <= EPS).all():
            self.no_tunneling_next_time = True
            return self.default_dt
        horzSize = self.rows * (self.columns + 1)
        vertSize = (self.rows - 1) * self.columns
        self.right[:, :] = rates[:horzSize].reshape(self.right.shape)
        self.left[:, :] = rates[horzSize:2 * horzSize].reshape(self.left.shape)
        self.down[:, :] = rates[2 * horzSize:2 * horzSize + vertSize].reshape(self.up.shape)
        self.up[:, :] = rates[2 * horzSize + vertSize:mid].reshape(self.down.shape)
        self.right_cp[:, :] = 2 * rates[mid:mid + horzSize].reshape(self.right.shape)
        self.left_cp[:, :] = 2 * rates[mid + horzSize:mid + 2 * horzSize].reshape(self.left.shape)
        self.down_cp[:, :] = 2 * rates[mid + 2 * horzSize:mid + 2 * horzSize + vertSize].reshape(self.up.shape)
        self.up_cp[:, :] = 2 * rates[mid + 2 * horzSize + vertSize:].reshape(self.down.shape)
        tunnelingTo = self.right[:, :-1] + self.left[:, 1:] + self.right_cp[:, :-1] + self.left_cp[:, 1:]
        tunnelingTo[1:, :] += self.down + self.down_cp
        tunnelingTo[:-1, :] += self.up + self.up_cp
        tunnelingFrom = self.right[:, 1:] + self.left[:, :-1] + self.right_cp[:, 1:] + self.left_cp[:, :-1]
        tunnelingFrom[1:, :] += self.up + self.up_cp
        tunnelingFrom[:-1, :] += self.down + self.down_cp
        changeAvg = np.abs(tunnelingTo - tunnelingFrom)
        changeVar = tunnelingTo + tunnelingFrom
        smallestChange = TAU_EPS * np.abs(self.getOccupation())
        smallestChange[smallestChange < 1] = 1
        tau = 0
        if (changeAvg > 0).any():
            tau = np.min(smallestChange[changeAvg > 0] / changeAvg[changeAvg > 0])
        if (changeVar > 0).any():
            tau2 = np.min(smallestChange[changeVar > 0] ** 2 / changeVar[changeVar > 0])
            if tau > 0:
                tau = min(tau, tau2)
            else:
                tau = tau2
        if tau <= 0:
            tau = self.default_dt
            self.no_tunneling_next_time = True
            return tau, 0
        else:
            return self.update_Q_for_long_steps(tau)

    def executeAction(self, ind, charge=1):
        """
        Executes a tunneling of quasi-particle or Cooper pair
        :param ind: action index according to th same order as in self.rates.
        :param charge: Deprecated.
        :return: (origin island index (i,j), destination island index (i',j'))
        """
        horzSize = self.rows*(self.columns+1)
        vertSize = (self.rows + 1)*self.columns
        if ind < 2*(horzSize + vertSize):
            fromDot, toDot = super().executeAction(ind, charge=1)
        else:
            ind = ind - 2*(horzSize + vertSize)
            fromDot, toDot = super().executeAction(ind, charge=2)
        return fromDot, toDot

    def executeMultipleActions(self, actionVec, charge=1):
        """
        Executing multiple tunnelings (for tau-leaping)
        :param actionVec: Number of tunnelings from each action,
        [right tunnelings, left tunnelings, down tunnelings, up tunnelings].
        :param charge: Deprecated.
        :return:
        """
        mid = self.rates.size // 2
        super().executeMultipleActions(actionVec[:mid], charge=1)
        super().executeMultipleActions(actionVec[mid:], charge=2)


class Simulator:
    """
    Preforms Gillespie simulation on an array of quantum dots
    """
    def __init__(self, index, VL0, VR0, VU0, VD0, Q0, n0, dotArray):
        """
        Initializing a simulation,
        :param index: Simulation instance index (integer, usually the same simulation runs multiple times)
        :param VL0: Initial voltage on left electrode (double).
        :param VR0: Initial voltage on right electrode (double).
        :param VU0: Initial voltage on top electrode (double).
        :param VD0: Initial voltage on bottom electrode (double).
        :param Q0: Initial gate charges (rowsXcolumns array).
        :param n0: Initial occupations (rowsXcolumns array).
        :param dotArray: initialized array instance (A copy of this would be used for simulation).
        """
        self.dotArray = copy(dotArray)
        self.randomGenerator = np.random.RandomState()
        self.dotArray.setGateCharge(Q0)
        self.dotArray.setOccupation(n0)
        self.VL = VL0
        self.VR = VR0
        self.VD = VD0
        self.VU = VU0
        self.index = index
        self.tauLeaping = dotArray.tauLeaping
        self.min_steps = MIN_STEPS*self.dotArray.columns*self.dotArray.rows

    def getArrayParameters(self):
        """
        :return: A string representation of array parameters.
        """
        return str(self.dotArray)

    def executeLeapingStep(self, printState=False):
        """
        Executes a batch of tunnelings according to Gillespie's tau-leaping approximation.
        :param printState: If true, system state after tunnelings would be printed (for debug).
        :return: Time from last tunneling batch, to the current one.
        """
        dt, intervals = self.dotArray.getLeapingTimeInterval()
        self.dotArray.nextLeapingSteps(dt)
        if printState:
            self.printState()
        return dt + intervals*self.dotArray.default_dt

    def executeStep(self, printState=False):
        """
        Executes tunneling according to Gillespie's algorithm.
        :param printState: If true, system state after tunnelings would be printed (for debug).
        :return: Time from last tunneling, to the current one.
        """
        r = self.randomGenerator.rand(2)
        dt, intervals = self.dotArray.getTimeInterval(r[0])
        self.dotArray.nextStep(dt, r[1])
        if printState:
            self.printState()
        return dt + intervals * self.dotArray.default_dt

    def calcCurrent(self, print_stats=False, fullOutput=False, currentMap=False, double_time=False):
        """
        Calculates the steady-state current in the system (assuming it is already in a steady state).
        :param print_stats:  If true, system state after tunnelings would be printed (for debug).
        :param fullOutput: If true, occupations and charges would also be calculated.
        :param currentMap: If true, a map of the local currents would be calculated.
        :param double_time: If true, the running time for calculating averages would be doubled.
        :return: (average current in the horizontal direction,
                  standard error of the average current in the horizontal direction,
                  average current in the vertical direction,
                  standard error of the average current in the vertical direction,
                 If fullOutput is True then also
                  (average occupation, average gate charge,
                   standard error for average occupation,
                   standard error for  average gate charge)
                If currentMap is True also
                (average current map)
        """
        final_t = self.dotArray.timeStep
        if double_time:
            final_t = 2 * final_t
        curr_t = 0
        steps = 0
        I_avg = 0
        I_var = 0
        vert_I_avg = 0
        vert_I_var = 0
        if fullOutput:
            n_avg = np.zeros((self.dotArray.getRows(), self.dotArray.getColumns()))
            n_var = np.zeros(n_avg.shape)
            Q_avg = np.zeros((self.dotArray.getRows(), self.dotArray.getColumns()))
            Q_var = np.zeros(Q_avg.shape)
        if currentMap:
            map_avg = np.zeros((self.dotArray.getRows()*2 + 1, self.dotArray.getColumns() + 1))
        curr_n = self.dotArray.getOccupation()
        curr_Q = self.dotArray.getGateCharge()
        while curr_t < final_t or steps < MIN_STEPS:
            if self.tauLeaping:
                dt = self.executeLeapingStep(printState=print_stats)
            else:
                dt = self.executeStep(printState=print_stats)
            steps += 1
            curr_I, curr_vert_I = self.dotArray.getCurrentFromRates()
            I_avg, I_var = self.update_statistics(curr_I, I_avg, I_var,
                                                  curr_t, dt)
            vert_I_avg, vert_I_var = self.update_statistics(curr_vert_I, vert_I_avg, vert_I_var,
                                                  curr_t, dt)
            if fullOutput:
                n_avg, n_var = self.update_statistics(curr_n, n_avg, n_var, curr_t, dt)
                Q_avg, Q_var = self.update_statistics(curr_Q, Q_avg, Q_var, curr_t, dt)
                curr_n = self.dotArray.getOccupation()
                curr_Q = self.dotArray.getGateCharge()
            if currentMap:
                map_avg += self.dotArray.getCurrentMap()*dt
            curr_t += dt
        err = self.get_err(I_var, steps, curr_t)
        vert_err = self.get_err(vert_I_var, steps, curr_t)
        res = (I_avg, err, vert_I_avg, vert_err)
        if fullOutput:
            res = res + (n_avg, Q_avg, self.get_err(n_var, steps, curr_t), self.get_err(Q_var, steps, curr_t))
        if currentMap:
            res = res + (map_avg/curr_t,)
        return res

    def calcAverageNForGivenQ(self, Q, n0, calcVoltages=False):
        """
        Running a simulation o calculate the average occupation for the given gate charges.
        :param Q: Gae charges.
        :param n0: Initial occupations.
        :param calcVoltages: If true, voltage on each island would also be calculated.
        :return: (average occupations, standard errors for average occupations)
                if calcVoltages is true also:
                 (average voltages,
                  average voltages calculated assuming no current through gate resistors)
        """
        curr_t = 0
        steps = 0
        n_avg = np.zeros((self.dotArray.getRows(), self.dotArray.getColumns()))
        n_var = np.zeros(n_avg.shape)
        self.dotArray.setGateCharge(Q)
        self.dotArray.setOccupation(n0)
        self.dotArray.getWork()
        curr_n = n0
        err = 2*ALLOWED_ERR
        while err > ALLOWED_ERR:
            if self.tauLeaping:
                dt = self.executeLeapingStep()
            else:
                dt = self.executeStep()
            steps += 1
            n_avg, n_var = self.update_statistics(curr_n, n_avg, n_var, curr_t, dt)
            curr_n = self.dotArray.getOccupation()
            curr_t += dt
            if steps % MIN_STEPS == 0:
                err = np.max(self.get_err(n_var, steps, curr_t))
        res = (n_avg, self.get_err(n_var, steps, curr_t))
        if calcVoltages:
            self.dotArray.setOccupation(n_avg)
            v = self.dotArray.getVoltages()
            v_from_gate = self.dotArray.getVoltagesFromGate()
            res = res + (v, v_from_gate)
        return res

    def plotAverageVoltages(self):
        """
        Plotting average voltages
        """
        X = np.arange(self.dotArray.columns)
        Y = np.arange(self.dotArray.rows)
        X, Y = np.meshgrid(X, Y)
        v = self.dotArray.invC.dot(self.dotArray.getNprime() + flattenToColumn(self.dotArray.getGateCharge()))
        v2 = self.dotArray.invCeq.dot(self.dotArray.getNprime())
        Z = v.reshape(self.dotArray.rows, self.dotArray.columns)
        Z2 = v2.reshape(self.dotArray.rows, self.dotArray.columns)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.plot_surface(X, Y, Z2)
        return fig, ax

    def getToSteadyState(self):
        """
        Running simulation until a steady state has been reached.
        :return:
        """
        curr_n = self.dotArray.getOccupation()
        curr_Q = self.dotArray.getGateCharge()
        allowed_err = ALLOWED_ERR / (self.dotArray.getRows())
        err = allowed_err * 2
        not_decreasing = 0
        while err > allowed_err and not_decreasing < STEADY_STATE_REP:
            n_avg = np.zeros((self.dotArray.getRows(), self.dotArray.getColumns()))
            n_var = np.zeros(n_avg.shape)
            Q_avg = np.zeros(n_avg.shape)
            Q_var = np.zeros(n_avg.shape)
            curr_t = 0
            while curr_t < self.dotArray.timeStep:
                if self.tauLeaping:
                    dt = self.executeLeapingStep()
                else:
                    dt = self.executeStep()
                n_avg, n_var = self.update_statistics(curr_n, n_avg, n_var,
                                                      curr_t, dt)
                Q_avg, Q_var = self.update_statistics(curr_Q, Q_avg, Q_var,
                                                      curr_t, dt)
                curr_n = self.dotArray.getOccupation()
                curr_Q = self.dotArray.getGateCharge()
                curr_t += dt
            new_err = self.dotArray.get_dist_from_steady(n_avg, Q_avg)
            if err < new_err:
                not_decreasing += 1
                if not_decreasing > STEADY_STATE_REP:
                    print("No convergence")
            err = new_err
        return True

    def saveState(self, I, IErr, vertI, vertIErr, n=None, Q=None, nErr=None, QErr=None, Imaps=None,
                  fullOutput=False, currentMap=False, basePath=''):
        """
        Saving the current state of the simulation.
        :param I: Average currents in the horizontal direction.
        :param IErr: Standard errors for the average currents in the horizontal direction.
        :param vertI: Average currents in the vertical direction.
        :param vertIErr: Standard errors for the average currents in the vertical direction.
        :param n: Average occupations.
        :param Q: Average gate charges.
        :param nErr: Standard errors for the average occupations.
        :param QErr: Standard errors for the average gate charges.
        :param Imaps: Local currents map.
        :param fullOutput: If true, full output will be saved.
        :param currentMap: If true, local current maps would be saved.
        :param basePath: location for storing the files.
        """
        baseName = basePath + "_temp_" + str(self.index)
        if fullOutput:
            np.save(baseName + "_ns", np.array(n))
            np.save(baseName + "_Qs", np.array(Q))
            np.save(baseName + "_nsErr", np.array(nErr))
            np.save(baseName + "_QsErr", np.array(QErr))
        np.save(baseName + "_n", self.dotArray.getOccupation())
        np.save(baseName + "_Q", self.dotArray.getGateCharge())
        if currentMap:
            np.save(baseName + "_current_map", np.array(Imaps))
        np.save(baseName + "_I", np.array(I))
        np.save(baseName + "_IErr", np.array(IErr))
        np.save(baseName + "_vertI", np.array(vertI))
        np.save(baseName + "_vertIErr", np.array(vertIErr))

    def loadState(self,  fullOutput=False, currentMap=False, basePath=''):
        """
        Loading a pre-saved system state
        :param fullOutput: If true, full output would be loaded.
        :param currentMap: If true, local current maps would be loaded.
        :param basePath: path for location where files are stored.
        :return: loaded state of the system.
        """
        baseName = basePath + "_temp_" + str(self.index)
        if not os.path.isfile(baseName + "_I.npy"):
            return None
        try:
            I = np.load(baseName + "_I.npy")
            loadLen = len(I)
            IErr = np.load(baseName + "_IErr.npy")
            vertI = np.load(baseName + "_vertI.npy")
            vertIErr = np.load(baseName + "_vertIErr.npy")
            if len(IErr) > loadLen:
                IErr = IErr[:loadLen]
            if len(vertI) > loadLen:
                vertI = vertI[:loadLen]
            if len(vertIErr) > loadLen:
                vertIErr = vertIErr[:loadLen]
            n = np.load(baseName + "_n.npy")
            Q = np.load(baseName + "_Q.npy")
            res = (I, IErr, vertI, vertIErr, n, Q)
            if fullOutput:
                ns = np.load(baseName + "_ns.npy")
                if len(ns) > loadLen:
                    ns = ns[:loadLen, :, :]
                Qs = np.load(baseName + "_Qs.npy")
                if len(Qs) > loadLen:
                    Qs = Qs[:loadLen, :, :]
                nsErr = np.load(baseName + "_nsErr.npy")
                if len(nsErr) > loadLen:
                    nsErr = nsErr[:loadLen, :, :]
                QsErr = np.load(baseName + "_QsErr.npy")
                if len(QsErr) > loadLen:
                    QsErr = QsErr[:loadLen, :, :]
                res = res + (ns, Qs, nsErr, QsErr)
            if currentMap:
                Imaps = np.load(baseName + "_current_map.npy")
                if len(Imaps) > loadLen:
                    Imaps = Imaps[:loadLen, :, :]
                res = res + (Imaps,)
            return res
        except ValueError:
            print("An error has occurred while loading. Starting a fresh run.")
            return None

    def calcIV(self, Vmax, Vstep, vSym, fullOutput=False, print_stats=False,
               currentMap=False, basePath="", resume=False, double_loop=False, double_time=False,
               average_vertical_directions=False):
        """
        Calculating the I-V curve of the system while voltage increase from Vmin to Vmax and back again..
        :param Vmax: Maximum external voltage.
        :param Vstep: Voltage step size.
        :param vSym: If true, external voltage would be symmetric, such that VL+VR = initial VR.
        :param fullOutput: If true, occupations and gate charges would also be calculated.
        :param print_stats: If true, the state would be printed after each step (for debug)
        :param currentMap: If true, local current maps would be calculated.
        :param basePath: path for storing results.
        :param resume: If true, will try to resume simulation from the last saved state.
        :param double_loop: If true, V would increase from minimum to maximum and back again twice.
        :param double_time: If true, the running time for calculating averages would be doubled.
        :param average_vertical_directions: if true will run one time with given VU, VD and then
         swap them and run again, results for current in the horizontal direction would be the sqrt of
          the squares mean in this case.
        :return: (average currents in the horizontal direction,
                  standard errors of the average currents in the horizontal direction,
                  average currents in the vertical direction,
                  standard errors of the average currents in the vertical direction,
                  external voltages)
                 If fullOutput is True then also
                  (average occupations, average gate charges,
                   standard error for average occupations,
                   standard error for  average gate charges)
                If currentMap is True also
                (average current maps)
        """
        I = []
        IErr = []
        vertI = []
        vertIErr = []
        ns = []
        Qs = []
        nsErr = []
        QsErr = []
        Imaps = []
        if vSym:
            Vstep /= 2
            Vmax /= 2
            VR_vec = np.arange(self.VR-(self.VL/2), self.VR - Vmax, -Vstep)
            VR_vec = np.hstack((VR_vec, np.flip(VR_vec)))
            VL_vec = np.arange(self.VL/2+self.VR, Vmax + self.VR, Vstep)
            VL_vec = np.hstack((VL_vec, np.flip(VL_vec)))
        else:
            VL_vec = np.arange(self.VL, Vmax+self.VR, Vstep)
            VL_vec = np.hstack((VL_vec, np.flip(VL_vec)))
            VR_vec = self.VR * np.ones(VL_vec.shape)
        if double_loop:
            VL_vec = np.hstack((VL_vec, VL_vec))
            VR_vec = np.hstack((VR_vec, VR_vec))
        VL_res = np.copy(VL_vec)
        VR_res = np.copy(VR_vec)
        if resume:
            resumeParams = self.loadState(fullOutput=fullOutput, currentMap=currentMap, basePath=basePath)
            if resumeParams is not None:
                I = list(resumeParams[0])
                IErr = list(resumeParams[1])
                vertI = list(resumeParams[2])
                vertIErr = list(resumeParams[3])
                Vind = len(I)
                VL_vec = VL_vec[Vind:]
                VR_vec = VR_vec[Vind:]
                self.dotArray.setOccupation(resumeParams[4])
                self.dotArray.setGateCharge(resumeParams[5])
                if fullOutput:
                    ns = list(resumeParams[6])
                    Qs = list(resumeParams[7])
                    nsErr = list(resumeParams[8])
                    QsErr = list(resumeParams[9])
                if currentMap:
                    Imaps = list(resumeParams[-1])
        for VL, VR in zip(VL_vec, VR_vec):
            self.dotArray.changeVext(VL, VR, self.VU, self.VD)
            # running once to get to steady state
            self.getToSteadyState()
            # now we are in steady state calculate current
            stepRes1 = self.calcCurrent(print_stats=print_stats, fullOutput=fullOutput, currentMap=currentMap,
                                        double_time=double_time)
            if average_vertical_directions:
                self.dotArray.changeVext(VL, VR, self.VD, self.VU)
                # running once to get to steady state
                self.getToSteadyState()
                # now we are in steady state calculate current
                stepRes2 = self.calcCurrent(print_stats=print_stats, fullOutput=fullOutput, currentMap=False,
                                            double_time=double_time)
                current = (stepRes1[0] + stepRes2[0])/2
                currentErr = (stepRes1[1] + stepRes2[1])/2
                vert_current = np.sqrt((stepRes1[2]**2 + stepRes2[2]**2)/2)
                vert_currentErr = np.sqrt((stepRes1[3]**2 + stepRes2[3]**2)/2)
                if fullOutput:
                    ns.append((stepRes1[4] + stepRes2[4])/2)
                    Qs.append((stepRes1[5] + stepRes1[5])/2)
                    nsErr.append((stepRes1[6] + stepRes2[6])/2)
                    QsErr.append((stepRes1[7] + stepRes2[7])/2)
            else:
                current = stepRes1[0]
                currentErr = stepRes1[1]
                vert_current = stepRes1[2]
                vert_currentErr = stepRes1[3]
                if fullOutput:
                    ns.append(stepRes1[4])
                    Qs.append(stepRes1[5])
                    nsErr.append(stepRes1[6])
                    QsErr.append(stepRes1[7])
            if currentMap:
                Imaps.append(stepRes1[-1])
            I.append(current)
            IErr.append(currentErr)
            vertI.append(vert_current)
            vertIErr.append(vert_currentErr)
            if self.index == 0:
                print("%.3f" % float(VL - VR), end=',', flush=True)
            self.saveState(I, IErr, vertI, vertIErr, ns, Qs, nsErr, QsErr, Imaps, fullOutput=fullOutput,
                           currentMap=currentMap, basePath=basePath)
        res = (np.array(I), np.array(IErr), np.array(vertI), np.array(vertIErr), VL_res - VR_res)
        if fullOutput:
            res = res + (np.array(ns), np.array(Qs), np.array(nsErr), np.array(QsErr))
        if currentMap:
            res = res + (np.array(Imaps),)
        return res

    def calcIT(self, Tmax, Tstep, fullOutput=False, print_stats=False,
               currentMap=False, basePath="", resume=False, double_time=False,
               average_vertical_directions=False):
        """
        Calculating the I-T curve (current as a function of temperature) of the system while voltage increase from
         Vmin to Vmax and back again..
        :param Tmax: Maximum temperature.
        :param Tstep: Temperature step size.
        :param fullOutput: If true, occupations and gate charges would also be calculated.
        :param print_stats: If true, the state would be printed after each step (for debug)
        :param currentMap: If true, local current maps would be calculated.
        :param basePath: path for storing results.
        :param resume: If true, will try to resume simulation from the last saved state.
        :param average_vertical_directions: if true will run one time with given VU, VD and then
         swap them and run again, results for current in the horizontal direction would be the sqrt of
          the squares mean in this case.
        :param double_time: If true, the running time for calculating averages would be doubled.
        :return: (average currents in the horizontal direction,
                  standard errors of the average currents in the horizontal direction,
                  average currents in the vertical direction,
                  standard errors of the average currents in the vertical direction,
                  temperatures)
                 If fullOutput is True then also
                  (average occupations, average gate charges,
                   standard error for average occupations,
                   standard error for  average gate charges)
                If currentMap is True also
                (average current maps)
        """
        I = []
        IErr = []
        vertI = []
        vertIErr = []
        ns = []
        Qs = []
        nsErr = []
        QsErr = []
        Imaps = []
        T_vec = np.arange(self.dotArray.getTemperature(), Tmax, Tstep)
        T_res = np.copy(T_vec)
        if resume:
            resumeParams = self.loadState(fullOutput=fullOutput, currentMap=currentMap, basePath=basePath)
            if resumeParams is not None:
                I = list(resumeParams[0])
                IErr = list(resumeParams[1])
                vertI = list(resumeParams[2])
                vertIErr = list(resumeParams[3])
                Tind = len(I)
                T_vec = T_vec[Tind:]
                self.dotArray.setOccupation(resumeParams[4])
                self.dotArray.setGateCharge(resumeParams[5])
                if fullOutput:
                    ns = list(resumeParams[6])
                    Qs = list(resumeParams[7])
                    nsErr = list(resumeParams[8])
                    QsErr = list(resumeParams[9])
                if currentMap:
                    Imaps = list(resumeParams[-1])
        for T in T_vec:
            self.dotArray.setTemperature(T)
            self.dotArray.changeVext(self.VL, self.VR, self.VU, self.VD)
            # running once to get to steady state
            self.getToSteadyState()
            # now we are in steady state calculate current
            stepRes1 = self.calcCurrent(print_stats=print_stats, fullOutput=fullOutput, currentMap=currentMap,
                                        double_time=double_time)
            if average_vertical_directions:
                self.dotArray.changeVext(self.VL, self.VR, self.VD, self.VU)
                # running once to get to steady state
                self.getToSteadyState()
                # now we are in steady state calculate current
                stepRes2 = self.calcCurrent(print_stats=print_stats, fullOutput=fullOutput, currentMap=False,
                                            double_time=double_time)
                current = (stepRes1[0] + stepRes2[0]) / 2
                currentErr = (stepRes1[1] + stepRes2[1]) / 2
                vert_current = np.sqrt((stepRes1[2] ** 2 + stepRes2[2] ** 2) / 2)
                vert_currentErr = np.sqrt((stepRes1[3] ** 2 + stepRes2[3] ** 2) / 2)
                if fullOutput:
                    ns.append((stepRes1[4] + stepRes2[4]) / 2)
                    Qs.append((stepRes1[5] + stepRes1[5]) / 2)
                    nsErr.append((stepRes1[6] + stepRes2[6]) / 2)
                    QsErr.append((stepRes1[7] + stepRes2[7]) / 2)
            else:
                current = stepRes1[0]
                currentErr = stepRes1[1]
                vert_current = stepRes1[2]
                vert_currentErr = stepRes1[3]
                if fullOutput:
                    ns.append(stepRes1[4])
                    Qs.append(stepRes1[5])
                    nsErr.append(stepRes1[6])
                    QsErr.append(stepRes1[7])
            if currentMap:
                Imaps.append(stepRes1[-1])
            I.append(current)
            IErr.append(currentErr)
            vertI.append(vert_current)
            vertIErr.append(vert_currentErr)
            if self.index == 0:
                print("%.3f" % float(T), end=',', flush=True)
            self.saveState(I, IErr, vertI, vertIErr, ns, Qs, nsErr, QsErr, Imaps=Imaps, fullOutput=fullOutput,
                           currentMap=currentMap, basePath=basePath)
        res = (np.array(I), np.array(IErr), np.array(vertI), np.array(vertIErr), T_res)
        if fullOutput:
            res = res + (np.array(ns), np.array(Qs), np.array(nsErr), np.array(QsErr))
        if currentMap:
            res = res + (np.array(Imaps),)
        return res

    def printState(self):
        """
        Prints the current state of the array
        """
        self.dotArray.printState()

    def update_statistics(self, value, avg, n_var, total_time, time_step):
        """
        Updating the statistics of a measured value according to West's
        algorithm (as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2973983/)
        :param value: Measured value this step
        :param avg: Average up to this step (result of last call to this function)
        :param n_var: non-normalized variance up to this step (result of last call to this function)
        :param total_time: Total simulation time not including last step
        :param time_step: The time of each step.
        :return: (avg, n_var) Updated values.
        to get the actual variance use Var = (n*n_var)/((n-1)*total_time)
        """
        new_time = total_time + time_step
        dist_from_avg = value - avg
        local_std = dist_from_avg*time_step / new_time
        new_n_var = n_var + dist_from_avg*total_time*local_std
        new_avg = avg + local_std
        return new_avg, new_n_var

    def get_err(self, n_var, steps, time):
        """
        Returns the standard error of mean of a variable based on the
        non-normalized variance (see method "update_statistics")
        :param n_var: non-normalized variance
        :param steps: number of steps for data point
        :param time: total running time for data point
        :return: standard deviation
        """
        return np.sqrt((1/((steps-1)*time))*n_var)


class GraphSimulator:
    """
    Calculating steady state current for an array of quantum dots by using "graph method"
    """
    def __init__(self, index, VL0, VR0, VU0, VD0, Q0, n0, dotArray):
        """
        Initializing simulator in graph method
        :param index: instance index (usually many simulations run in parallel, integer)
        :param VL0: Initial voltage on left electrode (double).
        :param VR0: Initial voltage on right electrode (double).
        :param VU0: Initial voltage on top electrode (double).
        :param VD0: Initial voltage on bottom electrode (double).
        :param Q0: Initial charge on gate capacitors (rowsXcolumns array of doubles).
        :param n0: Initial occupation on islands (rowsXcolumns array of integers).
        :param dotArray: Initialized array instance, a copy would be made.
        """
        self.dotArray = copy(dotArray)
        self.n0 = n0
        self.QG = Q0
        self.VL = VL0
        self.VR = VR0
        self.VU = VU0
        self.VD = VD0
        self.counter = 0
        self.index = index
        self.edgesMat = None
        self.states = None
        self.prob = None
        self.lyaponuv = None
        self.rates_diff_left = None
        self.rates_diff_down = None

    def reshape_to_array(self, a):
        """
        Reshape the given array to rowsXcolumns
        :param a: numpy array with size columns*rows
        :return: The reshaped array
        """
        return a.reshape((self.dotArray.getRows(), self.dotArray.getColumns()))

    def getArrayParameters(self):
        """
        :return: String representation os array parameters
        """
        return str(self.dotArray)

    def buildGraph(self, Q):
        """
        Building the master equation graph representation for the given gate charge.
        :param Q: Given gate charge.
        """
        states = []  # vertices
        states_dict = dict()  # to efficiently check if a vertex already exists.
        edges = []
        self.dotArray.setGateCharge(np.copy(Q))
        self.dotArray.setOccupation(np.copy(self.n0))
        states.append(np.copy(self.n0).flatten())
        tup_n = tuple(self.n0.flatten())
        states_dict[tup_n] = 0
        left_rates_diff = []
        down_rates_diff = []
        current_state_ind = 0
        next_state_ind = 1
        edges_line = [0]
        while current_state_ind < len(states):  # For each state, finding more states it is connected to.
            self.dotArray.setOccupation(self.reshape_to_array(np.copy(states[current_state_ind])))
            rates = self.dotArray.getRates()
            for ind, rate in enumerate(rates):
                if rate > 0:  # add new edge
                    fromDot, toDot = self.dotArray.executeAction(ind, 1)
                    n = self.dotArray.getOccupation().flatten()
                    tup_n = tuple(n)
                    if tup_n not in states_dict:
                        states_dict[tup_n] = next_state_ind
                        states.append(n)
                        next_state_ind += 1
                        edges_line.append(rate)
                    else:
                        edges_line[states_dict[tup_n]] += rate
                    self.dotArray.tunnel(toDot, fromDot, 1)
            edges.append(edges_line)
            edges_line = [0] * len(edges_line)
            current_state_ind += 1
            left_diff,  down_diff = self.get_edge_rates_diff(rates)
            left_rates_diff.append(left_diff)
            down_rates_diff.append(down_diff)
        edgesMat = np.zeros((len(edges), len(edges[-1])))
        for ind, line in enumerate(edges):
            edgesMat[ind, :len(line)] = line
        diagonal = np.sum(edgesMat, axis=1)  # Rates to go out from each state goes to the diagonal of the matrix.
        self.edgesMat = edgesMat.T - np.diagflat(diagonal)
        self.states = np.array(states)
        self.rates_diff_left = np.array(left_rates_diff)
        self.rates_diff_down = np.array(down_rates_diff)

    def find_probabilities(self, Q):
        """
        Finding a steady-state solution to the master equation.
        :param Q: Charge on each gate capacitor.
        """
        self.buildGraph(Q)
        # steady state probabilities are the belong to null-space of edgesMat.T and sum of probabilities must be 1
        a = np.vstack((self.edgesMat, np.ones((1, self.edgesMat.shape[1]))))
        b = np.zeros((a.shape[0], 1))
        b[-1, 0] = 1  # probabilities sum up to 1
        self.prob = np.linalg.lstsq(a, b, rcond=None)[0]  # solving the master equation for a steady state solution
        return True

    def get_average_state(self, Q):
        """
        Calculating average occupations on steady state for the given gate charges.
        :param Q: Gate charges.
        :return: Average occupations (rowsXcolumns array)
        """
        self.find_probabilities(Q)
        average_state = np.sum(np.multiply(self.states, self.prob), axis=0)
        return self.reshape_to_array(average_state)

    def get_average_voltages(self, Q):
        """
        Calculates the average voltage on each island, for a given gate charges.
        :param Q: Gate charges.
        :return: Average voltages (rowsXcolumns array).
        """
        n_copy = self.dotArray.getOccupation()
        Q_copy = self.dotArray.getGateCharge()
        n = self.get_average_state(Q)
        self.dotArray.setOccupation(n)
        self.dotArray.setOccupation(Q)
        v = self.dotArray.getVoltages()
        self.dotArray.setOccupation(n_copy)
        self.dotArray.setGateCharge(Q_copy)
        return v

    def get_voltages_from_gate(self, Q):
        """
        Calculates the average voltage on each island, for a given gate charges, assuming zero current through
         gate resistors.
        :param Q: Gate charges.
        :return: Average voltages (rowsXcolumns array).
        """
        Q_copy = self.dotArray.getGateCharge()
        self.dotArray.setGateCharge(Q)
        v = self.dotArray.getVoltagesFromGate()
        self.dotArray.setGateCharge(Q_copy)
        return v

    def calc_lyaponuv_integrand(self, Q):
        """
        calculates the lyapunov functional integrand for the given gate charges.
        :param Q: Given gate charges.
        :return: Integrand.
        """
        n_copy = self.dotArray.getOccupation()
        Q_copy = self.dotArray.getGateCharge()
        n = self.get_average_state(Q)
        self.dotArray.setOccupation(n)
        self.dotArray.setGateCharge(Q)
        diff_from_equi = np.sum(flattenToColumn(Q) - self.dotArray.get_steady_Q_for_n())
        self.dotArray.setOccupation(n_copy)
        self.dotArray.setGateCharge(Q_copy)
        return diff_from_equi

    def calc_on_grid(self, Qmin, Qmax, dq, f, res_len=1):
        """
        Calculates the values of f on a grid from Qmin to Qmax with dq steps. (Q can have more than one dimension).
        :param Qmin: Vector of minimal Q values for each dimension.
        :param Qmax: Vector of maximal Q values for each dimension.
        :param dq: Q step (same for each dimension).
        :param f: The function to calculate.
        :param res_len: how many dimensions the result have for each point.
        :return: Q grid, a list of results grid (each element on th list is a result for one dimension).
        """
        Qmin = Qmin.flatten()
        Qmax = Qmax.flatten()
        coordinates = [np.arange(Qmin[i], Qmax[i], dq) for i in range(Qmin.size)]
        grid = np.meshgrid(*coordinates, indexing='ij')
        grid_array = np.moveaxis(np.array(grid), 0, -1)
        res = [np.zeros(grid[0].shape) for i in range(res_len)]
        it = np.nditer(grid[0], flags=['multi_index'])
        while not it.finished:
            index = it.multi_index
            curr_Q = grid_array[index]
            for i in range(res_len):
                f_res = f(curr_Q).flatten()
                res[i][index] = f_res[i]
            it.iternext()
        return grid_array, res

    def set_lyaponuv(self, Qmin, Qmax, dq):
        """
        Calculates lyaponuv function on a grid
        :param Qmin: Vector of minimal Q values for each dimension.
        :param Qmax: Vector of maximal Q values for each dimension.
        :param dq: Q step (same for each dimension).
        """
        self.Q_grid, diff_from_eq = self.calc_on_grid(Qmin, Qmax, dq, self.calc_lyaponuv_integrand)
        res = diff_from_eq[0]
        for axis in range(len(res.shape)):
            res = cumtrapz(res, dx=dq, axis=axis, initial=0)
        self.lyaponuv = res

    def calc_lyaponuv_grad(self, Q):
        """
        Calculates the gradient of the lyaponuv functional for a given gate charges.
        :param Q: The given gate charges.
        :return: Lyapponuv gradient (1D array with the same size as Q).
        """
        n = self.get_average_state(self.reshape_to_array(Q))
        self.dotArray.setOccupation(n)
        self.dotArray.setGateCharge(Q)
        return Q.flatten() - self.dotArray.get_steady_Q_for_n().flatten()

    def plot_average_voltages(self, Qmin, Qmax, dq):
        """
        Plotting the average voltages (good for arrays with 2 islands only)
        :param Qmin: Vector of minimal Q values for each dimension.
        :param Qmax: Vector of maximal Q values for each dimension.
        :param dq: Q step (same for each dimension).
        """
        from mayavi import mlab
        Q_grid, voltages = self.calc_on_grid(Qmin, Qmax, dq, self.get_average_voltages, res_len=Qmin.size)
        _, voltages_from_gate = self.calc_on_grid(Qmin, Qmax, dq, self.get_voltages_from_gate, res_len=Qmin.size)
        fig1 = mlab.figure()
        surf1 = mlab.surf(Q_grid[:, :, 0], Q_grid[:, :, 1], voltages[0], colormap='Blues')
        surf2 = mlab.surf(Q_grid[:, :, 0], Q_grid[:, :, 1], voltages_from_gate[0], colormap='Oranges')
        fig1 = mlab.figure()
        surf1 = mlab.surf(Q_grid[:, :, 0], Q_grid[:, :, 1], voltages[1], colormap='Blues')
        surf2 = mlab.surf(Q_grid[:, :, 0], Q_grid[:, :, 1], voltages_from_gate[1], colormap='Oranges')
        mlab.show()

    def find_next_QG_using_lyaponuv(self, dq):
        """
        Locate the nearest minima of Lyapunov functional by calculating the functional and detecting peaks.
        :param dq: q steps.
        """
        q_shift = Q_SHIFT
        peaks = (np.array([]),)
        while not peaks[0].size:
            self.set_lyaponuv(self.QG - q_shift, self.QG + q_shift, dq)
            q_shift *= 2
            peaks = detect_local_minima(self.lyaponuv)
        Qind = np.argmin(np.sum((self.Q_grid[peaks] - self.QG)**2, axis=1))
        self.QG = self.Q_grid[peaks][Qind]

    def find_next_QG_using_gradient_descent(self):
        """
        Locate the nearest minima of Lyapunov functional by a simple gradient descent.
        """
        flag = False
        rep = 0
        lr = INI_LR
        while (not flag) and rep < GRAD_REP:
            res, flag = simple_gradient_descent(self.calc_lyaponuv_grad, self.QG, lr=lr, plot_lc=False, max_iter=100000,
                                               eps=1e-5)
            rep += 1
            lr = lr/10
        if flag:  # if gradient descent did not converge skipping the point
            self.QG = self.reshape_to_array(res)
        else:
            print("gradient descent didn't converge")

    def calcCurrent(self, fullOutput=False):
        """
        Calculates the steady-state current in the system (assuming it is already in a steady state).
        :param fullOutput: If true, occupations and charges would also be calculated.
        :return: (average current in the horizontal direction,
                  average current in the vertical direction,
                 If fullOutput is True then also
                  (average occupation, average gate charge)
        """
        self.find_next_QG_using_gradient_descent()
        n_avg = self.reshape_to_array(self.get_average_state(self.QG))
        self.n0 = np.floor(n_avg)
        left_current = np.sum(self.prob*self.rates_diff_left)
        down_current = np.sum(self.prob*self.rates_diff_down)
        res = (left_current, down_current)
        if fullOutput:
            avg_Q = self.reshape_to_array(self.QG)
            res += (n_avg, avg_Q)
        return res

    def get_edge_rates_diff(self, rates):
        """
        Calculate current by tunneling rates.
        :param rates: Tunneling rates
        :return: vertical current, horizontal current
        """
        horzSize = self.dotArray.rows * (self.dotArray.columns + 1)
        vertSize = (self.dotArray.rows + 1) * self.dotArray.columns
        right_tunneling_rates = rates[:horzSize]
        left_tunneling_rates = rates[horzSize:2*horzSize]
        down_tunneling_rates = rates[2*horzSize:2*horzSize + vertSize]
        up_tunneling_rates = rates[2*horzSize + vertSize:2*horzSize + 2*vertSize]
        left_diff = right_tunneling_rates[::self.dotArray.columns + 1] - \
                    left_tunneling_rates[::self.dotArray.columns + 1]
        right_diff = right_tunneling_rates[self.dotArray.columns:self.dotArray.columns + 1] - \
                     left_tunneling_rates[self.dotArray.columns:self.dotArray.columns + 1]
        top_diff = down_tunneling_rates[:self.dotArray.columns] - up_tunneling_rates[:self.dotArray.columns]
        bottom_diff = down_tunneling_rates[-self.dotArray.columns:] - up_tunneling_rates[-self.dotArray.columns:]
        return (left_diff + right_diff)/2, (top_diff + bottom_diff/2)

    def saveState(self, I, n=None, Q=None, fullOutput=False, basePath=''):
        """
        Saving the current state of simulation
        :param I: Calculated currents.
        :param n: Calculated average occupations.
        :param Q: Calculated average gate charges.
        :param fullOutput: If full output was calculated
        :param basePath: Pah for saving files.
        """
        baseName = basePath + "_temp_" + str(self.index)
        if fullOutput:
            np.save(baseName + "_ns", np.array(n))
            np.save(baseName + "_Qs", np.array(Q))
        np.save(baseName + "_n", self.n0)
        np.save(baseName + "_Q", self.QG)
        np.save(baseName + "_I", np.array(I))

    def loadState(self, fullOutput=False, basePath=''):
        """
        Loading the state of a previously interrupt simulation.
        :param fullOutput: True of full output was saved.
        :param basePath: Directory where data is stored.
        :return: Loaded data.
        """
        baseName = basePath + "_temp_" + str(self.index)
        I = np.load(baseName + "_I.npy")
        n = np.load(baseName + "_n.npy")
        Q = np.load(baseName + "_Q.npy")
        res = (I, n, Q)
        if fullOutput:
            ns = np.load(baseName + "_ns.npy")
            Qs = np.load(baseName + "_Qs.npy")
            res = res + (ns, Qs)
        return res

    def calcIV(self, Vmax, Vstep, vSym, fullOutput=False, print_stats=False,
               currentMap=False, basePath="", resume=False, double_loop=False,
               average_vertical_directions=False):
        """
        Calculating the I-V curve of the system while voltage increase from Vmin to Vmax and back again..
        :param Vmax: Maximum external voltage.
        :param Vstep: Voltage step size.
        :param vSym: If true, external voltage would be symmetric, such that VL+VR = initial VR.
        :param fullOutput: If true, occupations and gate charges would also be calculated.
        :param print_stats: Not implemented here.
        :param currentMap: If true, local current maps would be calculated.
        :param basePath: path for storing results.
        :param resume: If true, will try to resume simulation from the last saved state.
        :param double_loop: If true, V would increase from minimum to maximum and back again twice.
        :param average_vertical_directions: if true will run one time with given VU, VD and then
         swap them and run again, results for current in the horizontal direction would be the sqrt of
          the squares mean in this case.
        :return: (average currents in the horizontal direction,
                  standard errors of the average currents in the horizontal direction,
                  average currents in the vertical direction,
                  standard errors of the average currents in the vertical direction,
                  external voltages)
                 If fullOutput is True then also
                  (average occupations, average gate charges,
                   standard error for average occupations,
                   standard error for  average gate charges)
                If currentMap is True also
                (average current maps)
        """
        I = []
        Ivert = []
        ns = []
        Qs = []
        if vSym:
            Vstep /= 2
            Vmax /= 2
            VR_vec = np.arange(self.VR - (self.VL / 2), self.VR - Vmax, -Vstep)
            VR_vec = np.hstack((VR_vec, np.flip(VR_vec)))
            VL_vec = np.arange(self.VL / 2 + self.VR, Vmax + self.VR, Vstep)
            VL_vec = np.hstack((VL_vec, np.flip(VL_vec)))
        else:
            VL_vec = np.arange(self.VL, Vmax + self.VR, Vstep)
            VL_vec = np.hstack((VL_vec, np.flip(VL_vec)))
            VR_vec = self.VR * np.ones(VL_vec.shape)
        if double_loop:
            VL_vec = np.hstack((VL_vec, VL_vec))
            VR_vec = np.hstack((VR_vec, VR_vec))
        VL_res = np.copy(VL_vec)
        VR_res = np.copy(VR_vec)
        if resume:
            resumeParams = self.loadState(fullOutput=fullOutput, basePath=basePath)
            I = list(resumeParams[0])
            Vind = len(I)
            VL_vec = VL_vec[Vind:]
            VR_vec = VR_vec[Vind:]
            self.n0 = resumeParams[1]
            self.QG = resumeParams[2]
            if fullOutput:
                ns = list(resumeParams[3])
                Qs = list(resumeParams[4])
        for VL, VR in zip(VL_vec, VR_vec):
            if self.index == 0:
                print("%.3f" % float(VL-VR), end=',', flush=True)
            self.dotArray.changeVext(VL, VR, self.VU, self.VD)
            res = self.calcCurrent(fullOutput=fullOutput)
            if average_vertical_directions:
                self.dotArray.changeVext(VL, VR, self.VD, self.VU)
                res2 = self.calcCurrent(fullOutput=fullOutput)
                horizontalCurrent = (res[0] + res2[0])/2
                verticalCurrent = np.sqrt((res[1]**2 + res2[1]**2)/2)
                if fullOutput:
                    n = (res[2] + res2[2])/2
                    Q = (res[3] + res2[3])/2
                    ns.append(n)
                    Qs.append(Q)
            else:
                horizontalCurrent = res[0]
                verticalCurrent = res[1]
                if fullOutput:
                    n = res[2]
                    Q = res[3]
                    ns.append(n)
                    Qs.append(Q)
            I.append(horizontalCurrent)
            Ivert.append(verticalCurrent)
            self.saveState(I, ns, Qs, fullOutput=fullOutput, basePath=basePath)
        result = (np.array(I), np.zeros((len(I),)), VL_res - VR_res)
        if fullOutput:
            result = result + (ns, Qs, np.zeros((len(ns),)), np.zeros((len(Qs),)))
        return result


#  Main functions for running simulations  #
def runSingleSimulation(index, VL0, VR0, VU0, VD0, vSym, Q0, n0, Vmax, Vstep, dotArray,
                        fullOutput=False, printState=False, useGraph=False, currentMap=False,
                        basePath="", resume=False, double_time=False, double_loop=False,
                        calcIT=False, average_vertical_direction=False):
    """
    Running a single simulation instance
    :param index: Simulation instance index (Usually many instances run in parallel and we want to be able
     to know which is which).
    :param VL0: Initial voltage on the left electrode.
    :param VR0: Initial voltage on the right electrode.
    :param VU0: Initial voltage on the top electrode.
    :param VD0: Initial voltage on the bottom electrode.
    :param vSym: If true, voltage would be raised symmetrically on both sides, otherwise only VL changes.
    :param Q0: Initial charges on gate capacitors (rowsXcolumns array)
    :param n0: Initial occupations of islands (rowsXcolumns array of integers)
    :param Vmax: Maximal external voltage.
    :param Vstep: External voltage step size.
    :param dotArray: An initialized array, a copy would be made before running the simulation.
    So the same array could be used for many simulation instances.
    :param fullOutput: If true, average occupations and gate charges would be calculated too.
    :param printState: If true, the state of the array would be printed after every tunneling. For debug.
    :param useGraph: If true, the graph method will be used (use this only for very small arrays). Otherwise,
    Gillespie's KMC algorithm will be used.
    :param currentMap: If true, a map of local currents will be calculated. Not implemented for graph method.
    :param basePath: Directory in which results would be saved. In the same directory, temporary state of the
     simulation would be saved so that simulation could be resumed in case of an interruption.
    :param resume: If true, will try to resume simulation from a saved state.
    :param double_time: If true, each simulation step time would be doubled.
    :param double_loop: If true, voltage cycle would be done twice.
    :param calcIT: If true, the current would be calculated as a function of temperature, instead of voltage.
    Not implemented for graph method.
   :param average_vertical_direction: if true will run one time with given VU, VD and then
         swap them and run again, results for current in the horizontal direction would be the sqrt of
          the squares mean in this case.
    :return: Simulation results:
             (average currents in the horizontal direction,
              standard errors of the average currents in the horizontal direction,
              average currents in the vertical direction,
              standard errors of the average currents in the vertical direction,
              external voltages)
             If fullOutput is True then also
              (average occupations, average gate charges,
               standard error for average occupations,
               standard error for  average gate charges)
            If currentMap is True also
            (average current maps)
            And in the end, array parameters.
    """
    if useGraph:
        if calcIT:
            print("Graph method is only possible for T=0, cannot calculate I-T curves using this method.")
            exit(0)
        if currentMap:
            print("Warning: currents map calculation using graph method is not implemented, running without this"
                  " calculation.")
        simulator = GraphSimulator(index, VL0, VR0, VU0, VD0, Q0, n0, dotArray)
    else:
        simulator = Simulator(index, VL0, VR0, VU0, VD0, Q0, n0, dotArray)
    if calcIT:
        out = simulator.calcIT(Vmax, Vstep, fullOutput=fullOutput, print_stats=printState,
                               currentMap=currentMap, basePath=basePath, resume=resume, double_time=double_time,
                               average_vertical_directions=average_vertical_direction)
    else:
        out = simulator.calcIV(Vmax, Vstep, vSym, fullOutput=fullOutput, print_stats=printState,
                               currentMap=currentMap, basePath=basePath, resume=resume,
                               double_loop=double_loop, double_time=double_time,
                               average_vertical_directions=average_vertical_direction)
    array_params = simulator.getArrayParameters()
    return out + (array_params,)


def saveRandomParams(VG, Q0, n0, CG, RG, Ch, Cv, Rh, Rv, basePath):
    """
    Saving array params for resuming failed simulation.
    :param VG: Gate voltages.
    :param Q0: Initial gate charges.
    :param n0: Initial occupations.
    :param CG: Gate capacitance.
    :param RG: Gate resistance.
    :param Ch: Horizontal tunneling junctions capacitance.
    :param Cv: Vertical tunneling junctions capacitance.
    :param Rh: Horizontal tunneling junctions resistance.
    :param Rv: Vertical tunneling junctions resistance.
    :param basePath: directory for saving the data
    """
    baseName = basePath + "_temp_"
    np.save(baseName + "VG", VG)
    np.save(baseName + "Q0", Q0)
    np.save(baseName + "n0", n0)
    np.save(baseName + "CG", CG)
    np.save(baseName + "RG", RG)
    np.save(baseName + "Ch", Ch)
    np.save(baseName + "Cv", Cv)
    np.save(baseName + "Rh", Rh)
    np.save(baseName + "Rv", Rv)
    return True


def loadRandomParams(basePath):
    """
    Loading array parameters for resuming a failed run.
    :param basePath: Directory in which data was stored.
    :return: Loaded parameters.
    """
    baseName = basePath + "_temp_"
    if not os.path.isfile(baseName + "VG.npy"):
        return None
    VG0 = np.load(baseName + "VG.npy")
    Q0 = np.load(baseName + "Q0.npy")
    n0 = np.load(baseName + "n0.npy")
    CG = np.load(baseName + "CG.npy")
    RG = np.load(baseName + "RG.npy")
    Ch = np.load(baseName + "Ch.npy")
    Cv = np.load(baseName + "Cv.npy")
    Rh = np.load(baseName + "Rh.npy")
    Rv = np.load(baseName + "Rv.npy")
    return VG0, Q0, n0, CG, RG, Ch, Cv, Rh, Rv


def removeRandomParams(basePath):
    """
    Delete saved data in the end of a successful run.
    :param basePath: Directory in which temporary data was stored.
    """
    baseName = basePath + "_temp_"
    os.remove(baseName + "VG.npy")
    os.remove(baseName + "Q0.npy")
    os.remove(baseName + "n0.npy")
    os.remove(baseName + "CG.npy")
    os.remove(baseName + "RG.npy")
    os.remove(baseName + "Ch.npy")
    os.remove(baseName + "Cv.npy")
    os.remove(baseName + "Rh.npy")
    os.remove(baseName + "Rv.npy")
    return True


def runFullSimulation(VL0, VR0, VU0, VD0, vSym, VG0, Q0, n0, CG, RG, Ch, Cv, Rh, Rv, rows, columns,
                      Vmax, Vstep, leftElectrode, rightElectrode, upElectrode, downElectrode,
                      temperature=0, temperature_gradient=0, repeats=1, savePath=".", fileName="", fullOutput=False,
                      printState=False, useGraph=False, fastRelaxation=False,
                      currentMap=False, dbg=False, plotCurrentMaps=False, plotBinaryCurrentMaps=False, resume=False,
                      superconducting=False, gap=0, leaping=False, modifyR=False, plotVoltages=False,
                      frame_norm=False, double_time=False, double_loop=False, calcIT=False):
    """
    Running many different realizations for the same simulation and averaging results.
    :param VL0: Initial voltage on the left electrode.
    :param VR0: Initial voltage on the right electrode.
    :param VU0: Initial voltage on the top electrode.
    :param VD0: Initial voltage on the bottom electrode.
    :param vSym: If true, voltage would be raised symmetrically on both sides, otherwise only VL changes.
    :param Q0: Initial charges on gate capacitors (rowsXcolumns array)
    :param n0: Initial occupations of islands (rowsXcolumns array of integers)
    :param CG: Gate capacitance (rowsXcolumns array)
    :param RG: Gate resistance (rowsXcolumns array)
    :param Ch: Horizontal tunneling junctions capacitance (rowsXcolumns+1 array)
    :param Cv: Vertical tunneling junctions capacitance (rows+1Xcolumns array)
    :param Rh: Horizontal tunneling junctions resistance (rowsXcolumns+1 array)
    :param Rv: Vertical tunneling junctions resistance (rows+1Xcolumns array)
    :param rows: Number of rows in the array.
    :param columns: Number of columns in the array.
    :param Vmax: Maximal external voltage.
    :param Vstep: External voltage step size.
    :param left/right/up/downElectrode: location of electrodes
          ((rows,) size array with 1 where electrode is connected and 0 otherwise)
    :param temperature: Temperature on the left side of the array (double).
    :param temperature_gradient: Constant temperature gradient from left to right (double).
    :param repeats: How many realizations of the simulation to run (integer).
    :param savePath: Directory for output and temporary files (string).
    :param fileName: Output file names prefix (string).
    :param fullOutput: If true, occupation and gate charges would be also calculated in the simulation (boolean).
    :param printState: The state of the system would be printed after each tunneling. for debug (boolean)
    :param useGraph: If true, graph method will be used instead of Gillespie's KMC algorithm (use only for very small
     arrays)
    :param fastRelaxation: If true, fast relaxation limit would be assumed (boolean).
    :param currentMap: If true, a map of local currents would be saved (boolean, not implemented for graph method).
    :param dbg: If true, will run in debug mode, no parallel run (boolean).
    :param plotCurrentMaps: If true, the previously calculated current map would be plotted (boolean,
    only after a successful simulation has end)
    :param plotBinaryCurrentMaps:  If true, the previously calculated current map would be plotted. This map would be
     binary, 1 for "there is current" and 0 for "no current"(boolean, only after a successful simulation has end)
    :param resume: If true, would try to resume simulation from a saved state. Use for resuming interrupted simulation
     (boolean).
    :param superconducting: If true, simulation would run using a superconducting array, including Cooper-pairs and
    quasi-particles tunnelings (boolean).
    :param gap: Superconducting gap, used only for superconducting arrays (double).
    :param leaping: If true, tau-leaping approximation will be used (boolean).
    :param modifyR: If true, finite electron density of states would be assumed, altering tunneling resistance
     according to occupation (boolean).
    :param plotVoltages: If true, the steady-state voltages after each step would be plotted (boolean).
    :param frame_norm: Used for current map plotting. If true, the map for each voltage would be separately normalized
     (instead of using the same normalization for all. boolean).
    :param double_time: If true, simulation step would be twice as long (boolean).
    :param double_loop: If true, external voltage cycle would run twice (boolean).
    :param calcIT: If true, current would be calculated as a function of temperature instead of voltage. For this case
    Vmax would be used as maximum temperature and Vstep as temperature step.
    :return:
    """
    basePath = os.path.join(savePath, fileName)
    if useGraph:
        dbg = True
        repeats = 1
        if currentMap:
            currentMap = False
            print("Warning, current map calculation is not implemented for graph method. Running without"
                  " this calculation.")
    if plotCurrentMaps or plotBinaryCurrentMaps:
        if useGraph:
            print("Current map calculation is not implemented for graph method.")
            exit(0)
        print("Plotting Current Maps")
        avgImaps = np.load(basePath + "_Imap.npy")
        if calcIT:
            V = np.load(basePath + "_T.npy")
        else:
            V = np.load(basePath + "_V.npy")
        n=None
        if fullOutput:
            n = np.load(basePath + "_n.npy")
        saveCurrentMaps(avgImaps, V, basePath + "_Imap", full=fullOutput,
                        n=n, binary=plotBinaryCurrentMaps, frame_norm=frame_norm, calcIT=calcIT)
        exit(0)
    load = False
    if resume:
        print("Loading array parameters")
        loaded = loadRandomParams(basePath)
        if loaded is not None:
            VG0, Q0, n0, CG, RG, Ch, Cv, Rh, Rv = loaded
            load = True
        else:
            print("Couldn't load parameters, generating new parameters")
    if not load:
        print("Saving array parameters")
        saveRandomParams(np.array(VG0),
                         np.array(Q0), np.array(n0), np.array(CG),
                         np.array(RG), np.array(Ch), np.array(Cv),
                         np.array(Rh), np.array(Rv), basePath)

    if superconducting:
        prototypeArray = JJArray(rows, columns, VL0, VR0, VU0, VD0, np.array(VG0), np.array(Q0), np.array(n0),
                                 np.array(CG), np.array(RG), np.array(Ch), np.array(Cv), np.array(Rh), np.array(Rv),
                                 temperature, temperature_gradient, gap, leftElectrode, rightElectrode, upElectrode,
                                 downElectrode, fastRelaxation=fastRelaxation, tauLeaping=leaping, modifyR=modifyR)
        print("Superconducting prototype array was created")
    else:
        prototypeArray = DotArray(rows, columns, VL0, VR0, VU0, VD0, np.array(VG0), np.array(Q0), np.array(n0),
                                  np.array(CG), np.array(RG), np.array(Ch), np.array(Cv), np.array(Rh), np.array(Rv),
                                  temperature, temperature_gradient, leftElectrode, rightElectrode, upElectrode,
                                  downElectrode, fastRelaxation=fastRelaxation, tauLeaping=leaping, modifyR=modifyR)
        print("Normal prototype array was created")

    if plotVoltages:
        if useGraph:
            print("Voltages plotting is not implemented for graph method")
            exit(0)
        print("Plotting voltages")
        simulator = Simulator(0, VL0, VR0, VU0, VD0, Q0, n0, prototypeArray)
        simulator.plotAverageVoltages()
        exit(0)
    Is = []
    IsErr = []
    vertIs = []
    vertIsErr = []
    ns = []
    Qs = []
    nsErr = []
    QsErr = []
    Imaps = []
    if not dbg:
        print("Starting parallel run")
        pool = Pool(processes=repeats)
        results = []
        for repeat in range(repeats):

            res = pool.apply_async(runSingleSimulation,
                                    (repeat, VL0, VR0, VU0, VD0, vSym, Q0, n0, Vmax, Vstep, prototypeArray, fullOutput,
                                     printState, useGraph, currentMap,basePath, resume, double_time,
                                     double_loop, calcIT))
            results.append(res)
        for res in results:
            result = res.get()
            I = result[0]
            IErr = result[1]
            vertI = result[2]
            vertIErr = result[3]
            if fullOutput:
                n = result[5]
                Q = result[6]
                nErr = result[7]
                QErr = result[8]
                ns.append(n)
                Qs.append(Q)
                nsErr.append(nErr)
                QsErr.append(QErr)
            if currentMap:
                Imaps.append(result[-2])
            Is.append(I)
            IsErr.append(IErr)
            vertIs.append(vertI)
            vertIsErr.append(vertIErr)
        V = result[4]
        params = result[-1]

    else: #dbg
        print("Starting serial run")
        for repeat in range(repeats):
            result = runSingleSimulation(repeat, VL0, VR0, VU0, VD0, vSym, Q0, n0, Vmax, Vstep, prototypeArray,
                                         fullOutput, printState, useGraph, currentMap, basePath, resume, double_time,
                                         double_loop, calcIT)
            I = result[0]
            IErr = result[1]
            vertI = result[2]
            vertIErr = result[3]
            if fullOutput:
                n = result[5]
                Q = result[6]
                nErr = result[7]
                QErr = result[8]
                ns.append(n)
                Qs.append(Q)
                nsErr.append(nErr)
                QsErr.append(QErr)
            if currentMap:
                Imaps.append(result[-2])
            Is.append(I)
            IsErr.append(IErr)
            vertIs.append(vertI)
            vertIsErr.append(vertIErr)
        V = result[4]
        params = result[-1]
        # dbg
    print("Saving results")
    avgI = np.mean(np.array(Is), axis=0)
    avgVertI = np.mean(np.array(vertIs), axis=0)
    if repeats < 10:
        avgIErr = np.sqrt(np.sum(np.array(IsErr)**2, axis=0)/len(IsErr))
        avgVertIErr = np.sqrt(np.sum(np.array(vertIsErr)**2, axis=0)/len(vertIsErr))
    else:
        avgIErr = np.std(np.array(Is), axis=0)/np.sqrt(len(Is))
        avgVertIErr = np.std(np.array(vertIs), axis=0) / np.sqrt(len(vertIs))
    if fullOutput:
        avgN = np.mean(np.array(ns), axis=0)
        avgQ = np.mean(np.array(Qs), axis=0)
        avgNErr = np.sqrt(np.sum(np.array(nsErr) ** 2, axis=0)) / len(nsErr)
        avgQErr = np.sqrt(np.sum(np.array(QsErr) ** 2, axis=0)) / len(QsErr)
        np.save(basePath + "_n", avgN)
        np.save(basePath + "_Q", avgQ)
        np.save(basePath + "_nErr", avgNErr)
        np.save(basePath + "_QErr", avgQErr)
        np.save(basePath + "_full_I", np.array(Is))
        np.save(basePath + "_full_IErr", np.array(IsErr))
    fig = plt.figure()
    IplusErr = avgI + avgIErr
    IminusErr = avgI - avgIErr
    vertIplusErr = avgVertI + avgVertIErr
    vertIminusErr = avgVertI - avgVertIErr
    if calcIT:
        plt.plot(V, avgI, 'g-', V, IplusErr, 'g--',
                 V, IminusErr, 'g--')
        plt.xlabel('Temperature')
        plt.ylabel('Current from left to right')
        plt.savefig(basePath + "_IT.png")
        np.save(basePath + "_T", V)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(V, avgVertI, 'g-', V, vertIplusErr, 'g--',
                 V, vertIminusErr, 'g--')
        plt.xlabel('Temperature')
        plt.ylabel('Current from up to down')
        plt.savefig(basePath + "_vert_IT.png")
    elif double_loop:
        plt.plot(V[:V.size // 4], avgI[:V.size // 4], 'b-',
                 V[:V.size // 4], IplusErr[:V.size // 4], 'b--',
                 V[:V.size // 4], IminusErr[:V.size // 4], 'b--',
                 V[V.size // 4:V.size//2], avgI[V.size // 4:V.size//2], 'r-',
                 V[V.size // 4:V.size//2], IplusErr[V.size // 4:V.size//2], 'r--',
                 V[V.size // 4:V.size//2], IminusErr[V.size // 4:V.size//2], 'r--',
                 V[V.size // 2: 3*V.size//4], avgI[V.size // 2: 3*V.size//4], 'c-',
                 V[V.size // 2: 3*V.size//4], IplusErr[V.size // 2: 3*V.size//4], 'c--',
                 V[V.size // 2: 3*V.size//4], IminusErr[V.size // 2: 3*V.size//4], 'c--',
                 V[3 * V.size // 4:], avgI[3 * V.size // 4:], 'm-',
                 V[3 * V.size // 4:], IplusErr[3 * V.size // 4:], 'm--',
                 V[3 * V.size // 4:], IminusErr[3 * V.size // 4:], 'm--',
                 )
        plt.xlabel('Voltage')
        plt.ylabel('Current from left to right')
        plt.savefig(basePath + "_IV.png")
        np.save(basePath + "_V", V)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(V[:V.size // 4], avgVertI[:V.size // 4], 'b-',
                 V[:V.size // 4], vertIplusErr[:V.size // 4], 'b--',
                 V[:V.size // 4], vertIminusErr[:V.size // 4], 'b--',
                 V[V.size // 4:V.size//2], avgVertI[V.size // 4:V.size//2], 'r-',
                 V[V.size // 4:V.size//2], vertIplusErr[V.size // 4:V.size//2], 'r--',
                 V[V.size // 4:V.size//2], vertIminusErr[V.size // 4:V.size//2], 'r--',
                 V[V.size // 2: 3*V.size//4], avgVertI[V.size // 2: 3*V.size//4], 'c-',
                 V[V.size // 2: 3*V.size//4], vertIplusErr[V.size // 2: 3*V.size//4], 'c--',
                 V[V.size // 2: 3*V.size//4], vertIminusErr[V.size // 2: 3*V.size//4], 'c--',
                 V[3 * V.size // 4:], avgVertI[3 * V.size // 4:], 'm-',
                 V[3 * V.size // 4:], vertIplusErr[3 * V.size // 4:], 'm--',
                 V[3 * V.size // 4:], vertIminusErr[3 * V.size // 4:], 'm--',
                 )
        plt.xlabel('Voltage')
        plt.ylabel('Current from up to down')
        plt.savefig(basePath + "_vert_IV.png")	
    else:
        plt.plot(V[:V.size // 2], avgI[:V.size // 2], 'b-', V[:V.size // 2], IplusErr[:V.size // 2], 'b--',
                 V[:V.size // 2], IminusErr[:V.size // 2], 'b--',
                 V[V.size // 2:], avgI[V.size // 2:], 'r-', V[V.size // 2:], IplusErr[V.size // 2:], 'r--',
                 V[V.size // 2:], IminusErr[V.size // 2:], 'r--')
        plt.xlabel('Voltage')
        plt.ylabel('Current from left to right')
        plt.savefig(basePath + "_IV.png")
        np.save(basePath + "_V", V)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(V[:V.size // 2], avgVertI[:V.size // 2], 'b-', V[:V.size // 2],
                 IplusErr[:V.size // 2], 'b--',
                 V[:V.size // 2], IminusErr[:V.size // 2], 'b--',
                 V[V.size // 2:], avgVertI[V.size // 2:], 'r-', V[V.size // 2:],
                 IplusErr[V.size // 2:], 'r--',
                 V[V.size // 2:], IminusErr[V.size // 2:], 'r--')
        plt.xlabel('Voltage')
        plt.ylabel('Current from up to down')
        plt.savefig(basePath + "_vert_IV.png")
    plt.close(fig)
    fig = plt.figure()
    plt.close(fig)
    np.save(basePath + "_I", avgI)
    np.save(basePath + "_IErr", avgIErr)
    np.save(basePath + "_vertI", avgVertI)
    np.save(basePath + "_vertIErr", avgVertIErr)

    if currentMap:
        avgImaps = np.mean(np.array(Imaps), axis=0)
        np.save(basePath + "_Imap", avgImaps)
    for index in range(repeats):
        removeState(index, fullOutput=fullOutput, basePath=basePath, currentMap=currentMap, graph=use_graph)
    removeRandomParams(basePath)
    return params


def removeState(index, fullOutput=False, basePath='', currentMap=False, graph=False):
    """
    Deleting temporarily saved states after a successful run.
    :param index: Simulation instance index.
    :param fullOutput: Did it run with full output?
    :param basePath: directory where files were saved.
    :param currentMap: Did current map was calculated?
    :param graph: Did graph method was used?
    """
    baseName = basePath + "_temp_" + str(index)
    os.remove(baseName + "_I.npy")
    if not graph:
        os.remove(baseName + "_IErr.npy")
        os.remove(baseName + "_vertI.npy")
        os.remove(baseName + "_vertIErr.npy")
    os.remove(baseName + "_n.npy")
    os.remove(baseName + "_Q.npy")
    if fullOutput:
        os.remove(baseName + "_ns.npy")
        os.remove(baseName + "_Qs.npy")
        if not graph:
            os.remove(baseName + "_nsErr.npy")
            os.remove(baseName + "_QsErr.npy")
    if currentMap:
        os.remove(baseName + "_current_map.npy")
    return True


def saveCurrentMaps(Imaps, V, path, full=False, n=None, binary=False, frame_norm=False, calcIT=False):
    """
    Saving previously calculated current maps as a cartoon.
    :param Imaps: current maps.
    :param V: external voltage vector.
    :param path: directory for output.
    :param full: if true, average occupations on each island would also be added to the cartoon.
    :param n: Average occupations.
    :param binary: if true, maps would be binary (1 for "there is current" or 0 for "no current").
    :param frame_norm: if true, each frame would be normalized separately, instead of using the
     same normalization for all.
    :param calcIT: true for the case where I-T was calculated instead of I-V.
    """
    if binary:
        Imaps[Imaps > 0] = 1
        Imaps[Imaps < 0] = -1
        path += "_binary"
    if frame_norm:
        path += "_frame_norm"
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=24, bitrate=1800)
    fig, ax = plt.subplots()
    Imax = 1 if frame_norm else max(np.max(Imaps), -np.min(Imaps))
    Imin = -1 if frame_norm else min(np.min(Imaps), -np.max(Imaps))
    M, N = Imaps[0].shape
    im = ax.imshow(np.zeros(((M // 2) * 3 + 2, N * 3)), vmin=Imin,
                    vmax=Imax, animated=True, cmap='PuOr', aspect='equal')
    text = ax.text(1, 1, 'T = 0') if calcIT else ax.text(1, 1, 'Vext = 0')
    cb1 = plt.colorbar(im, shrink=0.25)
    cb1.set_label('Current')
    if full:
        nmax = 1 if frame_norm else np.max(n)
        nmin = -1 if frame_norm else np.min(n)
        im2 = ax.imshow(np.zeros(((M // 2) * 3 + 2, N * 3)),
                        vmin=nmin, vmax=nmax, animated=True, cmap='RdBu',
                        aspect='equal')
        cb2 = plt.colorbar(im2, shrink=0.25)
        cb2.set_label('Occupation')
        frames = [(Imaps[i], V[i], n[i]) for i in range(len(V))]
        im_ani = animation.FuncAnimation(fig,
                                         plotCurrentMaps(im, text, M, N, full=True, im2=im2, frame_norm=frame_norm,
                                                         calcIT=calcIT),
                                         frames=frames, interval=100,
                                         repeat_delay=1000,
                                         blit=True)
    else:
        frames = [(Imaps[i], V[i]) for i in range(len(V))]
        im_ani = animation.FuncAnimation(fig, plotCurrentMaps(im, text, M, N, frame_norm=frame_norm, calcIT=calcIT),
                                        frames=frames, interval=100,
                                         repeat_delay=1000,
                                        blit=True)
    im_ani.save(path + '.mp4', writer=writer)
    plt.close(fig)

def plotCurrentMaps(im, text, M, N, full=False, im2=None, frame_norm=False, calcIT=False):
    """
    Helping method for plotting current maps.
    :param im: matplotlib image in which the maps would be plotted.
    :param text: Text to add to the image.
    :param M: Number of rows in the image.
    :param N: Number of columns in the image.
    :param full: if true, average occupations on each island would also be added to the cartoon.
    :param im2: Second image for plotting island occupations.
    :param frame_norm: if true, each frame would be normalized separately, instead of using the
      same normalization for all.
    :param calcIT: true for the case where I-T was calculated instead of I-V.
    """
    J = np.zeros(((M//2)*3+2, N*3))
    horzRows = np.arange(2, (M//2)*3+2, 3)
    horzCols = np.repeat(np.arange(0, 3*N, 3), 2)
    horzCols[1::2] += 1
    vertRows = np.repeat(np.arange(0, (M//2)*3+1, 3), 2)
    vertRows[1::2] += 1
    vertCols = np.arange(2, 3*N, 3)
    Jmask = np.ones(J.shape)
    Jmask[np.ix_(horzRows, horzCols)] = 0
    Jmask[np.ix_(vertRows, vertCols)] = 0
    if full:
        dot_rows = horzRows
        dot_cols = vertCols[:-1]
        dots_im = J.copy()
        dots_im_mask = np.ones(J.shape)
        dots_im_mask[np.ix_(dot_rows, dot_cols)] = 0

    def updateCurrent(result):
        if full:
            I, Vext, n = result
        else:
            I, Vext = result
        if I is None:
            return im
        if frame_norm:
            normalization = np.max(np.abs(I))
            if normalization > 0:
                I = I/normalization
            if full:
                normalization = np.max(np.abs(n))
                if normalization > 0:
                    n = n/normalization
        J[np.ix_(horzRows, horzCols)] = np.repeat(I[1:M:2, :], 2, axis=1)
        J[np.ix_(vertRows, vertCols)] = np.repeat(I[0:M:2, :], 2, axis=0)
        J_masked = np.ma.masked_array(J, Jmask)
        im.set_array(J_masked)
        if calcIT:
            text.set_text('T = %.3f' % float(Vext))
        else:
            text.set_text('Vext = %.3f' % float(Vext))
        if full:
            dots_im[np.ix_(dot_rows, dot_cols)] = n
            dots_im_masked = np.ma.masked_array(dots_im, dots_im_mask)
            im2.set_array(dots_im_masked)
            return im, text, im2
        else:
            return im, text
    return updateCurrent


def getOptions():
    """
    Collecting command line parameters.
    """
    parser = OptionParser(usage="usage: %prog [options]")
    # Normal parameters
    parser.add_option("-T", "--temperature", dest="T",
                      help="Environment temperature (in units of planckConstant/timeUnits) [default: %default]",
                      default=0, type=float)
    parser.add_option("--temperature-gradient", dest="temperature_gradient",
                      help="Temperature gradient (in units of planckConstant/(timeUnits*Lattice constant))"
                           " [default: %default]",
                      default=0, type=float)
    parser.add_option("--gap", dest="gap",
                      help="superconducting gap (in units of planckConstant/timeUnits)[default: %default]",
                      default=0, type=float)
    parser.add_option("-M", "--height", dest="M", help="number of lines in "
                      "the array [default: %default]", default=1, type=int)
    parser.add_option("-N", "--width", dest="N", help="number of columns in "
                      "the array [default: %default]", default=1, type=int)
    parser.add_option("--vr", dest="VR", help="right electrode voltage (in units of"
                                              " planckConstant/electronCharge*timeUnits) [default: %default]",
                      default=0, type=float)
    parser.add_option("--vu", dest="VU",
                      help="upper electrode voltage (in units of"
                           " planckConstant/electronCharge*timeUnits) [default: %default]",
                      default=0, type=float)
    parser.add_option("--vd", dest="VD",
                      help="lower electrode voltage (in units of"
                           " planckConstant/electronCharge*timeUnits) [default: %default]",
                      default=0, type=float)
    parser.add_option("--right-electrode", dest="rightElectrode",
                      help="Location of right electrode in form of a binary array, i.e. if the array height is 7 and"
                           " the electrode is connected in the second and fifth rows then [0,1,0,0,1,0,0] [default:"
                           " connected to all rows]",
                      default="")
    parser.add_option("--left-electrode", dest="leftElectrode",
                      help="Location of left electrode in form of a binary array, i.e. if the array height is 7 and"
                           " the electrode is connected in the second and fifth rows then [0,1,0,0,1,0,0] [default:"
                           " connected to all rows]",
                      default="")
    parser.add_option("--up-electrode", dest="upElectrode",
                      help="Location of upper electrode in form of a binary array, i.e. if the array width is 5 and"
                           " the electrode is connected in the second and fifth rows then [0,1,0,0,1] [default: "
                           "connected to all columns]",
                      default="")
    parser.add_option("--down-electrode", dest="downElectrode",
                      help="Location of lower electrode in form of a binary array, i.e. if the array width is 5 and"
                           " the electrode is connected in the second and fifth rows then [0,1,0,0,1] [default: "
                           "connected to all columns]",
                      default="")
    parser.add_option("--vmin", dest="Vmin", help="minimum external voltage  (in units of"
                                              " planckConstant/electronCharge*timeUnits)"
                      " [default: %default]", default=0, type=float)
    parser.add_option("--vmax", dest="Vmax", help="maximum external voltage  (in units of"
                                              " planckConstant/electronCharge*timeUnits)"
                      " [default: %default]", default=10, type=float)
    parser.add_option("--vstep", dest="vStep", help="size of voltage step  (in units of"
                                              " planckConstant/electronCharge*timeUnits)[default: %default]",
                      default=1, type=float)
    parser.add_option("--symmetric-v", dest="vSym", help="Voltage raises symmetric on VR and VL["
                                                    "default: %default]", default=False, action='store_true')
    parser.add_option("--repeats", dest="repeats",
                      help="how many times to run calculation for averaging"
                      " [default: %default]", default=1, type=int)
    parser.add_option("--file-name", dest="fileName", help="optional "
                      "output files name", default='')
    parser.add_option("--distribution", dest="dist", help="probability distribution to use [Default:%default]",
                      default='uniform')
    parser.add_option("--full", dest="fullOutput", help="if true the "
                      "results n and Q will be also saved [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--graph", dest="use_graph", help="if true a simulation using graph solution for master equation"
                                                        "will be used [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--current-map", dest="current_map", help="if true frames for clip of current distribution during"
                                                                " simulation will be created and saved "
                                                                "[Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--plot-current-map", dest="plot_current_map", help="if true clip of current distribution will"
                                                                          " be plotted using former saved frames (from"
                                                                          " a former run with same file name and"
                                                                          " location and the flag --current-map"
                                                                          " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--plot-binary-current-map", dest="plot_binary_current_map",
                      help="if true a binary clip of current distribution will"
                           " be plotted using former saved frames (from"
                           " a former run with same file name and"
                           " location and the flag --current-map"
                           " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--frame-norm", dest="frame_norm",
                      help="if true the clip of current distribution will"
                           " be normalized per frame"
                           " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--dbg", dest="dbg", help="Avoids parallel running for debugging [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--resume", dest="resume", help="Resume failed run from last checkpoint [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--superconducting", dest="sc", help="use superconducting array [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--tau-leaping", dest="leaping", help="use tau leaping approximation [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--variable-ef", dest="modifyR", help="if true Fermi energy level for each island will"
                                                            " be changed according to constant density of states"
                                                            " assumption, else it will be assumed constant"
                                                            " (infinite density of states) [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--double-time", dest="double_time",
                      help="if true each simulation step will run twice as long as the default time"
                           " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--double-loop", dest="double_loop",
                      help="if true the voltage would be raised and lowered twice"
                           " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("--calc-it", dest="calcIT",
                      help="Instead of calculating IV curve calculates current as a function of the temperature,"
                           " in this case Vmax, Vstep would be used as Tmax, Tstep instead"
                           " [Default:%default]",
                      default=False, action='store_true')
    parser.add_option("-o", "--output-folder", dest="output_folder",
                      help="Output folder [default: current folder]",
                      default='.')
    parser.add_option("-i", "--load-from-file", dest="params_path",
                      help="If a parameters file is given all the array parameters would be loaded from that"
                           " file. The file should be in the same format as the resulted parameter file for a run."
                           " Ignores all other array related parameters, so the array would be exactly as specified in"
                           " the given file. [default: '']",
                      default='')

    #  Disorder Parameters  #
    parser.add_option("--vg-avg", dest="VG_avg", help="Gate voltage average  (in units of"
                                                      " planckConstant/electronCharge*timeUnits) [default: %default]",
                      default=1, type=float)
    parser.add_option("--vg-std", dest="VG_std", help="Gate voltage std  (in units of"
                                              " planckConstant/electronCharge*timeUnits) [default: %default]",
                      default=0, type=float)
    parser.add_option("--c-avg", dest="C_avg", help="capacitance of junctions average (in units of"
                                                    " timeUnits*electronCharge^2/planckConstant) [default: %default]",
                      default=1, type=float)
    parser.add_option("--c-std", dest="C_std", help="capacitance of junctions std (in units of"
                                                    " timeUnits*electronCharge^2/planckConstant) [default: %default]",
                      default=0, type=float)
    parser.add_option("--cg-avg", dest="CG_avg", help="Gate Capacitors capacitance average (in units of"
                                                    " timeUnits*electronCharge^2/planckConstant) [default: %default]",
                      default=1, type=float)
    parser.add_option("--cg-std", dest="CG_std", help="Gate Capacitors capacitance std (in units of"
                                                    " timeUnits*electronCharge^2/planckConstant) [default: %default]",
                      default=0, type=float)
    parser.add_option("--r-avg", dest="R_avg", help="junctions resistance average (in units of"
                                                    " planckConstant/electronCharge^2) [default: %default]",
                      default=1, type=float)
    parser.add_option("--r-std", dest="R_std", help="junctions resistance std (in units of"
                                                    " planckConstant/electronCharge^2) [default: %default]",
                      default=0, type=float)
    parser.add_option("--custom-rh", dest="custom_rh", help="list of r horizontal values ordered as numpy array."
                                                            " Overrides random r parameters [default: %default]",
                      default="")
    parser.add_option("--custom-rv", dest="custom_rv", help="list of r vertical values ordered as numpy array."
                                                            " Overrides random r parameters [default: %default]",
                      default="")
    parser.add_option("--custom-ch", dest="custom_ch", help="list of c horizontal values ordered as numpy array."
                                                            " Overrides random c parameters [default: %default]",
                      default="")
    parser.add_option("--custom-cv", dest="custom_cv", help="list of c vertical values ordered as numpy array."
                                                            " Overrides random c parameters [default: %default]",
                      default="")
    parser.add_option("--rg-avg", dest="RG_avg", help="Gate Resistors resistance average (in units of"
                                                      " planckConstant/electronCharge^2) [default: %default]",
                      default=1, type=float)
    parser.add_option("--rg-std", dest="RG_std", help="Gate Resistors resistance std (in units of"
                                                      " planckConstant/electronCharge^2) [default: %default]",
                      default=0, type=float)
    parser.add_option("--n-avg", dest="n0_avg", help="initial number of "
                      "electrons on each dot average [default:%default]",
                      default=0, type=float)
    parser.add_option("--n-std", dest="n0_std", help="initial number of "
                      "electrons on each dot std [default:%default]",
                      default=0, type=float)
    parser.add_option("--q-avg", dest="Q0_avg", help="initial charge on gate capacitors average (in units of"
                                                     " electronCharge) [default:%default]",
                      default=0, type=float)
    parser.add_option("--q-std", dest="Q0_std", help="initial charge on gate capacitors std (in units of"
                                                     " electronCharge) [default:%default]",
                      default=0, type=float)
    return parser.parse_args()


def saveParameters(path, fileName, options, array_params):
    """
    Saving a text file with all simulation's parameters.
    :param path: directory in which the file would be saved.
    :param fileName: name of the file.
    :param options: Command line options.
    :param array_params: Array parameters.
    :return:
    """
    optionsDict = options.__dict__
    with open(os.path.join(path, 'runningParameters_' + fileName +
            ".txt"), mode='w') as f:
        f.write("-------Running parameters--------\n")
        for key in vars(options):
            f.write(key + " = " + str(optionsDict[key]) + "\n")
        f.write(array_params)


def create_random_array(M, N, avg, std, dist, only_positive=False):
    if std == 0:
        res = avg*np.ones((M, N))
    elif dist == 'uniform' or (dist == 'exp' and not only_positive):
        res = np.random.uniform(low=avg-std, high=avg+std, size=(M, N))
    elif dist == 'two_points':
        r = np.random.rand(M, N)
        r[r > 0.5] = avg + std
        r[r <= 0.5] = avg - std
        res = r
    elif dist == 'normal':
        res = np.random.normal(loc=avg, scale=std, size=(M, N))
    elif dist == 'gamma':
        if avg == 0:
            shape = 1
            scale = 2
        else:
            shape = (avg/std)**2
            scale = std**2 / avg
        res = 0.1 + np.random.gamma(shape=shape, scale=scale, size=(M, N))
    elif dist == 'exp':
        r = np.random.uniform(low=np.log2(max(avg - std, 0.01)), high=np.log2(max(avg + std, 0.01)), size=(M, N))
        res = 2**r
    if only_positive and (res <= 0).any():
        print("Warning, changing to positive distribution")
        res = res + 0.1 - np.min(res.flatten())
    return res


def load_params_from_file(file_path):
    """
    Loading the simulation parameters from previous simulation.
    :param file_path: Path to the existing parameters file.
    :return:
    """
    runningParams = dict()
    arrayParams = dict()
    with open(file_path, 'r') as f:
        start = False
        for line in f:
            if ' = ' in line:
                splitted = line.split(' = ')
                key = splitted[0]
                try:
                    value = literal_eval(splitted[1].rstrip('\n'))
                except Exception:
                    value = splitted[1].rstrip('\n')
                runningParams[key] = value
            elif ': ' in line:
                if start:
                    key = splitted[0]
                    try:
                        splitted[1] = splitted[1].rstrip('\n')
                        splitted[1] = regex.sub('\[\s+', '[', splitted[1])
                        splitted[1] = regex.sub('\s+\]', ']', splitted[1])
                        splitted[1] = regex.sub('\s+', ',', splitted[1])
                        value = literal_eval(splitted[1])
                    except Exception:
                        value = splitted[1].rstrip('\n')
                    arrayParams[key] = value
                start = True
                splitted = line.split(': ')
            elif start:
                splitted[1] = splitted[1].replace('\n', ' ') + line
        key = splitted[0]
        try:
            splitted[1] = splitted[1].rstrip('\n')
            splitted[1] = regex.sub('\[\s+', '[', splitted[1])
            splitted[1] = regex.sub('\s+\]', ']', splitted[1])
            splitted[1] = regex.sub('\s+', ',', splitted[1])
            value = literal_eval(splitted[1])
        except Exception as e:
            value = splitted[1].rstrip('\n')
        arrayParams[key] = value
    return runningParams, arrayParams


if __name__ == "__main__":
    # Initializing Running Parameters
    options, args = getOptions()
    params_file = options.params_path
    if params_file:
        print("Loading parameters from file " + params_file)
        runningParams, arrayParams = load_params_from_file(params_file)
        rows = runningParams['M']
        columns = runningParams['N']
    else:
        rows = options.M
        columns = options.N
    vSym = options.vSym
    if vSym:
        VR0 = options.VR - options.Vmin/2
        VL0 = options.VR + options.Vmin/2
    else:
        VR0 = options.VR
        VL0 = VR0 + options.Vmin
    VU0 = options.VU
    VD0 = options.VD
    dist = options.dist
    T = options.T
    temperature_gradient = options.temperature_gradient
    gap = options.gap
    sc = options.sc
    leaping = options.leaping
    Q0 = create_random_array(rows, columns, options.Q0_avg, options.Q0_std, dist,
                             False)
    n0 = create_random_array(rows, columns, options.n0_avg, options.n0_std, dist, False)
    Vmax = options.Vmax
    Vstep = options.vStep

    repeats = options.repeats
    savePath = options.output_folder
    fileName = options.fileName
    fullOutput = options.fullOutput
    use_graph = options.use_graph
    fast_relaxation = options.RG_avg * options.CG_avg < options.C_avg * options.R_avg
    current_map = options.current_map
    dbg = options.dbg
    resume = options.resume
    plot_current_map = options.plot_current_map
    plot_binary_current_map = options.plot_binary_current_map
    frame_norm = options.frame_norm
    modifyR = options.modifyR
    if params_file:
        VG = arrayParams['VG']
        CG = arrayParams['CG']
        RG = arrayParams['RG']
        Ch = arrayParams['Ch']
        Cv = arrayParams['Cv']
        Rh = arrayParams['Rh']
        Rv = arrayParams['Rv']
    else:
        VG = create_random_array(rows, columns, options.VG_avg, options.VG_std, dist,
                                 False)
        CG = create_random_array(rows, columns, options.CG_avg, options.CG_std, dist, True)
        RG = create_random_array(rows, columns, options.RG_avg, options.RG_std, dist, True)
        if options.custom_ch.replace('\"', '') and options.custom_cv.replace('\"', ''):
            Ch = literal_eval(options.custom_ch.replace('\"', ''))
            Cv = literal_eval(options.custom_cv.replace('\"', ''))
        else:
            Ch = create_random_array(rows, columns + 1, options.C_avg, options.C_std, dist,
                                     True)
            Cv = create_random_array(rows + 1, columns, options.C_avg, options.C_std, dist,
                                     True)
        if options.custom_rh.replace('\"', '') and options.custom_rv.replace('\"', ''):
            Rh = literal_eval(options.custom_rh.replace('\"', ''))
            Rv = literal_eval(options.custom_rv.replace('\"', ''))
        else:
            Rh = create_random_array(rows, columns + 1, options.R_avg, options.R_std, dist,
                                     True)
            Rv = create_random_array(rows + 1, columns, options.R_avg, options.R_std, dist,
                                     True)

    # Running Simulation
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    elif not os.path.isdir(savePath):
        print("the given path exists but is a file")
        exit(0)
    if dbg:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    leftElectrode = literal_eval(options.leftElectrode.replace('\"', '')) if options.leftElectrode else [1]*rows
    rightElectrode = literal_eval(options.rightElectrode.replace('\"', '')) if options.rightElectrode else [1]*rows
    upElectrode = literal_eval(options.upElectrode.replace('\"', '')) if options.upElectrode else [1]*columns
    downElectrode = literal_eval(options.downElectrode.replace('\"', '')) if options.downElectrode else [1]*columns
    array_params = runFullSimulation(VL0, VR0, VU0, VD0, vSym, VG, Q0, n0, CG, RG, Ch, Cv, Rh, Rv, rows,  columns,
                                     Vmax, Vstep, leftElectrode, rightElectrode, upElectrode, downElectrode,
                                     temperature=T, temperature_gradient=temperature_gradient, repeats=repeats,
                                     savePath=savePath, fileName=fileName,
                                     fullOutput=fullOutput, printState=False, useGraph=use_graph,
                                     fastRelaxation=fast_relaxation, currentMap=current_map,
                                     dbg=dbg, plotCurrentMaps=plot_current_map,
                                     plotBinaryCurrentMaps=plot_binary_current_map, resume=resume,
                                     superconducting=sc, gap=gap, leaping=leaping,
                                     modifyR=modifyR, frame_norm=frame_norm,
                                     double_time=options.double_time, double_loop=options.double_loop,
                                     calcIT=options.calcIT)
    saveParameters(savePath, fileName, options, array_params)

    if dbg:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    exit(0)
