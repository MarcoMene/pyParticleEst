import numpy
import pyparticleest.simulator
import matplotlib.pyplot as plt
from scipy.special import factorial

from numpy import exp,log, sqrt

from builtins import range
import pyparticleest.interfaces as interfaces
from trials_mm.utils_plots import plot_acf_pacf

import pyparticleest.paramest.paramest as param_est
import pyparticleest.paramest.interfaces as pestint

from scipy.stats import norm
from scipy.optimize import Bounds

import pyparticleest.utils.kalman as kalman


def generate_dataset(steps, x0, P0_x, phi, Q):
    x = numpy.zeros(steps)
    y = numpy.zeros(steps)

    x[0] = numpy.random.normal(x0, sqrt(P0_x))
    y[0] = numpy.random.poisson(lam=exp(x[0]))

    for k in range(1, steps):
        x[k] = x0 + phi * x[k - 1] + numpy.random.normal(0, sqrt(Q))
        y[k] = numpy.random.poisson(lam=exp(x[k]))

    return (x, y)


class PoissonModel(interfaces.ParticleFiltering, pestint.ParamEstInterface, pestint.ParamEstBaseNumeric):
    """
            ---> MM implementation of Poisson

            x_{k+1} = x0 + phi*x_k + e,   e ~ N(0, Q)  <-- log rate
            y_x_k = Poisson(   exp(x_k) )
            x(0) ~ Norm( x0, P0  )

            """

    def __init__(self, x_0, P0, phi, Q):

        # P0 e Q sono varianze, non standard deviation!!

        super().__init__(param_bounds=None)
        self.P0 = numpy.copy(P0)
        self.set_params( (x_0, phi, Q))

    @property
    def x_0(self):
        return self.params[0]
    @property
    def phi(self):
        return self.params[1]
    @property
    def Q(self):
        return self.params[2]

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, sqrt(self.Q), (N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        """ Predict: Calculate xt+1 given xt using the supplied noise"""
        particles *= self.phi
        particles += self.x_0 + noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement --> log p( y_t | x_t|t-1 ) """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            l = exp(particles[k].reshape(-1, 1))
            logyprob[k] = -l + y * numpy.log(l) - numpy.log(factorial(y))
        return logyprob  # should be a vector, that's why reshape is needed

    def create_initial_estimate(self, N):
        return numpy.random.normal(self.x_0, sqrt(self.P0), size=(N,)).reshape((-1, 1))

    def set_params(self, params):
        self.params = numpy.copy(params)

    def eval_logp_x0(self, particles, t):
        return kalman.lognormpdf_scalar(particles, numpy.asarray(self.P0).reshape(1, 1))  # I think we should return the sum

    def logp_xnext(self, particles, next_part, u, t):
        diff = next_part - particles * self.phi - self.x_0
        return kalman.lognormpdf_scalar(diff, numpy.asarray(self.Q).reshape(1, 1))  # Q should be a nd array [[]]


    def logp_xnext_full(self, part, past_trajs, pind,
                        future_trajs, find, ut, yt, tt, cur_ind):
        """
        copiata
        """

        # Default implemenation for markovian models, just look at the next state
        return self.logp_xnext(particles=part, next_part=future_trajs[0].pa.part[find],
                               u=ut[cur_ind], t=tt[cur_ind])



if __name__ == '__main__':
    # numpy.random.seed(666)

    steps = 200
    num = 50
    nums = 50

    x0 = 0.2
    P0_x = .5
    phi = 0.8
    Q = 1.

    (x, y) = generate_dataset(steps, x0, P0_x, phi, Q)
    plt.plot(exp(x), 'r-', label="exp(x)  [true]")
    plt.plot(y, 'bx', label="y")

    model = PoissonModel(x0, P0_x, phi, Q)  # it's better to enlarge variance for initial sampling
    sim = pyparticleest.simulator.Simulator(model, u=None, y=y)

    print(f"model parameters first iteration {model.params}")

    numpy.random.seed(666)
    sim.simulate(num, nums, smoother="mcmc", meas_first=True)   # meas_first is needed to treat the first y as measurement

    (vals, _) = sim.get_filtered_estimates()
    filter_mean = sim.get_filtered_mean()

    # svals = sim.get_smoothed_estimates()
    smoothed_mean = sim.get_smoothed_mean()
    #
    # # # Plot "smoothed" trajectories to illustrate that the particle filter
    # # # suffers from degeneracy when considering the full trajectories
    # # plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
    plt.plot(exp(filter_mean), color="y", label="exp(filter_mean)")
    plt.xlabel('t')

    plt.title('Poisson particle filter, variable rate')
    plt.legend()

    # residuals = y - filter_mean.T[0][:-1]
    # plot_acf_pacf(residuals, title="residuals autocorrelations")

    plt.figure(2)
    plt.plot(vals[:, :, 0], 'k.', markersize=0.8)
    plt.plot(x, 'r-', label="x [true]")
    plt.plot(filter_mean, color="y", label="filter_mean")
    plt.plot(smoothed_mean, color="pink", label="smoothed_mean")
    plt.legend()


    # learning paramaters

    def callback(params, Q, cur_iter):
        print(f"params = {params}")

    print(f"Real parameters {x0, phi, Q}")

    param0 = numpy.asarray([x0*1.2, phi/1.2, Q*1.2])
    estimator = param_est.ParamEstimation(model, u=None, y=y)   # subclass of Simulator

    model.set_param_bounds(Bounds([-numpy.inf, -1, 0.0001], [numpy.inf, 1, 10]))

    callback(param0, None, -1)
    params_estimated, _ = estimator.maximize(param0, num, nums, smoother='mcmc', meas_first=True, max_iter=5,
                       callback=callback)


    model.set_params(params_estimated)
    print(f"model parameters second iteration {model.params}")

    numpy.random.seed(666)
    sim.simulate(num, nums, smoother="mcmc", meas_first=True)   # meas_first is needed to treat the first y as measurement

    plt.figure(3)
    plt.title("filtering with estimated parameters")
    plt.plot(exp(x), 'r-', label="exp(x)  [true]")
    plt.plot(y, 'bx', label="y")
    (vals, _) = sim.get_filtered_estimates()
    filter_mean = sim.get_filtered_mean()

    # svals = sim.get_smoothed_estimates()
    smoothed_mean = sim.get_smoothed_mean()
    #
    # # # Plot "smoothed" trajectories to illustrate that the particle filter
    # # # suffers from degeneracy when considering the full trajectories
    # # plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
    plt.plot(exp(filter_mean), color="y", label="exp(filter_mean)")
    plt.xlabel('t')

    plt.title('Poisson particle filter, variable rate')
    plt.legend()

    # residuals = y - filter_mean.T[0][:-1]
    # plot_acf_pacf(residuals, title="residuals autocorrelations")

    plt.figure(4)
    plt.title("filtering with estimated parameters")
    plt.plot(vals[:, :, 0], 'k.', markersize=0.8)
    plt.plot(x, 'r-', label="x [true]")
    plt.plot(filter_mean, color="y", label="filter_mean")
    plt.plot(smoothed_mean, color="pink", label="smoothed_mean")
    plt.legend()



    plt.show()
