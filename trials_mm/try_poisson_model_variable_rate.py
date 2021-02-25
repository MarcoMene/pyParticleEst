import numpy
import pyparticleest.simulator
import matplotlib.pyplot as plt
from scipy.special import factorial

from numpy import exp,log

from builtins import range
import pyparticleest.interfaces as interfaces
from trials_mm.utils_plots import plot_acf_pacf


def generate_dataset(steps, x0, P0_x, phi, Q):
    x = numpy.zeros(steps)
    y = numpy.zeros(steps)

    x[0] = numpy.random.normal(x0, P0_x)
    y[0] = numpy.random.poisson(lam=exp(x[0]))

    for k in range(1, steps):
        x[k] = x0 + phi * x[k - 1] + numpy.random.normal(0, Q)
        y[k] = numpy.random.poisson(lam=exp(x[k]))

    return (x, y)


class PoissonModel(interfaces.ParticleFiltering):
    """
            ---> MM implementation of Poisson

            x_{k+1} = x0 + phi*x_k + e,   e ~ N(0, Q)  <-- log rate
            y_x_k = Poisson(   exp(x_k) )
            x(0) ~ Norm( x0, P0  )

            """

    def __init__(self, x_0, P0, phi, Q):
        self.P0 = numpy.copy(P0)
        self.x_0 = x_0
        self.phi = phi
        self.Q = Q

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, self.Q, (N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        """ Predict: Calculate xt+1 given xt using the supplied noise"""
        particles *= self.phi
        particles += self.x_0 + noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement --> p( y_t | x_t|t-1 ) """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            l = exp(particles[k].reshape(-1, 1))
            logyprob[k] = -l + y * numpy.log(l) - numpy.log(factorial(y))
        return logyprob  # should be a vector, that's why reshape is needed

    def create_initial_estimate(self, N):
        return numpy.random.normal(self.x_0, self.P0, size=(N,)).reshape((-1, 1))


if __name__ == '__main__':
    numpy.random.seed(666)

    steps = 100
    num = 50
    nums = 10

    x0 = 0.2
    P0_x = .5
    phi = 0.8
    Q = 1.

    (x, y) = generate_dataset(steps, x0, P0_x, phi, Q)
    plt.plot(exp(x), 'r-', label="exp(x)  [true]")
    plt.plot(y, 'bx', label="y")

    model = PoissonModel(x0, P0_x, phi, Q)  # it's better to enlarge variance for initial sampling
    sim = pyparticleest.simulator.Simulator(model, u=None, y=y)
    sim.simulate(num, nums, smoother="ancestor", meas_first=True)   # meas_first is needed to treat the first y as measurement

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

    plt.show()
