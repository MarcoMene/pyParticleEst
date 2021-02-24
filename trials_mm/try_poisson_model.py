import numpy
import pyparticleest.simulator
import matplotlib.pyplot as plt
from scipy.special import factorial

from builtins import range
import pyparticleest.interfaces as interfaces
from trials_mm.utils_plots import plot_acf_pacf


def generate_dataset(steps, x_mean, P0_x, delta_t=1.):
    x = numpy.zeros(steps)
    y = numpy.zeros(steps)

    k = x_mean ** 2 / P0_x ** 2  # shape
    theta = P0_x ** 2 / x_mean  # scale
    x[0] = numpy.random.gamma(shape=k, scale=theta)
    y[0] = numpy.random.poisson(lam=x[0] * delta_t)

    for k in range(1, steps):
        x[k] = x[k - 1]
        y[k] = numpy.random.poisson(lam=x[k] * delta_t)

    return (x, y)


class PoissonModel(interfaces.ParticleFiltering):
    """
            ---> MM implementation of Poisson

            x_{k+1} = x_k  <-- No variance on underlying rate
            y_x_k = Poisson(x_k * dt)
            x(0) ~ Gamma(x_mean, P0)

            """

    def __init__(self, P0, x_mean=1.):
        self.P0 = numpy.copy(P0)
        self.x_mean = x_mean

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.zeros((N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        """ Calculate xt+1 given xt using the supplied noise"""
        particles += noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement --> p( y_t | x_t|t-1 ) """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            l = particles[k].reshape(-1, 1)
            logyprob[k] = -l + y * numpy.log(l) - numpy.log(factorial(y))
        return logyprob  # should be a vector, that's why reshape is needed

    def create_initial_estimate(self, N):
        k = self.x_mean ** 2 / self.P0 ** 2  # shape
        theta = self.P0 ** 2 / self.x_mean  # scale
        return numpy.random.gamma(shape=k, scale=theta, size=(N,)).reshape((-1, 1))


if __name__ == '__main__':
    numpy.random.seed(666)

    steps = 100
    num = 500
    nums = 100

    x_mean = 0.5
    P0_x = 1.0

    delta_t = 1

    (x, y) = generate_dataset(steps, x_mean, P0_x, delta_t=delta_t)

    model = PoissonModel(P0=P0_x, x_mean=1.)
    sim = pyparticleest.simulator.Simulator(model, u=None, y=y)
    sim.simulate(num, nums, smoother="ancestor")
    plt.plot(x, 'r-', label="true x")
    plt.plot(y, 'bx', label="y")

    (vals, _) = sim.get_filtered_estimates()
    filter_mean = sim.get_filtered_mean()

    # plt.plot(range(steps + 1), vals[:, :, 0], 'k.', markersize=0.8)

    svals = sim.get_smoothed_estimates()
    smoothed_mean = sim.get_smoothed_mean()

    # # Plot "smoothed" trajectories to illustrate that the particle filter
    # # suffers from degeneracy when considering the full trajectories
    plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
    plt.plot(range(steps + 1), filter_mean, color="y", label="filter_mean")
    plt.plot(range(steps + 1), smoothed_mean, color="pink", label="smoothed_mean")
    plt.xlabel('t')
    plt.title('Poisson particle filter')
    plt.legend()

    # residuals = y - filter_mean.T[0][:-1]
    # plot_acf_pacf(residuals, title="residuals autocorrelations")

    plt.show()
