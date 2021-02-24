import matplotlib.pyplot as plt
from bsp_data_science.types import ArrayLike
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def show_plot():
    print("Enjoy the plot")
    plt.show()


def plot_acf_pacf(x: ArrayLike, title=f"ACF - PACF", lags=10):
    """
    Plots ACF/PACF of a give array side by side
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    plot_acf(x, lags=lags, ax=ax[0])
    plot_pacf(x, lags=lags, ax=ax[1])
