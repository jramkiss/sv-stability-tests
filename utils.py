import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from jax import lax
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import pfjax as pf
from functools import partial

import warnings
warnings.filterwarnings("ignore")


def quantile_index (logw, q):
    """
    Returns the index of the q-th quantile of logw
    """
    w = pf.utils.logw_to_prob(logw)
    val = jnp.quantile(w, q=q)
    nearest_ind = jnp.argmin(jnp.abs(val - w)) # find index of closest point to val
    return nearest_ind


def plot_posteriors(pg_out, theta_true, theta_init, warmup_frac=5):
    num_warmup = pg_out["theta"].shape[0] // warmup_frac

    posteriors = pd.DataFrame(
        pg_out["theta"][num_warmup:], 
        columns = ["theta", "kappa", "mu"])

    plot_posteriors = pd.melt(posteriors, var_name = "param")

    g = sns.FacetGrid(plot_posteriors,
                      sharex=True, sharey=0,
                      col="param", height=4, aspect=1.3, col_wrap=3)
    g.map(plt.plot, "value");

    g = sns.FacetGrid(plot_posteriors,
                      sharex=False, sharey=1,
                      col="param", height=4, aspect=1.3, col_wrap=3)
    g.map(sns.histplot, "value")
    [g.axes[i].axvline(x=theta_true[i], color = "firebrick", ls='-') for i in range(len(theta_true))];
    [g.axes[i].axvline(x=theta_init[i], color = "green", ls='--') for i in range(len(theta_true))];
    [g.axes[i].axvline(x=pg_out["theta"][:, i].mean(), color = "black", ls='-') for i in range(len(theta_true))];

