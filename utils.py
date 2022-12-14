from particle_gibbs import particle_gibbs
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


def init_latents(key, model, y_meas, theta_init, n_particles):
    # estimate of latent states for PG
    filtering_dist = pf.particle_filter(
        model=model,
        key=key,
        y_meas=y_meas,
        n_particles=n_particles,
        theta=theta_init,
        history=True)
    latent_state_est = jax.vmap(
        lambda x, w: jnp.average(x, axis=0, weights=pf.utils.logw_to_prob(w)),
        in_axes=(0, 0))(filtering_dist["x_particles"],
                        filtering_dist["logw"])
    return latent_state_est


def parameter_estimates(key, model, y_meas, theta_init, n_particles, n_iter, logprior, rw_sd=None):
    """
    Sample posteriors for parameters of `model` with Particle MWG sampler
    """
    if rw_sd is None:
        rw_sd = jnp.abs(theta_init)/10

    # estimate of latent states for PG
    filtering_dist = pf.particle_filter(
        model=model,
        key=key,
        y_meas=y_meas,
        n_particles=n_particles,
        theta=theta_init,
        history=True)
    latent_state_est = jax.vmap(
        lambda x, w: jnp.average(x, axis=0, weights=pf.utils.logw_to_prob(w)),
        in_axes=(0, 0))(filtering_dist["x_particles"],
                        filtering_dist["logw"])

    # run particle MWG:
    pg_out = particle_gibbs(
        key=key,
        model=model,
        n_iter=n_iter,
        theta_init=theta_init,
        x_state_init=latent_state_est,
        y_meas=y_meas,
        n_particles=n_particles,
        rw_sd=rw_sd,
        adapt_max=0.1,
        adapt_rate=0.5,
        logprior=logprior
    )

    print("Acceptance rate: ", pg_out["accept_rate"])
    return pg_out

