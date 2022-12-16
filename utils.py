from pmmh import pmmh
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


def plot_posteriors(pg_out, theta_true, warmup_frac=5):
    num_warmup = pg_out["theta"].shape[0] // warmup_frac

    posteriors = pd.DataFrame(
        pg_out["theta"][num_warmup:], 
        columns = ["theta", "kappa", "alpha"])

    plot_posteriors = pd.melt(posteriors, var_name = "param")

    g = sns.FacetGrid(plot_posteriors,
                      sharex=True, sharey=0,
                      col="param", height=4, aspect=1.3, col_wrap=3)
    g.map(plt.plot, "value");

    g = sns.FacetGrid(plot_posteriors,
                      sharex=False, sharey=1,
                      col="param", height=4, aspect=1.3, col_wrap=3)
    g.map(sns.histplot, "value")
    [g.axes[i].axvline(x=theta_true[i], color = "firebrick", ls='-', linewidth = 3) for i in range(len(theta_true))];
    [g.axes[i].axvline(x=pg_out["theta"][:, i].mean(), color = "royalblue", ls='-', linewidth = 3) for i in range(len(theta_true))];


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


def parameter_estimates(key, model, y_meas, theta_init, n_particles, n_iter, logprior, adapt_max=0.0, rw_sd=None):
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
    pg_out = pmmh(
        key=key,
        model=model,
        n_iter=n_iter,
        theta_init=theta_init,
        x_state_init=latent_state_est,
        y_meas=y_meas,
        n_particles=n_particles,
        rw_sd=rw_sd,
        adapt_max=adapt_max,
        adapt_rate=0.5,
        logprior=logprior
    )

    print("Acceptance rate: ", pg_out["accept_rate"])
    return pg_out

def discrepancy_samples(pos_1, pos_2, disc_measure, n):
    """
    Performs random shuffling of pooled population and returns discrepancy samples
    Non-parametric, and no assumption required
    """
    df = pd.DataFrame()
    len1 = len(pos_1)
    pooled = pd.concat([pos_1, pos_2])

    for i in range(n):
        shuffled = pooled.sample(frac=1, random_state=i)
        pos_1_sample = shuffled.iloc[:len1,:]
        pos_2_sample = shuffled.iloc[len1:,:]
        df_tmp = pd.DataFrame(disc_measure(pos_1_sample, pos_2_sample)).transpose()
        df = df.append(df_tmp)

    return df

