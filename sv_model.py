"""
Stochastic volatility model: 

    dX_t = \mu dt + \sqrt{Z_t} dW^x_t 
    dZ_t = (\theta + \kappa Z_t) dt + \sigma_z \sqrt{Z_t} dW_t^z

"""


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from jax import lax
import pfjax as pf
import pfjax.sde as sde


def euler_sim_nojump(key, x, dt, drift_diff, theta):
    """
    Simulate SDE with dense diffusion using Euler-Maruyama discretization.
    Args:
        key: PRNG key.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.
    Returns:
        Simulated SDE values. A vector of size `n_dims`.
    """
    _, diff_subkey, jump_subkey = random.split(key, 3)
    diff_process = drift_diff(diff_subkey, x, theta, dt)
#     jump_process = jump(jump_subkey, x, theta, dt)
    return diff_process #jnp.append(diff_process + jump_process, jump_process)

class SDEModel(object):
    def __init__(self, dt, n_res):
        self._dt = dt
        self._n_res = n_res
        
        def euler_sim(self, key, x, dt, theta):
            return euler_sim_nojump(key, x, dt, self.drift_diff, theta)
        
        setattr(self.__class__, 'euler_sim', euler_sim)
    
    def state_sample(self, key, x_prev, theta):
        def fun(carry, t):
            key, subkey = random.split(carry["key"])
            x = self.euler_sim(
                key=subkey, x=carry["x"],
                dt=self._dt/self._n_res, theta=theta
            )
            res = {"x": x, "key": key}
            return res, x
        init = {"x": x_prev[-1], "key": key}
        last, full = lax.scan(fun, init, jnp.arange(self._n_res))
        return full
    
    def is_valid_state(self, x, theta):
        return not jnp.sum(x < 0) > 0
    
    def meas_lpdf(self, y_curr, x_curr, theta):
        return 1.0


class StochVol(SDEModel):
    
    def __init__(self, dt, n_res, sigma_z):
        """ remove sigma_z from the model because it is apparently hard to estimate (form Golightly) """
        super().__init__(dt, n_res)
        self._n_state = (self._n_res, 2)
        self._sigma_z = sigma_z
    
    def _unpack(self, theta):
        return theta[0], theta[1], theta[2] #, theta[3]
#         return theta[0], theta[1], self._sigma_z, theta[3]
        
    def _validator(self, x):
        """ make sure vol and price never go negative.
        We want an absorbing state for Z, not a reflective state
        """
        return jnp.maximum(x, 1e-10)  # jnp.abs(x)
    
    def drift(self, x, theta):
#         _theta, kappa, sigma_z, mu = self._unpack(theta)
        x = self._validator(x)
        _theta, kappa, mu = self._unpack(theta)
        mu = jnp.array([_theta + kappa*x[0], mu])
        return mu
    
    def diff(self, x, theta):
        x = self._validator(x)
        _theta, kappa, mu = self._unpack(theta)
        Sigma = jnp.array([[x[0]*(self._sigma_z**2), 0],
                           [0, x[0]]])
        return Sigma
    
    def drift_diff(self, key, x, theta, dt):
        mu = self.drift(x, theta)
        Sigma = self.diff(x, theta)
        diff_process = jax.random.multivariate_normal(key, mean = x+mu*dt, cov=Sigma*dt)
        # diff_process = diff_process.at[0].set(jnp.abs(diff_process[0]))
        return diff_process
        
    def meas_sample(self, key, x_curr, theta):
        return x_curr[-1][1]
    
    def state_lpdf(self, x_curr, x_prev, theta):
        r"""
        Sample from Euler transition density: `p(x_curr | x_prev, theta)`
        """
        x0 = jnp.concatenate([x_prev[-1][None], x_curr[:-1]])
        x1 = x_curr
        
        def euler_lpdf_jump(x_curr, x_prev, dt, theta):
            return jsp.stats.norm.logpdf(
                x=x_curr[1],
                loc=x_prev[1] + self.drift(x_prev, theta)[1]*dt,
                scale=jnp.sqrt(self.diff(x_prev, theta)[1,1]*dt)
            ) 
        
        lp = jax.vmap(lambda xp, xc:
                      euler_lpdf_jump(
                          x_curr=xc, x_prev=xp,
                          dt=self._dt/self._n_res,
                          theta=theta))(x0, x1)
        return jnp.sum(lp)
    
    def _state_lpdf_for(self, x_curr, x_prev, theta):
        dt_res = self._dt/self._n_res
        x0 = jnp.append(jnp.expand_dims(
            x_prev[self._n_res-1], axis=0), x_curr[:self._n_res-1], axis=0)
        x1 = x_curr
        lp = jnp.array(0.0)
        
        for t in range(self._n_res):
            lp = lp + jnp.sum(jsp.stats.norm.logpdf(
                x=x1[t][1],
                loc=x0[t][1] + self.drift(x0[t], theta)[1]*dt_res,
                scale=jnp.sqrt(self.diff(x0[t], theta)[1,1]*dt_res)
            ))
        return lp
    
    def _bridge_param(self, x, y_curr, theta, n):
        _theta, kappa, mu = self._unpack(theta)
        k = self._n_res - n
        dt_res = self._dt/self._n_res
        x = self._validator(x)
        vol=x[0]
        price=x[1]

        mu_z = vol + (_theta+kappa*vol)*dt_res 
        sig2_z = vol*(self._sigma_z**2)*dt_res

        mu_x = price + (y_curr - price)/k 
        sig2_x = (k - 1)/k*vol*dt_res

        return mu_z, sig2_z, mu_x, sig2_x
    
    def pf_step(self, key, x_prev, y_curr, theta):
        def scan_fun(carry, t):
            key = carry["key"]
            x = carry["x"]
            
            mu_z, sig2_z, mu_x, sig2_x = self._bridge_param(x, y_curr, theta, t)
            key, z_subkey, x_subkey = random.split(key,3)

            x_prop = jnp.array([mu_z + jnp.sqrt(sig2_z) * random.normal(z_subkey),
                                jnp.where(t < self._n_res-1, 
                                          mu_x + jnp.sqrt(sig2_x) * random.normal(x_subkey),
                                          y_curr)])

            lp_prop = jnp.where(t < self._n_res-1,
                                jsp.stats.norm.logpdf(x=x_prop[1], loc=mu_x, scale=jnp.sqrt(sig2_x)),
                                0.0)

            res_carry = {
                "x": x_prop,
                "key": key,
                "lp": carry["lp"] + lp_prop
            }
            res_stack = {"x": x_prop, "lp": lp_prop}
            return res_carry, res_stack

        key, subkey = random.split(key)
        scan_init = {
            "x": x_prev[self._n_res-1],
            "key": subkey,
            "lp": jnp.array(0.)
        }
        
        ns = jnp.arange(self._n_res)

        last, full = lax.scan(scan_fun, scan_init, (ns))
        x_prop = full["x"]
        logw_trans = self.state_lpdf(
            x_curr=x_prop,
            x_prev=x_prev, 
            theta=theta
        )

        logw = logw_trans - last["lp"]
        return x_prop, logw
    
    def _pf_step_for(self, key, x_prev, y_curr, theta):
        dt_res = self._dt/self._n_res
        x_curr = []
        x_state = x_prev[self._n_res-1]
        lp = jnp.array(0.0)
        
        key, jump_subkey, z_subkey, x_subkey = random.split(key, 4)
        
        for t in range(self._n_res):
            key, z_subkey, x_subkey = random.split(key,3)
            mu_z, sig2_z, mu_x, sig2_x = self._bridge_param(x_state, y_curr, theta, t)
            
            x_state = jnp.array([jnp.abs(mu_z + jnp.sqrt(sig2_z) * random.normal(z_subkey))+1e-10,
                                 jnp.where(t<self._n_res-1, 
                                           mu_x + jnp.sqrt(sig2_x) * random.normal(x_subkey),
                                           y_curr)])

            lp_prop = jnp.where(t<self._n_res-1,
                           jsp.stats.norm.logpdf(x=x_state[1], loc=mu_x, scale=jnp.sqrt(sig2_x)),
                           0.0)
            
            x_curr.append(x_state)
            lp = lp + lp_prop
        
        x_prop = jnp.array(x_curr)
        
        logw_trans = self._state_lpdf_for(
            x_curr=x_prop,
            x_prev=x_prev, 
            theta=theta
        )
        
        logw = logw_trans - lp
        return x_prop, logw
    
    def pf_init(self, key, y_init, theta):
        key, subkey = random.split(key)
        x_init = y_init
        z_init = random.truncated_normal(
            subkey,
            lower=0.5,
            upper=10)
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init))
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + (self._n_state[1], )),
                       jnp.expand_dims(jnp.array([z_init, x_init]), axis = 0), axis=0), \
            logw