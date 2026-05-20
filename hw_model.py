import pandas as pd
import numpy as np


class HullWhiteModel:
    """
    Hull-White one-factor short rate model (no simulation).
    
    This class implements the analytical formulas for the 
    extended Vasicek/Hull-White model:
    
        dr(t) = a * (θ(t) - r(t)) dt + σ dW(t)
    
    where:
        a     = mean reversion speed
        σ     = volatility of the short rate
        θ(t)  = time-dependent drift fitted to the initial curve
    
    Attributes
    ----------
    curve : Curve
        Instance representing the initial discount curve P(0, T).
    parameters : dict
        Dictionary containing model parameters:
            - 'a' : float, mean reversion speed
            - 'sigma' : float, volatility
            - 'r0' : float, initial short rate
    a : float
        Mean reversion speed.
    sigma : float
        Volatility parameter.
    r0 : float
        Initial short rate.
    """

    def __init__(self, curve, parameters=None):
        """
        Initialize the Hull-White model with a given discount curve and parameters.

        Parameters
        ----------
        curve : Curve
            Discount curve used to fit θ(t) and compute forwards.
        parameters : dict, optional
            Dictionary with 'a', 'sigma', and 'r0'. If None, defaults are used: {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        """

        self.curve = curve

        # Default parameters
        defaults = {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        if parameters is None: parameters = {}
        self.parameters = {'a': parameters.get('a', defaults['a']),'sigma': parameters.get('sigma', defaults['sigma']),'r0': parameters.get('r0', defaults['r0'])}

    def inst_forward_rate(self, t):
        """
        Compute the instantaneous forward rate f(0, t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Instantaneous forward rate at time t.
        """
        return self.curve.inst_forward_rate(t)

    def discount_factor(self, t):
        """
        Compute the discount factor P(0, t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor for maturity t.
        """
        return self.curve.discount(t)
    
    def forward_rate(self, T1, T2):
        """
        Compute the simple forward rate F(0; T1, T2) implied by the discount curve. 
        Parameters
        ----------      
        T1 : float
            Start time of the forward rate.
        T2 : float
            End time of the forward rate.
        Returns 
        -------
        float
            Forward rate between T1 and T2.
        """
        return self.curve.forward_rate(T1, T2)

    def alpha(self, t):
        """
        Compute α(t), the deterministic shift function in Hull–White.

        Formula:
            α(t) = f(0, t) + (σ² / (2a²)) * (1 - e^{-a t})²

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            α(t) value.
        """
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        fwd = self.inst_forward_rate(t)
        return fwd + (sigma**2) / (2 * a**2) * (1 - np.exp(-a * t))**2

    def B(self, t, T):
        """
        Compute B(t, T) function used in bond pricing.

        Formula:
            B(t, T) = (1 - e^{-a (T - t)}) / a

        Parameters
        ----------
        t : float
            Start time in years.
        T : float
            Maturity time in years.

        Returns
        -------
        float
            B(t, T) value.
        """
        a = self.parameters['a']
        return (1 - np.exp(-a * (T - t))) / a

    def A(self, t, T):
        """
        Compute A(t, T) function used in zero-coupon bond pricing.

        Formula:
            A(t, T) = [P(0, T) / P(0, t)] * exp(B(t, T) * f(0, t) - 
                       (σ² / (4a)) * (1 - e^{-2a t}) * B(t, T)²)

        Parameters
        ----------
        t : float
            Start time in years.
        T : float
            Maturity time in years.

        Returns
        -------
        float
            A(t, T) value.
        """
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        P_t = self.discount_factor(t)
        P_T = self.discount_factor(T)
        fwd = self.inst_forward_rate(t)
        B = self.B(t, T)
        return (P_T / P_t) * np.exp(
            B * fwd - (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2
        )

    def short_rate(self, t, z=None):
        """
        Compute the short rate r(t) under the risk-neutral measure 
        using the exact distribution.

        Distribution:
            r(t) ~ Normal(mean = E[r(t)], variance = V[r(t)])

        Parameters
        ----------
        t : float
            Time in years.
        z : float, optional
            Standard normal draw. If None, one is generated.

        Returns
        -------
        float
            Simulated short rate at time t.
        """
        if z is None:
            z = np.random.normal()

        r0 = self.parameters['r0']
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        E = r0 * np.exp(-a * t) + self.alpha(t) - np.exp(-a * t) * self.alpha(0)
        return E + np.sqrt(V) * z

    def short_rate_forward(self, t, z=None):
        """
        Compute the short rate r(t) under the T-forward measure.

        This changes the drift to account for the bond maturing at t
        being the numeraire.

        Parameters
        ----------
        t : float
            Time in years.
        z : float, optional
            Standard normal draw. If None, one is generated.

        Returns
        -------
        float
            Simulated short rate at time t under the T-forward measure.
        """
        if z is None:
            z = np.random.normal()

        a = self.parameters['a']
        sigma = self.parameters['sigma']
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        E = self.curve.inst_forward_rate(t) 
        return E + np.sqrt(V) * z


class HullWhiteSimulation:
    """
    Monte Carlo simulation engine for the Hull–White one-factor model.

    Provides:
        - Exact simulation of r(T) at a single maturity (no path generation)
        - Euler–Maruyama path simulation under the risk-neutral measure
        - Simulation under the T-forward measure
        - Analytical validation of simulated mean and variance

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance providing parameters and curve.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps for Euler path simulation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, model: HullWhiteModel, n_paths=10**5, n_steps=100, seed=2025):
        """
        Initialize the Hull–White simulation engine.

        Parameters
        ----------
        model : HullWhiteModel
            Hull–White model instance.
        n_paths : int, optional
            Number of Monte Carlo paths (default: 100,000).
        n_steps : int, optional
            Number of steps for Euler path simulation (default: 100).
        seed : int, optional
            Random seed for reproducibility (default: 2025).
        """
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = np.random.seed(seed)

    def simulate_short_rate_direct(self, T):
        """
        Simulate r(T) using the exact analytical distribution 
        under the risk-neutral measure.

        Parameters
        ----------
        T : float
            Simulation horizon in years.

        Returns
        -------
        ndarray
            Array of simulated short rates (n_paths,).
        """
        z = np.random.normal(size=self.n_paths)
        r = np.array([self.model.short_rate(T, z=z_i) for z_i in z])
        return r

    def simulate_short_rate_direct_forward(self, T):
        """
        Simulate r(t) under the T-forward measure 
        using the exact analytical distribution.

        Parameters
        ----------
        T : float
            Simulation horizon in years.

        Returns
        -------
        ndarray
            Array of simulated short rates (n_paths,).
        """
        z = np.random.normal(size=self.n_paths)
        r = np.array([self.model.short_rate_forward(T, z=z_i) for z_i in z])
        return r

    def simulate_short_rate_euler(self, T):
        """
        Simulate r(t) paths under the risk-neutral measure 
        using Euler–Maruyama discretization.

        Parameters
        ----------
        T : float
            Simulation horizon in years.

        Returns
        -------
        r : ndarray
            Simulated short rate paths of shape (n_paths, n_steps+1).
        times : ndarray
            Corresponding time grid.
        """
        dt = T / self.n_steps
        times = np.linspace(0, T, self.n_steps + 1)
        x = np.zeros((self.n_paths, self.n_steps + 1))
        r = np.zeros_like(x)

        x[:, 0] = self.model.parameters['r0'] - self.model.alpha(0)
        r[:, 0] = self.model.parameters['r0']

        for i in range(1, self.n_steps + 1):
            z = np.random.normal(size=self.n_paths)
            x[:, i] = x[:, i - 1] - self.model.parameters['a'] * x[:, i - 1] * dt + self.model.parameters['sigma'] * np.sqrt(dt) * z
            r[:, i] = x[:, i] + self.model.alpha(times[i])

        return r, times

    def validate_simulation(self, T):
        """
        Compare the simulated mean and standard deviation 
        with analytical values for r(T).

        Parameters
        ----------
        T : float
            Simulation horizon in years.

        Returns
        -------
        DataFrame
            Table comparing Euler simulation, direct simulation, and analytic values.
        """
        r_euler, _ = self.simulate_short_rate_euler(T)
        r_euler_end = r_euler[:, -1]
        r_direct = self.simulate_short_rate_direct(T)

        analytic_mean = self.model.parameters['r0'] * np.exp(-self.model.parameters['a'] * T) + self.model.alpha(T) - np.exp(-self.model.parameters['a'] * T) * self.model.alpha(0)
        analytic_std = np.sqrt((self.model.parameters['sigma']**2) / (2 * self.model.parameters['a']) * (1 - np.exp(-2 * self.model.parameters['a'] * T)))

        data = {
            "Mean": [np.mean(r_euler_end), np.mean(r_direct), analytic_mean],
            "Std Dev": [np.std(r_euler_end), np.std(r_direct), analytic_std]
        }
        df = pd.DataFrame(data, index=["Euler Simulation", "Direct Simulation", "Analytic"])
        return df


class HullWhiteCurveBuilder:
    """
    Hull–White curve builder that provides both analytical formulas and Monte Carlo 
    simulation utilities for pricing zero-coupon bonds, discount factors, 
    forward rates, and long-term rates, using a pre-built discount curve.

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance constructed from the provided curve and parameters.
    sim : HullWhiteSimulation
        Monte Carlo simulation engine built from the Hull–White model.
    curve : Curve
        Pre-initialized discount curve used for forwards and short rate calculations.
    """

    def __init__(self, curve, params=None, n_paths=10**5, n_steps=100, seed=2025, smooth=1e-7):
        """
        Initialize the Hull–White curve builder using a pre-built Curve instance and 
        Hull–White model parameters.

        Parameters
        ----------
        Curve : Curve
            Pre-initialized discount curve instance containing times to maturity 
            and discount factors.
        params : dict, optional
            Dictionary containing Hull–White model parameters (optional):
                - 'a' : float, mean reversion speed
                - 'sigma' : float, volatility
                - 'r0' : float, initial short rate
            If None, defaults are used: {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        n_paths : int, optional
            Number of Monte Carlo paths (default: 100,000).
        n_steps : int, optional
            Number of discretization steps per path (default: 100).
        seed : int, optional
            Random seed for reproducibility (default: 2025).
        smooth : float, optional
            Smoothing parameter for the discount curve (not used if Curve is already initialized).

        Workflow
        --------
        1. Use the provided Curve instance for discount factors and instantaneous forwards.
        2. Build the Hull–White model using the curve and provided parameters.
        3. Initialize the Monte Carlo simulation engine for short rate paths.
        """
        self.curve = curve
        self.model = HullWhiteModel(self.curve, params)
        self.sim = HullWhiteSimulation(self.model, n_paths=n_paths, n_steps=n_steps, seed=seed)


    def short_rate(self, t, fwd_measure=False):
        """
        Simulate the short rate r(t) at a single time t using the exact distribution.

        Parameters
        ----------
        t : float
            Time in years.
        fwd_measure : bool, optional
            If True, simulate r(t) under the T-forward measure.

        Returns
        -------
        ndarray
            Array of simulated short rates (n_paths,).
        """
        if fwd_measure:
            return self.sim.simulate_short_rate_direct_forward(t)
        else:
            return self.sim.simulate_short_rate_direct(t)


    def zero_coupon_bond(self, t, T, fwd_measure=False):
        """
        Price a zero-coupon bond analytically under either the risk-neutral 
        or T-forward measure.

        Formula:
            P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        Parameters
        ----------
        t : float
            Current time in years.
        T : float
            Bond maturity in years.
        fwd_measure : bool, optional
            If True, simulate r(t) under the T-forward measure.

        Returns
        -------
        ndarray
            Bond price distribution.
        """
        if fwd_measure:
            r_t = self.sim.simulate_short_rate_direct_forward(t)
        else:
            r_t = self.sim.simulate_short_rate_direct(t)

        A = self.model.A(t, T)
        B = self.model.B(t, T)
        price = A * np.exp(-B * r_t)
        return price

    def discount_factor(self, t, T):
        """
        Compute the discount factor between t and T using Monte Carlo paths.

        Parameters
        ----------
        t : float
            Start time in years.
        T : float
            End time in years.

        Returns
        -------
        ndarray
            Discount factor distribution
        """
        r_paths, times = self.sim.simulate_short_rate_euler(T)
        idx_T = np.searchsorted(times, T)
        idx_t = np.searchsorted(times, t)

        dt = times[1] - times[0]
        integral_r = np.sum(r_paths[:, idx_t:idx_T] * dt, axis=1)
        df = np.exp(-integral_r)
        return df

    def inst_forward_rate(self, t, T):
        """
        Compute the instantaneous forward rate f(t, T) using zero-coupon bonds.

        Parameters
        ----------
        t : float
            Current time in years.
        T : float
            Forward rate maturity in years.

        Returns
        -------
        ndarray
            Instantaneous forward rates.
        """
        r_t = self.sim.simulate_short_rate_direct(t)
        fwd_T = self.model.inst_forward_rate(T)
        fwd_t = self.model.inst_forward_rate(t)
        B = self.model.B(t, T)
        a = self.model.parameters['a']
        sigma = self.model.parameters['sigma']
        K = (sigma**2) * (1 - np.exp(-2 * a * t)) / (2 * a)
        f = fwd_T + np.exp(-a * (T - t)) * (r_t - fwd_t + K * B)
        return f

    def long_rate(self, t, T, fwd_measure=False):
        """
        Compute the continuous compounding long-term rate R(t, T).

        Formula:
            R(t, T) = -log(A(t, T)) / (T - t) + (B(t, T) / (T - t)) * r(t)

        Parameters
        ----------
        t : float
            Current time in years.
        T : float
            Long rate maturity in years.
        fwd_measure : bool, optional
            If True, simulate r(t) under the T-forward measure.

        Returns
        -------
        ndarray
            Long rate distribution.
        """

        if fwd_measure:
            r_t = self.sim.simulate_short_rate_direct_forward(t)
        else:   
            r_t = self.sim.simulate_short_rate_direct(t)

        A = self.model.A(t, T)
        B = self.model.B(t, T)
        alpha = -np.log(A) / (T - t)
        beta = B / (T - t)
        R = alpha + beta * r_t
        return R
    
    def forward_rate(self, t, T1, T2, fwd_measure=False):
        """
        Compute the simple forward rate F(t; T1, T2) between T1 and T2.

        Formula:
            F(t; T1, T2) = (1 / (T2 - T1)) * (P(t, T1) / P(t, T2) - 1)

        Parameters
        ----------
        t : float
            Current time in years.
        T1 : float
            Start of the forward period in years.
        T2 : float
            End of the forward period in years.
        fwd_measure : bool, optional
            If True, simulate r(t) under the T2-forward measure.

        Returns
        -------
        ndarray
            Forward rates for each Monte Carlo path.
        """
        P_t_T1 = self.zero_coupon_bond(t, T1, fwd_measure=fwd_measure)
        P_t_T2 = self.zero_coupon_bond(t, T2, fwd_measure=fwd_measure)
        F = (1 / (T2 - T1)) * (P_t_T1 / P_t_T2 - 1)
        return F
    

    def coupon_bearing_bond(self, t, Tau, K, N, fwd_measure=False):
        """
        Compute the future values of a coupon-bearing bond.

        Parameters
        ----------
        t : float
            Current time in years.
        Tau : list of float
            Remaining payment dates of the bond (T1, T2, ..., TN).
        K : float
            Coupon rate (fixed rate per period).
        N : float
            Notional (scaling factor).
        fwd_measure : bool, optional
            If True, simulate r(t) under the T-forward measure.

        Returns
        -------
        ndarray
            Bond price distribution.
        """

        Delta = Tau[1] - Tau[0]

        # Remove payments already made
        if Tau[0] < t:
            Tau = [time for time in Tau if time >= t]

        C = K * N * Delta
        CB = 0

        for i in range(len(Tau)):
            P_t_Ti = self.zero_coupon_bond(t, Tau[i], fwd_measure = fwd_measure)
            CB += C * P_t_Ti

        CB = CB + N * self.zero_coupon_bond(t, Tau[-1], fwd_measure = fwd_measure) 

        return CB
    

