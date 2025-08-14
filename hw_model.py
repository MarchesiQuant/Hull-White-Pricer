import pandas as pd
import numpy as np
from scipy.stats import norm


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

    def __init__(self, curve, parameters):
        """
        Initialize the Hull-White model with a given discount curve and parameters.

        Parameters
        ----------
        curve : Curve
            Discount curve used to fit θ(t) and compute forwards.
        parameters : dict
            Dictionary with 'a', 'sigma', and 'r0'.
        """
        self.curve = curve
        self.parameters = parameters
        self.a = parameters['a']
        self.sigma = parameters['sigma']
        self.r0 = parameters['r0']

    def forward_rate(self, t):
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
        return self.curve.forward(t)

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

    def _alpha(self, t):
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
        fwd = self.forward_rate(t)
        return fwd + (sigma**2) / (2 * a**2) * (1 - np.exp(-a * t))**2

    def _B(self, t, T):
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

    def _A(self, t, T):
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
        fwd = self.forward_rate(t)
        B = self._B(t, T)
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
        E = r0 * np.exp(-a * t) + self._alpha(t) - np.exp(-a * t) * self._alpha(0)
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
        E = self.curve.forward(t)
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

        x[:, 0] = self.model.r0 - self.model._alpha(0)
        r[:, 0] = self.model.r0

        for i in range(1, self.n_steps + 1):
            z = np.random.normal(size=self.n_paths)
            x[:, i] = x[:, i - 1] - self.model.a * x[:, i - 1] * dt + self.model.sigma * np.sqrt(dt) * z
            r[:, i] = x[:, i] + self.model._alpha(times[i])

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

        analytic_mean = self.model.r0 * np.exp(-self.model.a * T) + self.model._alpha(T) - np.exp(-self.model.a * T) * self.model._alpha(0)
        analytic_std = np.sqrt((self.model.sigma**2) / (2 * self.model.a) * (1 - np.exp(-2 * self.model.a * T)))

        data = {
            "Mean": [np.mean(r_euler_end), np.mean(r_direct), analytic_mean],
            "Std Dev": [np.std(r_euler_end), np.std(r_direct), analytic_std]
        }
        df = pd.DataFrame(data, index=["Euler Simulation", "Direct Simulation", "Analytic"])
        return df


class HullWhiteCurveBuilder:
    """
    Analytical and Monte Carlo pricing utilities for bonds, discount factors,
    forward rates, and long-term rates under the Hull–White model.

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance.
    sim : HullWhiteSimulation
        Simulation engine.
    """

    def __init__(self, model, simulation):
        """
        Initialize the curve builder.

        Parameters
        ----------
        model : HullWhiteModel
            Hull–White model instance.
        simulation : HullWhiteSimulation
            Simulation engine for Monte Carlo computations.
        """
        self.model = model
        self.sim = simulation

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
            Bond price(s) for each Monte Carlo path.
        """
        if fwd_measure:
            r_t = self.sim.simulate_short_rate_direct_forward(t)
        else:
            r_t = self.sim.simulate_short_rate_direct(t)

        A = self.model._A(t, T)
        B = self.model._B(t, T)
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
            Discount factor for each Monte Carlo path.
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
        Compute the instantaneous forward rate F(t, T) using zero-coupon bonds.

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
        fwd_T = self.model.forward_rate(T)
        fwd_t = self.model.forward_rate(t)
        B = self.model._B(t, T)
        a = self.model.parameters['a']
        sigma = self.model.parameters['sigma']
        K = (sigma**2) * (1 - np.exp(-2 * a * t)) / (2 * a)
        F = fwd_T + np.exp(-a * (T - t)) * (r_t - fwd_t + K * B)
        return F

    def long_rate(self, t, T):
        """
        Compute the long-term rate R(t, T).

        Formula:
            R(t, T) = -log(A(t, T)) / (T - t) + (B(t, T) / (T - t)) * r(t)

        Parameters
        ----------
        t : float
            Current time in years.
        T : float
            Long rate maturity in years.

        Returns
        -------
        ndarray
            Long rates for each Monte Carlo path.
        """
        A = self.model._A(t, T)
        B = self.model._B(t, T)
        r_t = self.sim.simulate_short_rate_direct(t)
        alpha = -np.log(A) / (T - t)
        beta = B / (T - t)
        R = alpha + beta * r_t
        return R


class HullWhitePricer:
    """
    Pricing engine for interest rate derivatives under the Hull–White one-factor model.

    Supports:
        - Zero-coupon bond options (calls & puts)
        - Caps and floors
        - Monte Carlo or closed-form valuation

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance providing parameters and analytical formulas.
    curve_sim : HullWhiteCurveBuilder
        Curve builder capable of generating zero-coupon bonds, discount factors, etc.
    """

    def __init__(self, model, curve_sim):
        """
        Initialize the Hull–White pricer.

        Parameters
        ----------
        model : HullWhiteModel
            Hull–White model instance.
        curve_sim : HullWhiteCurveBuilder
            Curve builder for Monte Carlo valuations.
        """
        self.model = model
        self.curve_sim = curve_sim

    def zero_bond_put(self, T, S, K, mc=False):
        """
        Value a European put option on a zero-coupon bond P(T, S).

        Parameters
        ----------
        T : float
            Option maturity in years.
        S : float
            Bond maturity in years (must be S > T).
        K : float
            Strike price.
        mc : bool, optional
            If True, value by Monte Carlo; otherwise use closed form.

        Returns
        -------
        float
            Present value of the put option.
        """
        if T == 0:
            P_0S = self.model.discount_factor(S)
            return max(K - P_0S, 0)

        if mc:
            D_T = self.curve_sim.discount_factor(0, T)
            P_TS = self.curve_sim.zero_coupon_bond(T, S)
            payoff = np.maximum(K - P_TS, 0)
            V0 = np.mean(D_T * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            B = self.model._B(T, S)
            P_S = self.model.discount_factor(S)
            P_T = self.model.discount_factor(T)
            sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B
            h = (1 / sigma_p) * np.log(P_S / (K * P_T)) + 0.5 * sigma_p
            V0 = K * P_T * norm.cdf(-h + sigma_p) - P_S * norm.cdf(-h)

        return V0

    def zero_bond_call(self, T, S, K, mc=False):
        """
        Value a European call option on a zero-coupon bond P(T, S).

        Parameters
        ----------
        T : float
            Option maturity in years.
        S : float
            Bond maturity in years (must be S > T).
        K : float
            Strike price.
        mc : bool, optional
            If True, value by Monte Carlo; otherwise use closed form.

        Returns
        -------
        float
            Present value of the call option.
        """
        if mc:
            D_T = self.curve_sim.discount_factor(0, T)
            P_TS = self.curve_sim.zero_coupon_bond(T, S)
            payoff = np.maximum(P_TS - K, 0)
            V0 = np.mean(D_T * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            B = self.model._B(T, S)
            P_S = self.model.discount_factor(S)
            P_T = self.model.discount_factor(T)
            sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B
            h = (1 / sigma_p) * np.log(P_S / (K * P_T)) + 0.5 * sigma_p
            V0 = P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sigma_p)

        return V0

    def cap(self, Tau, N, K, mc=False):
        """
        Value an interest rate cap using caplets.

        Parameters
        ----------
        Tau : list of float
            Payment times for caplets (first entry is fixing time, not payment).
        N : float
            Notional amount.
        K : float
            Cap strike rate.
        mc : bool, optional
            If True, value via Monte Carlo; otherwise use closed form.

        Returns
        -------
        float
            Present value of the cap.
        """
        Cap = 0
        if mc:
            for i in range(1, len(Tau)):
                T1 = Tau[i - 1]
                T2 = Tau[i]
                R_T = self.curve_sim.long_rate(T1, T2)
                Delta = T2 - T1
                payoff = Delta * np.maximum(R_T - K, 0)
                D_T = self.curve_sim.discount_factor(0, T2)
                Cap += np.mean(D_T * payoff)
        else:
            for i in range(1, len(Tau)):
                t_prev = Tau[i - 1]
                t_curr = Tau[i]
                Delta = t_curr - t_prev
                K_bond = 1 + K * Delta
                put_price = self.zero_bond_put(t_prev, t_curr, 1 / K_bond)
                Cap += K_bond * put_price

        return N * Cap

    def floor(self, Tau, N, K, mc=False):
        """
        Value an interest rate floor using floorlets.

        Parameters
        ----------
        Tau : list of float
            Payment times for floorlets (first entry is fixing time, not payment).
        N : float
            Notional amount.
        K : float
            Floor strike rate.
        mc : bool, optional
            If True, value via Monte Carlo; otherwise use closed form.

        Returns
        -------
        float
            Present value of the floor.
        """
        Floor = 0
        if mc:
            for i in range(1, len(Tau)):
                T1 = Tau[i - 1]
                T2 = Tau[i]
                R_T = self.curve_sim.long_rate(T1, T2)
                Delta = T2 - T1
                payoff = Delta * np.maximum(K - R_T, 0)
                D_T = self.curve_sim.discount_factor(0, T2)
                Floor += np.mean(D_T * payoff)
        else:
            for i in range(1, len(Tau)):
                t_prev = Tau[i - 1]
                t_curr = Tau[i]
                Delta = t_curr - t_prev
                K_bond = 1 + K * Delta
                call_price = self.zero_bond_call(t_prev, t_curr, 1 / K_bond)
                Floor += K_bond * call_price

        return N * Floor
