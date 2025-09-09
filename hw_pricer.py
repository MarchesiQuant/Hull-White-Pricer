import numpy as np
from scipy.stats import norm



class HullWhitePricer:
    """
    Pricing engine for interest rate derivatives under the Hull–White one-factor model,
    using a single HullWhiteCurveBuilder instance.

    Supports:
        - Zero-coupon bond options (calls & puts)
        - Caps and floors
        - Swaps and swaptions
        - Monte Carlo or closed-form valuation

    Attributes
    ----------
    curve_builder : HullWhiteCurveBuilder
        Hull–White curve builder providing the model, simulation engine, and discount curve.
    """

    def __init__(self, curve_builder):
        """
        Initialize the Hull–White pricer using a single HullWhiteCurveBuilder instance.

        Parameters
        ----------
        curve_builder : HullWhiteCurveBuilder
            Pre-initialized Hull–White curve builder containing the model, simulation engine,
            and discount curve.
        """
        self.curve_builder = curve_builder
        self.model = curve_builder.model
        self.curve_sim = curve_builder

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
            If True, value via Monte Carlo (fwd measure); otherwise use closed form.

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
                F_T1 = self.curve_sim.forward_rate(T1, T1, T2, fwd_measure=True)
                Delta = T2 - T1
                payoff = Delta * np.maximum(F_T1 - K, 0)
                P_T2 = self.model.discount_factor(T2)
                Cap += P_T2 * np.mean(payoff)
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
            If True, value via Monte Carlo (fwd measure); otherwise use closed form.

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
                F_T1 = self.curve_sim.forward_rate(T1, T1, T2, fwd_measure=True)
                Delta = T2 - T1
                payoff = Delta * np.maximum(K - F_T1, 0)
                P_T2 = self.model.discount_factor(T2)
                Floor += P_T2*np.mean(payoff)
        else:
            for i in range(1, len(Tau)):
                t_prev = Tau[i - 1]
                t_curr = Tau[i]
                Delta = t_curr - t_prev
                K_bond = 1 + K * Delta
                call_price = self.zero_bond_call(t_prev, t_curr, 1 / K_bond)
                Floor += K_bond * call_price

        return N * Floor
    
    # This needs a fix
    def swap(self, Tau, N, K, mc=False):
        """
        Value a plain vanilla interest rate swap.

        Parameters
        ----------
        Tau : list of float
            Payment times for the fixed leg (first entry is start time).
        N : float
            Notional amount.
        K : float
            Fixed rate.
        mc : bool, optional
            If True, value via Monte Carlo (fwd measure); otherwise use closed form.

        Returns
        -------
        float
            Present value of the swap.
        """

        Annuity = 0
        for i in range(1, len(Tau)):
            Delta = Tau[i] - Tau[i-1]
            P_T = self.model.discount_factor(Tau[i])
            Annuity += Delta * P_T

        Fixed_leg = Annuity * K
        Floating_leg = 0

        if mc:
            for i in range(1, len(Tau)):
                T1 = Tau[i - 1]
                T2 = Tau[i]
                Delta = T2 - T1
                P_T2 = self.model.discount_factor(T2)               
                F_T1 = self.curve_sim.forward_rate(T1, T1, T2, fwd_measure=True)
                Floating_leg += P_T2 * Delta * np.mean(F_T1)
        else:
            Floating_leg = self.model.discount_factor(Tau[0]) - self.model.discount_factor(Tau[-1])

        Swap = (Floating_leg - Fixed_leg) * N
        return Swap
