import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from hw_model import HullWhiteCurveBuilder


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

    def __init__(self, curve, n_paths=10**5, n_steps=252, seed=2025, hw_params=None):
        self.curve = curve
        self.curve_sim = HullWhiteCurveBuilder(curve, params = hw_params, n_paths = n_paths, n_steps = n_steps, seed = seed)
        self.model = self.curve_sim.model


    def set_simulation(self, n_paths=None, n_steps=None, seed=None):
        """
        Update Monte Carlo simulation settings in-place after initialization.

        Parameters
        ----------
        n_paths : int, optional
            New number of Monte Carlo paths.
        n_steps : int, optional
            New number of Euler discretization steps.
        seed : int, optional
            Random seed to reseed NumPy's RNG.
        """
        if n_paths is not None: self.curve_sim.sim.n_paths = int(n_paths)
        if n_steps is not None: self.curve_sim.sim.n_steps = int(n_steps)
        if seed is not None: np.random.seed(seed)


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
            If True, value by Monte Carlo, using the forward measure; otherwise use closed form.

        Returns
        -------
        float
            Present value of the put option.
        """
        if T == 0:
            P_0S = self.model.discount_factor(S)
            return max(K - P_0S, 0)

        if mc:
            D_T = self.model.discount_factor(T)
            P_TS = self.curve_sim.zero_coupon_bond(T, S, fwd_measure=True)
            payoff = np.maximum(K - P_TS, 0)
            V0 = np.mean(D_T * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            B = self.model.B(T, S)
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
            If True, value by Monte Carlo, using the forward measure; otherwise use closed form.

        Returns
        -------
        float
            Present value of the call option.
        """
        if mc:
            D_T = self.model.discount_factor(T)
            P_TS = self.curve_sim.zero_coupon_bond(T, S, fwd_measure=True)
            payoff = np.maximum(P_TS - K, 0)
            V0 = np.mean(D_T * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            B = self.model.B(T, S)
            P_S = self.model.discount_factor(S)
            P_T = self.model.discount_factor(T)
            sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B
            h = (1 / sigma_p) * np.log(P_S / (K * P_T)) + 0.5 * sigma_p
            V0 = P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sigma_p)

        return V0
    
    
    def caplet(self, T1, T2, N, K, method='js'):
        """
        Value an interest rate caplet.

        Parameters
        ----------
        T1 : float
            Fixing time.
        T2 : float
            Payment time (must be T2 > T1).
        N : float
            Notional amount.
        K : float
            Caplet strike rate.
        method : str, optional
            If 'mc', value via Monte Carlo (fwd measure); if 'js', use zero bond put, if 'cf', use closed form.

        Returns
        -------
        float
            Present value of the caplet.
        """
        Delta = T2 - T1
        K_bond = 1 + K * Delta

        if method == 'mc':
            F_T1 = self.curve_sim.forward_rate(T1, T1, T2, fwd_measure=True)
            payoff = Delta * np.maximum(F_T1 - K, 0)
            P_T2 = self.model.discount_factor(T2)
            Caplet = P_T2 * np.mean(payoff)

        elif method == 'js':
            put_price = self.zero_bond_put(T1, T2, 1 / K_bond)
            Caplet = K_bond * put_price

        elif method == 'cf':
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            B = self.model.B(T1, T2)
            P_T2 = self.model.discount_factor(T2)
            P_T1 = self.model.discount_factor(T1)
            sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T1)) / (2 * a)) * B
            h = (1 / sigma_p) * np.log(P_T2 * K_bond / P_T1) + 0.5 * sigma_p
            Caplet = (P_T1 * norm.cdf(-h + sigma_p) - K_bond * P_T2 * norm.cdf(-h))

        return N * Caplet
    

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
    
    
    def swap(self, Tau, N, K, payer = True, mc=False):
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
        payer : bool
            If True, value a payer swap; otherwise a receiver swap.
        mc : bool, optional
            If True, value via Monte Carlo (fwd measure); otherwise use closed form.

        Returns
        -------
        float
            Present value of the swap.
        """

        w = 1 if payer else -1
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

        Swap = N * w * (Floating_leg - Fixed_leg) 
        return Swap
    

    def swaption(self, Tau, N, K, payer = True, mc=False):
        """
        Value a European payer swaption.

        Parameters
        ----------
        Tau : list of float
            Payment times for the fixed leg (first entry is start time).
        N : float
            Notional amount.
        K : float
            Fixed rate.
        payer : bool
            If True, value a payer swaption; otherwise a receiver swaption.
        mc : bool, optional
            If True, value via Monte Carlo (fwd measure); otherwise use closed form.

        Returns
        -------
        float
            Present value of the swaption.
        """

        w = 1 if payer else -1
        T = Tau[0]  # Expiry
        S = Tau[-1] # Maturity

        if mc:
            P_N = self.curve_sim.zero_coupon_bond(T, S, fwd_measure=True)
            P_T = self.model.discount_factor(T)
            floating_leg = 1 - P_N
            fixed_leg = 0
            for i in range(1, len(Tau)):
                T1 = Tau[i - 1]
                T2 = Tau[i]
                Delta = T2 - T1
                P_i = self.curve_sim.zero_coupon_bond(T, T2, fwd_measure=True)
                fixed_leg += Delta * K * P_i 
            
            swaption = P_T * N * np.mean(np.maximum(w * (floating_leg - fixed_leg), 0))

        else:
            r_star = self._find_rstar(T, Tau, K)
            fixed_leg = 0
            for i in range(1, len(Tau)):
                T1 = Tau[i - 1]
                T2 = Tau[i]
                Delta = T2 - T1
                B = self.model.B(T, T2)
                A = self.model.A(T, T2)
                K_i = A * np.exp(-B * r_star)
                option = self.zero_bond_put(T, T2, K_i) if payer else self.zero_bond_call(T, T2, K_i)   
                fixed_leg += Delta * K * option
            
            # Floating leg: option on zero-coupon bond maturing at S
            B_N = self.model.B(T, S)
            A_N = self.model.A(T, S)
            K_N = A_N * np.exp(-B_N * r_star)
            floating_leg = self.zero_bond_put(T, S, K_N) if payer else self.zero_bond_call(T, S, K_N)
            
            swaption = N * (floating_leg + fixed_leg)

        return swaption
    

    def coupon_bond(self, Tau, C, N):
        """
        Value a coupon bond.

        Parameters
        ----------
        Tau : list of float
            Payment dates of the bond (T1, T2, ..., TN).
        C : float
            Coupon rate (annualized).
        N : float
            Notional (scaling factor).

        Returns
        -------
        float
            PV of the coupon bond.
        """

        bond_price = 0
        Delta = (Tau[-1] - Tau[0])

        for i in range(len(Tau)):
            P_T = self.curve.discount(Tau[i])
            cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
            bond_price += cashflow * P_T

        return bond_price
    

    def floating_rate_note(self, Tau, N):
        """
        Value a floating rate note (FRN).

        Parameters
        ----------
        Tau : list of float
            Payment dates of the bond (T1, T2, ..., TN).

        N : float
            Notional (scaling factor).

        Returns
        -------
        float
            PV of the floating rate note.
        """

        disc_cf = self.swap(Tau, N, K=0, payer=False, mc=False)
        disc_notional = N * self.model.discount_factor(Tau[-1])
        frn_price = disc_cf + disc_notional

        return frn_price


    def bond_option(self, T, Tau, C, K, N, call=True, mc=False):
        """
        European option on a coupon bond using simulation or analytical Jamshidian decomposition.

        Parameters
        ----------
        T : float
            Option expiry.
        Tau : list of float
            Payment dates of the bond (T1, T2, ..., TN).
        C : float
            Coupon rate (annualized). 
        K : float
            Strike price of the option (absolute price, not percentage).
        N : float
            Notional (scaling factor).
        call : bool
            True for a call, False for a put.
        mc : bool
            If True, Monte Carlo valuation under forward measure, else closed-form.

        Returns
        -------
        float
            PV of the European option on the coupon bond.
        """

        if mc:
            CB_t = self.curve_sim.coupon_bearing_bond(T, Tau, C, N)
            P_T = self.model.discount_factor(T)
            bond_option = P_T * np.mean((np.maximum(CB_t - K, 0) if call else np.maximum(K - CB_t, 0)))

        else:
            # Find critical short rate r* using Jamshidian decomposition for bond options
            r_star = self._find_rstar_bond(T, Tau, C, N, K)
            bond_option = 0
            Delta = (Tau[-1] - Tau[0]) 

            # Decompose into portfolio of zero-coupon bond options
            for i in range(len(Tau)):
                B = self.model.B(T, Tau[i])
                A = self.model.A(T, Tau[i])
                K_i = A * np.exp(-B * r_star)  # Strike for each zero-coupon bond
                
                # Value option on each zero-coupon bond
                option = self.zero_bond_call(T, Tau[i], K_i) if call else self.zero_bond_put(T, Tau[i], K_i)
                
                # Apply correct cashflow: C is annualized rate, multiply by Delta
                cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
                bond_option += cashflow * option

        return bond_option


    # --- Helper methods for Jamshidian decomposition --- #

    def _jamshidian_root(self, T, Tau, K, r_star):
        """
        Jamshidian root-finding function for swaption pricing.

        Parameters
        ----------
        T : float
            Option expiry time.
        Tau : list of float
            Payment times for the swap.
        K : float
            Fixed rate.
        r_star : float
            Short rate candidate.

        Returns
        -------
        float
            Root equation value.
        """
        root = 0
        for i in range(1, len(Tau)):
            T1 = Tau[i - 1]
            T2 = Tau[i]
            Delta = T2 - T1
            B = self.model.B(T, T2)
            A = self.model.A(T, T2)
            P_i = A * np.exp(-B * r_star)
            root += Delta * K * P_i

        root = root - (1 - P_i)
        return root


    def _find_rstar(self, T, Tau, K, x_min=-3, x_max=3):
        """
        Find critical short rate r* using Brent's method.

        Parameters
        ----------
        T : float
            Option expiry time.
        Tau : list of float
            Payment times for the swap.
        K : float
            Fixed rate.
        x_min : float, optional
            Lower bound for root search.
        x_max : float, optional
            Upper bound for root search.

        Returns
        -------
        float
            Critical short rate r*.
        """
        f = lambda r: self._jamshidian_root(T, Tau, K, r)
        r_star = brentq(f, x_min, x_max, xtol=1e-12)
        return r_star


    def _jamshidian_root_bond(self, T, Tau, C, N, K_strike, r_star):
        """
        Jamshidian root-finding function for bond option pricing.
        Solves: sum(cashflow_i * P(T, T_i; r*)) = K_strike

        Parameters
        ----------
        T : float
            Option expiry time.
        Tau : list of float
            Coupon payment dates.
        C : float
            Coupon rate (annualized).
        N : float
            Notional amount.
        K_strike : float
            Strike price of the bond option.
        r_star : float
            Short rate candidate.

        Returns
        -------
        float
            Root equation value (bond price - strike).
        """
        bond_price = 0
        Delta = (Tau[-1] - Tau[0])
        
        for i in range(len(Tau)):
            B = self.model.B(T, Tau[i])
            A = self.model.A(T, Tau[i])
            P_i = A * np.exp(-B * r_star)
            cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
            bond_price += cashflow * P_i
        
        return bond_price - K_strike


    def _find_rstar_bond(self, T, Tau, C, N, K_strike, x_min=-3, x_max=3):
        """
        Find critical short rate r* for bond option using Brent's method.

        Parameters
        ----------
        T : float
            Option expiry time.
        Tau : list of float
            Coupon payment dates.
        C : float
            Coupon rate (annualized).
        N : float
            Notional amount.
        K_strike : float
            Strike price of the bond option.
        x_min : float, optional
            Lower bound for root search.
        x_max : float, optional
            Upper bound for root search.

        Returns
        -------
        float
            Critical short rate r*.
        """
        f = lambda r: self._jamshidian_root_bond(T, Tau, C, N, K_strike, r)
        r_star = brentq(f, x_min, x_max, xtol=1e-12)
        return r_star


        



 
