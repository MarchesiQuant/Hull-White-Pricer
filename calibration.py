from scipy.optimize import minimize
from numpy import sqrt


class HullWhiteCalibrator:
    """
    Calibrates the Hull–White one-factor model parameter `sigma` (volatility)
    with a fixed mean reversion speed `a`.
    """

    def __init__(self, pricer, market_prices, calibrate_to='Caplets', a_fixed=0.01):
        """
        Initializes the Hull–White calibrator fixing 'a' and calibrating only 'sigma'.

        Parameters
        ----------
        pricer : HullWhitePricer
            Pricer instance capable of pricing the derivatives in the market dataset.
        market_prices : dict
            Dictionary containing market data.
        calibrate_to : str, optional
            Type of instruments to calibrate to ('Caplets' or 'Swaptions').
        a_fixed : float, optional
            Fixed mean reversion parameter. Default is 0.01.
        """

        self.pricer = pricer
        self.model = pricer.model
        self.market_prices = market_prices
        self.calibrate_to = calibrate_to
        self.a_fixed = a_fixed
        self.history = []

        # Set fixed a
        self.model.parameters['a'] = self.a_fixed


    def objective(self, sigma):
        """
        Objective function for calibration: relative squared error
        between model and market prices, fixing 'a' and varying 'sigma'.

        Parameters
        ----------
        sigma : float
            Volatility parameter to test.

        Returns
        -------
        float
            The sum of relative squared errors.
        """
        sigma = sigma[0]  # optimizer passes as array
        self.model.parameters['sigma'] = sigma
        self.model.parameters['a'] = self.a_fixed

        error = 0.0
        n = len(self.market_prices['Prices'])
        for i in range(n):
            market_price = self.market_prices['Prices'][i]
            K = self.market_prices['Strike'][i] / 100
            N = self.market_prices['Notional'][i]

            if self.calibrate_to == 'Caplets':
                T = self.market_prices['Expiry'][i]
                S = self.market_prices['Maturity'][i]
                model_price = self.pricer.caplet(T, S, N, K)

            elif self.calibrate_to == 'Swaptions':
                Tau = self.market_prices['Dates'][i]
                DF = self.pricer.curve.discount(Tau[0])
                model_price = self.pricer.swaption(Tau, N, K)/DF # Forward Premium 

            else:
                raise ValueError("Calibracion only implemented for 'Caplets' and 'Swaptions'.")

            error += (1/n) * ((model_price - market_price)**2) / (market_price**2 + 1e-6)

        self.history.append((sigma, sqrt(error)))
        return sqrt(error)


    def callback(self, sigma):
        """
        Print current sigma and error during optimization.
        """
        sigma = sigma[0]
        if self.history:
            _, err = self.history[-1]
            print(f"a fixed: {self.a_fixed:.6f}, sigma: {sigma:.6f}, RMSRE: {err:.5e}")


    def calibrate(self, init_sigma=0.03, bounds=[(1e-4, 0.75)], method='L-BFGS-B'):
        """
        Run optimization to calibrate only sigma.

        Parameters
        ----------
        init_sigma : float, optional
            Initial guess for sigma.
        bounds : list[tuple[float, float]], optional
            Bounds for sigma.
        method : str, optional
            Optimization method.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of the optimization.
        """

        result = minimize(self.objective, [init_sigma], bounds=bounds, method=method, callback=self.callback)

        if result.success:
            sigma_opt = result.x[0]
            self.model.parameters['sigma'] = sigma_opt
            self.model.parameters['a'] = self.a_fixed

            print(f"\nCalibration successful:")
            print(f"Iterations: {result.nit}")
            print(f"Number of instruments: {len(self.market_prices['Prices'])}")
            print(f"Total Error: {result.fun:>+8.3%}\n")
            print("Parameters:") 
            print(f"Fixed a: {self.a_fixed:.5f}")
            print(f"Optimal sigma: {sigma_opt:.5f}\n")

            for i in range(len(self.market_prices['Prices'])):
                market_price = self.market_prices['Prices'][i]
                K = self.market_prices['Strike'][i] / 100
                N = self.market_prices['Notional'][i]

                if self.calibrate_to == 'Caplets':
                    T = self.market_prices['Expiry'][i]
                    S = self.market_prices['Maturity'][i]
                    model_price = self.pricer.caplet(T, S, N, K)
                    dif = model_price / market_price - 1
                    print(f"Caplet {i:>2}: {T:>5.2f}Y to {S:<5.2f}Y | Model: {model_price:>8.2f} | Market: {market_price:>8.2f} | Diff: {dif:>+8.3%}")

                elif self.calibrate_to == 'Swaptions':
                    Tau = self.market_prices['Dates'][i]
                    DF = self.pricer.curve.discount(Tau[0])
                    model_price = self.pricer.swaption(Tau, N, K)/DF # Forward Premium 
                    dif = model_price / market_price - 1
                    print(f"Swaption {i:>3}: {Tau[0]:>5.2f}Y to {Tau[-1]:<5.2f}Y | Model: {model_price:>8.2f} | Market: {market_price:>8.2f} | Diff: {dif:>+8.3%}")

        else:
            print("Calibration failed:", result.message)

        return result
