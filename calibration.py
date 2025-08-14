from scipy.optimize import minimize


class HullWhiteCalibrator:
    """
    Calibrates the Hull–White one-factor model parameters `a` (mean reversion speed)
    and `sigma` (volatility) to market prices of interest rate derivatives.

    This class adjusts the model so that its prices match market-observed prices
    as closely as possible, minimizing a relative squared error metric.
    """

    def __init__(self, model, pricer, market_prices):
        """
        Initializes the Hull–White calibrator.

        Parameters
        ----------
        model : HullWhiteModel
            The Hull–White model instance to be calibrated.
        pricer : HullWhitePricer
            Pricer instance capable of pricing the derivatives in the market dataset.
        market_prices : dict
            Dictionary containing market data with keys:
            - 'Price' (list[float]): Observed market prices.
            - 'Strike' (list[float]): Strike rates.
            - 'Dates' (list[list[float]]): List of time grids (e.g., caplet maturities).
            - 'Notional' (list[float]): Notional amounts for each instrument.
        """
        self.model = model
        self.pricer = pricer
        self.market_prices = market_prices
        self.history = []

    def objective(self, params):
        """
        Objective function for calibration: computes the relative squared error
        between model and market prices for a given (a, sigma) pair.

        Parameters
        ----------
        params : tuple[float, float]
            The parameters to test: (a, sigma).

        Returns
        -------
        float
            The sum of relative squared errors across all instruments.
        """
        a, sigma = params
        self.model.parameters['a'] = a
        self.model.parameters['sigma'] = sigma

        error = 0.0
        for i in range(len(self.market_prices['Price'])):
            market_price = self.market_prices['Price'][i]
            K = self.market_prices['Strike'][i]
            Tau = self.market_prices['Dates'][i]
            N = self.market_prices['Notional'][i]

            model_price = self.pricer.cap(Tau, N, K)

            # Relative squared error, with small epsilon to avoid division by zero
            error += ((model_price - market_price) ** 2) / (market_price ** 2 + 1e-6)

        self.history.append((a, sigma, error))
        return error

    def callback(self, params):
        """
        Callback function for the optimizer: prints current parameters and error.

        Parameters
        ----------
        params : tuple[float, float]
            Current (a, sigma) being tested by the optimizer.
        """
        a, sigma = params
        if self.history:
            _, _, err = self.history[-1]
            print(f"a: {a:.5f}, sigma: {sigma:.5f}, Error: {err:.5e}")

    def calibrate(self, init_params=(0.01, 0.01), bounds=[(1e-4, 0.5), (1e-4, 0.5)], method='Powell'):
        """
        Runs the optimization procedure to calibrate (a, sigma).

        Parameters
        ----------
        init_params : tuple[float, float], optional
            Initial guess for (a, sigma). Default is (0.01, 0.01).
        bounds : list[tuple[float, float]], optional
            Bounds for (a, sigma). Default is [(1e-4, 0.5), (1e-4, 0.5)].
        method : str, optional
            Optimization method to use. Default is 'Powell'.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result of the optimization process.
        """
        result = minimize(
            self.objective,
            init_params,
            bounds=bounds,
            method=method,
            callback=self.callback
        )

        if result.success:
            a_opt, sigma_opt = result.x
            self.model.parameters['a'] = a_opt
            self.model.parameters['sigma'] = sigma_opt
            print(" ")

            # Print percentage differences between calibrated model prices and market prices
            for i in range(len(self.market_prices['Price'])):
                market_price = self.market_prices['Price'][i]
                K = self.market_prices['Strike'][i]
                Tau = self.market_prices['Dates'][i]
                N = self.market_prices['Notional'][i]

                model_price = self.pricer.cap(Tau, N, K)
                dif = model_price / market_price - 1
                print(f"Cap {int(max(Tau))}y difference: {100 * dif: .4f}%")
        else:
            print("Calibration failed:", result.message)

        return result
