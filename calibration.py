from scipy.optimize import minimize


# Calibrador a mercado para sigma y a 
class HullWhiteCalibrator:
    def __init__(self, model, pricer, market_prices):
        self.model = model
        self.pricer = pricer
        self.market_prices = market_prices
        self.history = []

    def objective(self, params):
        a, sigma = params
        self.model.parameters['a'] = a
        self.model.parameters['sigma'] = sigma

        error = 0
        for i in range(len(self.market_prices['Price'])):
            market_price = self.market_prices['Price'][i]
            K = self.market_prices['Strike'][i]
            Tau = self.market_prices['Dates'][i]
            N = self.market_prices['Notional'][i]
            model_price = self.pricer.cap(Tau, N, K)
            error += ((model_price - market_price) ** 2) / (market_price ** 2 + 1e-6)

        self.history.append((a, sigma, error))
        return error

    def callback(self, params):
        a, sigma = params
        if self.history:
            _, _, err = self.history[-1]
            print(f"a: {a:.5f}, sigma: {sigma:.5f}, Error: {err:.5e}")

    def calibrate(self, init_params=(0.01, 0.01), bounds=[(1e-4, 0.5), (1e-4, 0.5)], method='Powell'):
        result = minimize(self.objective, init_params, bounds=bounds, method=method, callback=self.callback)
        if result.success:
            a_opt, sigma_opt = result.x
            self.model.parameters['a'] = a_opt
            self.model.parameters['sigma'] = sigma_opt
            print(" ")

            for i in range(len(self.market_prices['Price'])):
                market_price = self.market_prices['Price'][i]
                K = self.market_prices['Strike'][i]
                Tau = self.market_prices['Dates'][i]
                N = self.market_prices['Notional'][i]
                model_price = self.pricer.cap(Tau, N, K)
                dif = model_price/market_price - 1
                print(f"Diferencias Cap {int(max(Tau))}y: {100*dif: .4f}%")
        else:
            print("Calibración fallida:", result.message)
        return result