import pandas as pd 
import numpy as np
from scipy.stats import norm


# Clase HullWhite (sin simulacion, solo funciones)
class HullWhiteModel:
    def __init__(self, curve, parameters):
        """
        Inicializa el modelo Hull-White con una curva de descuento.

        Parameters:
        -----------
        curve : Curve
            Instancia de la clase Curve que representa la curva de descuento.

        parameters : dict
            Parámetros del modelo Hull-White, como la velocidad de reversión y la volatilidad
        """
        self.curve = curve
        self.parameters = parameters
        self.a = parameters['a']            # Velocidad de reversión a la media
        self.sigma = parameters['sigma']    # Volatilidad del modelo  
        self.r0 = parameters['r0']          # Tasa de interés inicial
        

    def forward_rate(self, t):
        """Calcula la tasa forward instantánea en el tiempo t."""
        return self.curve.forward(t)
    
    def discount_factor(self, t):
        """Calcula el factor de descuento en el tiempo t."""
        return self.curve.discount(t)
    
    def _alpha(self, t):
        """Calcula la función alpha del modelo Hull-White."""
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        fwd = self.forward_rate(t)
        return fwd + (sigma**2)/(2*a**2) * (1- np.exp(-a * t))**2
    
    def _B(self, t, T ):
        """Calcula la función B del modelo Hull-White."""
        a = self.parameters['a']
        return (1 - np.exp(-a * (T - t))) / a
    
    
    def _A(self, t, T):
        """Calcula la función A del modelo Hull-White."""
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        P_t = self.discount_factor(t)
        P_T = self.discount_factor(T)
        fwd = self.forward_rate(t)
        B = self._B(t, T)
        return (P_T/P_t)*np.exp(B*fwd - (sigma**2 / (4 * a)) * (1 - np.exp(-2*a*t)) * B**2)


    def short_rate(self, t, z = None):
        """
        Calcula la tasa de interés a corto plazo en el tiempo t.

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).
        Returns:
        --------
        float
            Tasa de interés a corto plazo.
        """
        if z is None:
            z = np.random.normal()

        r0 = self.parameters['r0']
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        E = r0 * np.exp(-a * t) + self._alpha(t) - np.exp(-a * t)*self._alpha(0)
        return E + np.sqrt(V) * z
    

    def short_rate_forward(self, t, T, z = None):
        """
        Calcula la tasa de interés a corto plazo bajo la medida forward.

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).
        T : float
            Vencimiento del bono cupon cero que se usa como numerario (en años).
        z : float, optional

        Returns:
        --------
        float
            Tasa de interés a corto plazo bajo la medida forward.
        """

        if z is None:
            z = np.random.normal()

        a = self.parameters['a']
        sigma = self.parameters['sigma']
        alpha_t = self._alpha(t)
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        M = (sigma/a)**2 * (1 - np.exp(-a * t) - np.exp(-a*T)*np.sinh(a*t))
        E = - M + alpha_t
        return E + np.sqrt(V) * z
    


# Simulador de Hull-White exacto, por Euler y bajo la medida forward neutra 
class HullWhiteSimulation:
    def __init__(self, model: HullWhiteModel, n_paths=10**5, n_steps=100, seed = 2025):
        """
        Simulador de trayectorias para el modelo Hull-White.

        Parameters:
        -----------
        model : HullWhite
            Instancia del modelo Hull-White.
        n_paths : int
            Número de trayectorias de Monte Carlo.
        n_steps : int
            Número de pasos de tiempo para Euler.
        """
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = np.random.seed(seed)


    def simulate_short_rate_direct(self, T):
        """
        Simula r(T) usando la distribución cerrada de Hull-White bajo la medida cuenta bancaria.
        No genera trayectorias completas.

        Parameters:
        -----------
        T : float
            Horizonte de simulación en años.

        Returns:
        --------
        r : ndarray
            Vector de tasas simuladas de tamaño (n_paths,).
        """
        z = np.random.normal(size=self.n_paths)
        r = np.array([self.model.short_rate(T, z=z_i) for z_i in z])
        return r
    
    # Algo falla aqui
    def simulate_short_rate_direct_forward(self, t, T):
        """
        Simula r(T) usando la distribución cerrada de Hull-White bajo la medida forward neutra.
        No genera trayectorias completas.

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).
        T : float
            Vencimiento del bono cupon cero que se usa como numerario (en años).

        Returns:
        --------
        r : ndarray
            Vector de tasas simuladas de tamaño (n_paths,).
        """
        z = np.random.normal(size=self.n_paths)
        r = np.array([self.model.short_rate_forward(t, T, z=z_i) for z_i in z])
        return r


    def simulate_short_rate_euler(self, T):
        """
        Simula r(t) usando el método de Euler para trayectorias completas bajo la medida cuenta bancaria.
        Parameters:
        ----------- 
        T : float
            Horizonte de simulación en años.
        Returns:
        --------
        r : ndarray
            Matriz de trayectorias de tasas de interés de tamaño (n_paths, n_steps + 1).
        times : ndarray 
            Vector de tiempos correspondientes a las trayectorias.
        """

        dt = T / self.n_steps
        times = np.linspace(0, T, self.n_steps + 1)
        x = np.zeros((self.n_paths, self.n_steps + 1))
        r = np.zeros_like(x)
        x[:, 0] = self.model.r0 - self.model._alpha(0)
        r[:, 0] = self.model.r0

        for i in range(1, self.n_steps + 1):
            z = np.random.normal(size=self.n_paths)
            x[:, i] = x[:, i-1] - self.model.a * x[:, i-1] * dt + self.model.sigma * np.sqrt(dt) * z
            r[:, i] = x[:, i] + self.model._alpha(times[i])

        return r, times
    

    def validate_simulation(self, T):
        """Compara media y desviación simuladas (Euler y directa) con valores analíticos."""

        r_euler, _ = self.simulate_short_rate_euler(T)
        r_euler_end = r_euler[:, -1]
        r_direct = self.simulate_short_rate_direct(T)

        analytic_mean = self.model.r0 * np.exp(-self.model.a * T) + self.model._alpha(T) - np.exp(-self.model.a * T) * self.model._alpha(0)
        analytic_std = np.sqrt((self.model.sigma**2) / (2 * self.model.a) * (1 - np.exp(-2 * self.model.a * T)))

        data = {"Mean": [np.mean(r_euler_end),np.mean(r_direct),analytic_mean],"Std Dev":[np.std(r_euler_end),np.std(r_direct), analytic_std]}

        df = pd.DataFrame(data, index=["Euler Simulation", "Direct Simulation", "Analytic"])
        return df


# Clase HullWhiteCurveBuilder para calcular bonos, tipos a plazo, factores de descuento y tipos forward
class HullWhiteCurveBuilder:
    def __init__(self, model, simulation):
        self.model = model
        self.sim = simulation
    

    def zero_coupon_bond(self, t, T, fwd_measure=False):

        """        
        Calcula el precio de un bono cupón cero entre t y T por analiticamente

        Parameters:
        ----------- 
        t : float
            Tiempo actual (en años).
        T : float
            Tiempo de vencimiento del bono (en años).
        fwd_measure : bool, optional
            Si es True, calcula el precio bajo la medida forward neutra (default False).

        Returns:

        --------
        float       
            Precio del bono cupón cero entre t y T.

        """  
        # Fórmula analítica: P(t,T) = A(t,T) * exp(-B(t,T)*r(t))
        if fwd_measure:
            r_t = self.sim.simulate_short_rate_direct_forward(t, T)

        else:
            r_t = self.sim.simulate_short_rate_direct(t)
            
        A = self.model._A(t, T)
        B = self.model._B(t, T)
        price = A * np.exp(-B * r_t)
        return price
    
        
    # El factor de descuento es path-dependent, luego su calculo es bastante tedioso
    def discount_factor(self, t, T):
        """
        Calcula el factor de descuento entre t y T por montecarlo.

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).

        T : float
            Tiempo futuro (en años).

        Returns:
        --------
        float
            Factor de descuento entre t y T.
        """

        # Simular short rate path desde t hasta T
        r_paths, times = self.sim.simulate_short_rate_euler(T)

        # Índice final para T en el vector times
        idx_T = np.searchsorted(times, T)
        idx_t = np.searchsorted(times, t)

        # Calcular integral numérica del short rate en cada path entre t y T
        dt = times[1] - times[0]
        integral_r = np.sum(r_paths[:, idx_t:idx_T] * dt, axis=1)

        # Factor de descuento estocastico
        df = np.exp(-integral_r)

        return df


    def inst_forward_rate(self, t, T):
        """
        Calcula el tipo forward instantáneo F(t,T) usando la relación con los bonos cupón cero.

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).
        T : float
            Tiempo futuro para el forward (en años).

        Returns:
        --------
        float
            Tipo forward instantáneo F(t,T).
        """
        r_t = self.sim.simulate_short_rate_direct(t)
        fwd_T= self.model.forward_rate(T)
        fwd_t = self.model.forward_rate(t)
        B = self.model._B(t, T)
        a = self.model.parameters['a']
        sigma = self.model.parameters['sigma']
        K = (sigma**2)*(1-np.exp(-2*a*t))/(2*a)
        F = fwd_T + np.exp(-a*(T-t))*(r_t - fwd_t + K*B)
        return F
        
        
        
    def long_rate(self, t, T):
        """
        Calcula el tipo de interés a plazo R(t,T).

        Parameters:
        -----------
        t : float
            Tiempo actual (en años).
        T : float
            Tiempo futuro para el tipo a largo plazo (en años).

        Returns:
        --------
        float
            Tipo a plazo R(t,T).
        """

        A = self.model._A(t, T)
        B = self.model._B(t, T)
        r_t = self.sim.simulate_short_rate_direct(t)
        alpha = - np.log(A) / (T - t)
        beta = B/(T-t)
        R = alpha + beta * r_t
        return R
    


# Clase para valorar opciones sobre bonos, caps, floors, swaps, swaptions y exoticos
class HullWhitePricer:
    def __init__(self, model, curve_sim):

        self.model = model
        self.curve_sim = curve_sim


    def zero_bond_put(self, T, S, K, mc = False):
        """
        Valora una opción put sobre un bono cupón cero.

        Parameters:
        -----------
        T : float
            Tiempo de vencimiento de la opcion (en años).
        S : float
            Tiempo de vencimiento del bono subyacente (en años). S > T.
        K : float
            Precio de ejercicio de la opción put.

        Returns:
        --------
        float
            Valor de la opción put.
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
            sigma_p = sigma * np.sqrt((1 - np.exp(-2*a*T)) / (2 * a))*B 
            h = (1/sigma_p)*np.log(P_S / (K * P_T)) + 0.5 * sigma_p
            V0 = K * P_T * norm.cdf(-h + sigma_p) - P_S * norm.cdf(-h)
        
        return V0
    
    def zero_bond_call(self, T, S, K, mc = False):
        """
        Valora una opción call sobre un bono cupón cero.

        Parameters:
        -----------
        T : float
            Tiempo de vencimiento de la opcion (en años).
        S : float
            Tiempo de vencimiento del bono subyacente (en años). S > T.
        K : float
            Precio de ejercicio de la opción call.

        Returns:
        --------
        float
            Valor de la opción call.
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
            sigma_p = sigma * np.sqrt((1 - np.exp(-2*a*T)) / (2 * a))*B 
            h = (1/sigma_p)*np.log(P_S / (K * P_T)) + 0.5 * sigma_p
            V0 = P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sigma_p) 
        
        return V0
    

    def cap(self, Tau, N, K, mc = False):
        """
        Valora un cap sobre un bono cupón cero.

        Parameters:
        -----------
        Tau : list
            Lista de tiempos de pagos de los caplets (la primera no paga, es de fixing)
        N : int
            Nominal del cap.
        K : float
            Strike del cap.

        Returns:
        --------
        float
            Valor del cap.
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
                K_bond = (1 + K * Delta)
                put_price = self.zero_bond_put(t_prev, t_curr, 1/K_bond)
                Cap += K_bond * put_price

        return N * Cap


    def floor(self, Tau, N, K, mc=False):
        """
        Valora un floor sobre un bono cupón cero.

        Parameters:
        -----------
        Tau : list
            Lista de tiempos de pagos de los floorlets (la primera no paga, es de fixing)
        N : int
            Nominal del floor.
        K : float
            Strike del floor.

        Returns:
        --------
        float
            Valor del floor.
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
                call_price = self.zero_bond_call(t_prev, t_curr, 1/K_bond)
                Floor += K_bond * call_price

        return N * Floor
