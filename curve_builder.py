from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import numpy as np


# Clase para manejar curvas de mercado (calculo de la curva fwd instantanea e interpolacion)
class Curve:
    def __init__(self, time, discount_factors, smooth=1e-7):
        """
        Clase para manejar curva de descuento y curva forward instantánea.

        Parameters:
        -----------
        time : array_like
            Tiempos (en años) de los nodos de la curva.
        discount_factors : array_like
            Factores de descuento en los tiempos indicados.
        smooth : float, optional
            Parámetro de suavizado para el spline (default 1e-7).
        """
        self.time = np.array(time)
        self.discount_factors = np.array(discount_factors)
        self.smooth = smooth

        self._build_interpolators()

    def _build_interpolators(self):
        """Construye las funciones de descuento y forward en un solo paso."""
        
        # Interpolador de factores de descuento
        self.discount_func = interp1d(
            self.time,
            self.discount_factors,
            kind='cubic',
            fill_value="extrapolate",
            bounds_error=False
        )
        # Interpolador spline para forwards (derivada del logaritmo de P)
        lnP = np.log(self.discount_factors)
        self.forward_spline = UnivariateSpline(self.time, lnP, s=self.smooth)

    def discount(self, t):
        """Devuelve el factor de descuento interpolado."""
        return self.discount_func(t)

    def forward(self, t):
        """Devuelve el forward instantáneo interpolado."""
        t = np.array(t)
        return -self.forward_spline.derivative(1)(t)