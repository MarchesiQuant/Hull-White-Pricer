from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import numpy as np


class Curve:
    """
    Class to handle market curves, including discount factors and
    instantaneous forward rates, with cubic interpolation and spline smoothing.
    """

    def __init__(self, time, discount_factors, smooth=1e-7):
        """
        Initializes a market curve object containing the discount curve
        and the corresponding instantaneous forward curve.

        Parameters
        ----------
        time : array_like
            Time to maturity (in years) of the curve nodes.
        discount_factors : array_like
            Discount factors at the given maturities.
        smooth : float, optional
            Smoothing parameter for the forward spline interpolation (default: 1e-7).
        """
        self.time = np.array(time)
        self.discount_factors = np.array(discount_factors)
        self.smooth = smooth

        self._build_interpolators()

    def _build_interpolators(self):
        """
        Builds both the discount factor interpolation function and
        the instantaneous forward rate spline in a single step.
        """
        # Cubic interpolation for discount factors
        self.discount_func = interp1d(
            self.time,
            self.discount_factors,
            kind='cubic',
            fill_value="extrapolate",
            bounds_error=False
        )

        # Spline for the log of discount factors (needed for forward rate derivation)
        lnP = np.log(self.discount_factors)
        self.forward_spline = UnivariateSpline(self.time, lnP, s=self.smooth)

    def discount(self, t):
        """
        Returns the interpolated discount factor P(0, t).

        Parameters
        ----------
        t : float or array_like
            Time(s) to maturity.

        Returns
        -------
        float or np.ndarray
            Discount factor(s) corresponding to t.
        """
        return self.discount_func(t)

    def inst_forward_rate(self, t):
        """
        Returns the interpolated instantaneous forward rate f(0, t).

        Parameters
        ----------
        t : float or array_like
            Time(s) to maturity.

        Returns
        -------
        float or np.ndarray
            Instantaneous forward rate(s) corresponding to t.
        """
        t = np.array(t)
        return -self.forward_spline.derivative(1)(t)
    

    def forward_rate(self, T1, T2):
        """
        Computes the simple forward rate F(0; T1, T2) implied by the discount curve.

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
        
        P1 = self.discount(T1)
        P2 = self.discount(T2)

        return (P1 / P2 - 1.0) / (T2 - T1)
