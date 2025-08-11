# Hull-White Interest Rate Model Library

This repository provides a set of Python modules to build discount curves, implement the Hull-White short rate model, simulate interest rate paths, price interest rate derivatives, and calibrate the model to market data. It is designed for quantitative finance practitioners and researchers interested in interest rate modeling and derivative valuation.

---

## Modules

### 1. `curve_builder.py`

- Implements the `Curve` class to manage market discount curves.
- Uses cubic interpolation and smoothing splines to construct discount factors and instantaneous forward rates.
- Provides methods to obtain discount factors and forward rates for arbitrary maturities.

### 2. `hw_model.py`

- Implements the Hull-White one-factor short rate model classes:
  - `HullWhiteModel`: encapsulates the Hull-White model dynamics and analytic functions.
  - `HullWhiteSimulation`: Monte Carlo simulator for short rate paths via exact and Euler schemes under both risk-neutral and forward measures.
  - `HullWhiteCurveBuilder`: computes zero-coupon bond prices, discount factors, forward rates, and long rates based on Hull-White simulations.
  - `HullWhitePricer`: prices common interest rate derivatives such as zero-coupon bond options (calls and puts), caps, and floors with both analytic formulas and Monte Carlo methods.

### 3. `calibration.py`

- Implements the `HullWhiteCalibrator` class to calibrate the Hull-White model parameters (`a` and `sigma`) to market prices of caps or other interest rate instruments.
- Uses nonlinear least squares optimization (`scipy.optimize.minimize`) to minimize pricing errors.

---

## Features

- Smooth interpolation of market discount curves.
- Analytical and simulation-based methods for interest rate dynamics.
- Monte Carlo path generation with large-scale simulation support.
- Pricing of vanilla interest rate options with flexibility for analytic or simulation approaches.
- Model calibration framework for real market data.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/hull-white-model.git
cd hull-white-model
