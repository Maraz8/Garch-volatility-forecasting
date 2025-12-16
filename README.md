# Volatility Forecasting and Risk Backtesting Framework  
**Author: Filippo Tiberi — BSc Finance, Bocconi University**

This project provides an end-to-end framework for volatility modelling, forecasting and Value-at-Risk (VaR) backtesting.  
It extends the original architecture with additional models and risk-testing functionality to support quantitative research applications.

---

## Overview

This repository implements and compares multiple approaches for modelling and forecasting financial market volatility:

### **Econometric Models**
- **GARCH(1,1)**  
- **EGARCH**  
- **Heston–Nandi GARCH (HNG)** — *added in this extended version*  
  - full Python implementation with MLE estimation  
  - conditional variance reconstruction  
  - multi-step volatility forecasting  

### **Deep Learning Models**
- **LSTM-based volatility model** — *added in this extended version*  
  - custom-built sequence model using Keras  
  - supervised windowed dataset construction  
  - in-sample volatility estimation  
  - recursive multi-step forecasting  

---

## Value-at-Risk (VaR) Engine

A modular VaR engine is included for risk assessment:

- **Parametric Normal VaR**
- Support for **long** and **short** positions  
- Fully vectorized implementation for speed  
- Automatic alignment between returns and forecasted volatility  

### Features implemented:
- Compute VaR at any confidence level (e.g., 95%, 99%)
- Compute violation series (0/1 process)
- Integrates directly with all volatility models

---

## Statistical Backtests (Added in this version)

### **Kupiec POF Test**
Evaluates whether the empirical violation frequency matches the theoretical VaR confidence level.  
Returns:
- LR statistic  
- p-value  
- total violations  
- sample size  

### **Christoffersen Conditional Coverage Test**
Tests both:
- unconditional coverage  
- independence of violations  
Combines:
- Kupiec POF  
- Christoffersen independence likelihood ratio  

Results include:
- LR_ind  
- LR_pof  
- LR_cc  
- p-value for full conditional coverage  

---

