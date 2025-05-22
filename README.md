# Portfolio Optimization API

A sophisticated portfolio optimization API that combines traditional financial theory with modern machine learning techniques to generate optimal investment portfolios.

## Overview

This API provides a robust solution for portfolio optimization by combining:
- Modern Portfolio Theory (Markowitz)
- Conditional Value at Risk (CVaR) optimization
- Machine Learning-based return forecasting using Prophet
- Custom risk metrics that balance multiple risk factors

## Key Components

- `main.py`: FastAPI application with the main optimization endpoint
- `helpers.py`: Utility functions for risk calculations and return forecasting
- `Dockerfile`: Container configuration for easy deployment
- `requirements.txt`: Python dependencies

## API Endpoints

### Health Check
```bash
curl -X GET "http://localhost:8080/ping"
```
Response:
```json
{
    "status": "ok"
}
```

### Portfolio Optimization
```bash
curl -X POST "http://localhost:8080/optimize-portfolio" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@returns.csv" \
  -F "risk_level=0.5" \
  -F "max_weight=0.3"
```

Expected Response:
```json
{
    "optimal_portfolio": {
        "AAPL": 0.25,
        "MSFT": 0.30,
        "GOOGL": 0.20,
        "AMZN": 0.25
    }
}
```

## Optimization Methodology

Our portfolio optimization approach combines several sophisticated techniques:

1. **Return Forecasting**
   - Uses Prophet (or AutoARIMA alternative) for time series forecasting
   - Generates expected returns for each asset
   - Incorporates trend and seasonality patterns

2. **Risk Metrics**
   - Traditional volatility (Markowitz)
   - Conditional Value at Risk (CVaR)
   - Custom balanced risk metric that combines:
     - Scaled volatility
     - Scaled CVaR
     - Portfolio diversification risk

3. **Optimization Process**
   - Constrained optimization using SLSQP
   - Risk level targeting
   - Maximum weight constraints
   - Full investment constraint (weights sum to 1)

## Input Requirements

- CSV file with historical returns
- Date index
- Asset returns in columns
- Minimum 100 data points
- No missing values (Just allowed NaN in first row)

## Deployment

The API is containerized using Docker and can be deployed using:

```bash
docker build -t portfolio-optimizer .
docker run -p 8080:8080 portfolio-optimizer
```

## Dependencies

- FastAPI
- Pandas
- NumPy
- SciPy
- Prophet
- Uvicorn

## Notes

- The optimization process requires at least 100 data points for reliable statistical analysis
- Risk level should be between 0 and 1
- Maximum weight per asset should be between 0 and 1

## 🚫 Commercial Use Restriction

This repository is released for **educational and personal use only**.  
**Commercial or business use of this code, in whole or in part, is strictly prohibited without prior written permission from the author.**

If you are interested in using this code in a commercial context, please contact the author to request a proper license.
