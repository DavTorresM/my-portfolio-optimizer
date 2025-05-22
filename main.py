from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, confloat, field_validator
from typing import Dict
import pandas as pd
import uvicorn
import logging
import io

import numpy as np



from helpers import *
import numpy as np
from scipy.optimize import minimize

#import cmdstanpy
#cmdstanpy.install_cmdstan()

# Configure logging for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Optimizer API")


# Pydantic model for validating form fields
class PortfolioConstraints(BaseModel):
    risk_level: confloat(ge=0.0, le=1.0)
    max_weight: confloat(gt=0.0, le=1.0)

# Pydantic model for validating output
class PortfolioWeights(BaseModel):
    weights: Dict[str, confloat(ge=0.0, le=1.0)]

    @field_validator('weights')
    @classmethod
    def validate_sum(cls, v):
        total = sum(v.values())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"Sum of weights must be exactly 1.0, got {total}")
        return v

@app.get("/ping")
def ping():
    """
    Health check endpoint.
    """
    logger.info("Ping received")
    return {"status": "ok"}

@app.post("/optimize-portfolio")
async def optimize_portfolio(
    file: UploadFile = File(...),
    risk_level: float = Form(...),
    max_weight: float = Form(...)
):
    """
    Receives a CSV file with asset returns and optimization constraints,
    then returns an optimal portfolio respecting the constraints.
    """
    logger.info("Received request to optimize portfolio")

    # Validate numeric fields using Pydantic
    try:
        constraints = PortfolioConstraints(risk_level=risk_level, max_weight=max_weight)
    except Exception as e:
        logger.error(f"Invalid constraints: {e}")
        raise HTTPException(status_code=400, detail="risk_level must be between 0 and 1, and max_weight greater than 0 and <= 1.")

    # Validate and parse the CSV file
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), index_col=0, parse_dates=True)
        # Replace NaN in the first row with 0
        df.iloc[0] = df.iloc[0].fillna(0)
        # Sort DataFrame by date (from oldest to most recent)
        df = df.sort_index()

        # Create mapping dictionary
        col_names = df.columns
        letters = generate_column_labels(len(col_names))
        dict_letters = dict(zip(col_names, letters))
        # Rename DataFrame columns
        df = df.rename(columns=dict_letters)
        
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV file format")

    if df.empty:
        logger.warning("Uploaded CSV is empty")
        raise HTTPException(status_code=400, detail="CSV file is empty")
    
    if len(df)<100:
        logger.warning("The dataset is too small for the statistical treatment used.")
        raise HTTPException(status_code=400, detail="The dataset is too small for the statistical treatment used.")

    if not all(isinstance(col, str) for col in df.columns):
        logger.warning("CSV headers are not all strings (tickers)")
        raise HTTPException(status_code=400, detail="All column headers must be string tickers")

    if df.isnull().values.any():
        logger.warning("CSV contains missing values")
        raise HTTPException(status_code=400, detail="CSV contains missing values")

    # Start the optimization process
    PERIODS_FUTURE = 15

    # Calculate expected returns for a specific period by ticket
    #df_hat_stats = get_stats_returns_hat(df, periods_future=PERIODS_FUTURE)
    stddev_hat = df.describe().loc["std"] #df_hat_stats.loc["std"]  # First risk metric: expected standard deviation
    returns_hat = df.describe().loc["mean"] #df_hat_stats.loc["mean"]  # Expected return metric

    # Generate a second risk metric
    df_risk = get_risk_metrics(df)
    vect_cvar = -1*df_risk.loc["CVaR"]  # Using CVaR as a complementary risk metric

    # Generate a third risk metric to measure portfolio risk due to low diversification (using covariance)
    cov_matrix = np.array(df.cov())
    # Simulate extreme volatility scenarios with random portfolios to scale the metric
    vol_min, vol_max = simulate_volatility_scenarios(cov_matrix)

    # Define the optimization problem
    def objective(weights):
        return_hat = np.dot(returns_hat, weights)
        return -return_hat  # Maximizing return_hat is equivalent to minimizing -return_hat

    def risk_constraint(weights):
        risk = my_balanced_risk_metric(cov_matrix, vol_min, vol_max, vect_cvar, stddev_hat, weights)
        return risk_level - risk  # We want the risk to be equal to risk_level

    def weights_sum_constraint(weights):
        return np.sum(weights) - 1  # The sum of weights must equal 1

    # Initialize weights
    num_tickets = len(df.columns)
    initial_weights = np.array([1/num_tickets] * num_tickets)

    # Constraints and bounds
    constraints = (
        {'type': 'ineq', 'fun': risk_constraint},
        {'type': 'eq', 'fun': weights_sum_constraint}
    )
    bounds = [(0, max_weight) for _ in range(num_tickets)]  # maximum weight per asset

    # Execute the optimizer
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        # Optimal weights
        optimal_weights = np.round(result.x, decimals=9)
        
        # Verify sum is exactly 1
        if not np.isclose(np.sum(optimal_weights), 1.0, atol=1e-9):
            logger.error("Sum of weights is not exactly 1")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Optimization error",
                    "message": "The optimization result does not sum to 1. Please try again.",
                    "technical_details": f"Sum of weights: {np.sum(optimal_weights)}"
                }
            )

        optimal_portfolio = dict(zip(dict_letters, optimal_weights))
        
        # Validate with Pydantic
        try:
            validated_portfolio = PortfolioWeights(weights=optimal_portfolio)
            logger.info(f"Optimal weights calculated and validated successfully: {optimal_weights}")
            return JSONResponse(content={"optimal_portfolio": validated_portfolio.model_dump()["weights"]})
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Validation error",
                    "message": "The optimization result could not be validated. Please try again.",
                    "technical_details": str(e)
                }
            )
    else:
        logger.error(f"Optimization failed: {result.message}")
        logger.warning("The target risk level might be too low to achieve")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Optimization failed",
                "message": "Could not find a solution that meets the specified constraints. Please try with a higher risk level or adjust the maximum weight constraints.",
                "technical_details": result.message
            }
        )



# Run the API
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port='8080')