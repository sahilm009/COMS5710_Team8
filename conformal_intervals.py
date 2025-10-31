"""
A brief description of the purposes of this script for solving robustness via generating model-agnostic prediction bands:

This file essentially performs (locally adaptive) conformal prediction given:
- A validation set with predictions and true values (used to compute conformity scores)
- A test set with with predictions

This can be used AFTER fitting the model and generating predictions for a validation and test set... 
Hopefully this code won't need to be adapted too much as we're changing the model itself.

Output should be a prediction band for each test set prediction

The main functions here are as follows:
1. compute_conformity_scores(), which calculates scores from validation predictions
2. get_quantile_threshold(), which obtains the threshold for desired 1-alpha coverage
3. generate_prediction_intervals(), which applies the threshold to test predictions
4. conformal_prediction(), which implements the overall procedure using functions 1-3...
"""

import numpy as np
import pandas as pd


def compute_conformity_scores(predictions, true_counts, score_type='gamma'):
    """
    This function computes conformity scores from predictions and true cell counts.
    
    Arguments are as follows:
    predictions : An array of validation set model predictions
    true_counts : An array of true cell counts for the validation set
    score_type : A string which represents one of two possible conformity scores (for now, can mess with this later):
        'absolute': |y - y_hat| <- NOT locally adaptive, 
        'gamma': |y - y_hat| / y_hat <- we're weighting the conformity score by the mean (assumes variance scales linearly... can scale by different orders of y_hat, potentially)
    
    Output is as follows:
    scores : An ndarray of conformity scores
    """
    # Changing stuff to an np.array if necessary...
    predictions = np.array(predictions)
    true_counts = np.array(true_counts)
    residuals = np.abs(true_counts - predictions) # absolute residual calculation
    
    if score_type == 'gamma': # This is the gamma conformity score as implemented in MAPIE... adapts band width based on predicted mean...
        scores = residuals / (predictions + 1e-8) # Small correction factor incase prediction is 0 
    else:  # absolute case, no local adaptivity
        scores = residuals
    
    return scores


def get_quantile_threshold(conformity_scores, coverage_level):
    """
    This function obtains quantile thresholds for the given desired level of coverage (1-alpha).
    
    Arguments are as follows:
    conformity_scores : An array of conformity scores from validation set (compute_conformity_scores generates these)
    coverage_level: The desired of coverage for prediction bands (a number between 0 and 1)
    
    Output is as follows:
    threshold : A float representing the quantile threshold ... used to determine band per prediction
    """
    n = len(conformity_scores)
    adjusted_level = np.ceil((n + 1) * coverage_level) / n
    adjusted_level = min(adjusted_level, 1.0)
    
    threshold = np.quantile(conformity_scores, adjusted_level)
    return threshold


def generate_prediction_intervals(predictions, threshold, score_type='gamma'):
    """
    Generate prediction intervals using the conformity score threshold.
    
    Arguments:
    predictions : An array of test set predictions
    threshold : The conformity score threshold from validation set (obtained via get_quantile_threshold)
    score_type : String representing conformity score type 'absolute' or 'gamma'
    
    Output:
    lower_bounds : An array of lower bounds of prediction intervals for each test set prediction
    upper_bounds : An array of upper bounds of prediction intervals for each test set prediction
    """
    predictions = np.array(predictions)
    
    if score_type == 'gamma':
        half_width = threshold * predictions
    else:  # absolute
        half_width = threshold
    
    lower_bounds = predictions - half_width
    upper_bounds = predictions + half_width
    
    # Ensure lower bound >= 0 for cell counts
    lower_bounds = np.maximum(lower_bounds, 0)
    
    return lower_bounds, upper_bounds


def conformal_prediction(val_predictions, val_true_counts, 
                        test_predictions, test_true_counts=None,
                        coverage_level=0.90, score_type='gamma'):
    """
    Complete conformal prediction workflow.
    
    Input:
    val_predictions : An array of validation set predictions
    val_true_counts : An array of validation set true countss
    test_predictions : An array oftest set predictions
    test_true_counts : An (optional) array of true counts for the test set(for evaluation)
    coverage_level : The desired coverage level (e.g., 0.90)
    score_type : A string 'gamma' (adaptive) or 'absolute' (constant width) for which conformity score to use
    
    Output:
    results : pd.DataFrame of test set predictions with corresponding intervals
    """
    
    # Compute conformity scores on validation set
    conformity_scores = compute_conformity_scores(val_predictions, val_true_counts, score_type)
    
    # Get quantile threshold
    threshold = get_quantile_threshold(conformity_scores, confidence_level)
    
    # Generate intervals for test set
    lower_bounds, upper_bounds = generate_prediction_intervals(
        test_predictions, threshold, score_type
    )
    
    # Create results DataFrame
    results = pd.DataFrame({
        'predicted_count': test_predictions,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds,
        'interval_width': upper_bounds - lower_bounds
    })
    
    # Adds additional evaluation if true test counts provided, looking at coverage, etc.
    if test_true_counts is not None: # necessary for this evaluation
        test_true_counts = np.array(test_true_counts)
        results['true_count'] = test_true_counts
        results['error'] = results['predicted_count'] - test_true_counts
        results['abs_error'] = np.abs(results['error'])
        results['covered'] = (test_true_counts >= lower_bounds) & (test_true_counts <= upper_bounds)
        
        # Printing summary statistics of coverage
        coverage = results['covered'].mean()
        print(f"Coverage: {coverage:.1%} ({results['covered'].sum()}/{len(results)})")
        print(f"Mean interval width: {results['interval_width'].mean():.2f}")
        print(f"MAE: {results['abs_error'].mean():.2f}")
    
    return results

################## end of main functions ##########
    
# A small example with dummy data
np.random.seed(1234)
    
 # Validation set
val_predictions = np.random.rand(250) * 100 + 50
val_true_counts = val_predictions + np.random.randn(250) * 10
    
    # Test set
test_predictions = np.random.rand(108) * 100 + 50
test_true_counts = test_predictions + np.random.randn(108) * 10
    
    # Run conformal prediction
results = conformal_prediction(
    val_predictions=val_predictions,
    val_true_counts=val_true_counts,
    test_predictions=test_predictions,
    test_true_counts=test_true_counts,
    confidence_level=0.90,
    score_type='gamma')
    
    print("\nFirst 10 predictions:")
    print(results.head(10))
