
'''
Marco and Financial Economertics 
EMET 3008 and EMET8010 Lab Problem Week2 
Thomas Yang and Jianhua Mei
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
## statsmodels
# 1.Regression model: eg. OLS GLS
# 2.Times series model: eg. AR, MA, ARMA and ARIMA
# 3.Non-parametric method: Kernel Density Estimation
# 4.Hypothetical test     

#%% Lab Problem 1: OLS Estimation
'''
Simulate data that follows the simple linear regression model. The aim of this 
problem is to estimate the regression parameters and test the assumptions of OLS.
'''
# Q1
'''
1.Generate a vector of 500 observations from a standard normal distribution 
and assign it as the independent variable X.
'''


#np.random.seed(0)

# mean
mu = 0
# std    
sigma = 1 
# number of observations
obs = 500

# Generate normal distirbution sample
X = np.random.normal(loc=mu, scale=sigma, size=obs)

plt.figure(1)

# Histogram Plot
plt.hist(X, bins=100)

# Set label and title
plt.title("Histogram of Sampled Data")
plt.xlabel("X Value")
plt.ylabel("Frequency")

# Show Histogram 
plt.show()

# Q2 

'''
Construct the dependent variable Y using the formula Y = 2 + 3X + ε, where ε is
an error term composed of 500 observations drawn from a standard normal distribution
'''
#np.random.seed(100)
# mean
mu = 0
# std    
sigma = 1 
# number of observations
#obs = 500

epsilon = np.random.normal(loc=mu, scale=sigma, size=obs)

# Figure
plt.figure(2)

# Histogram Plot
plt.hist(epsilon, bins=100)

# Set label and title
plt.title("Histogram of epsilon")
plt.xlabel("epsilon Value")
plt.ylabel("Frequency")

# Show Histogram 
plt.show()

# Method 1 for loop
Y1 = []
for i in range(len(X)):
    Y1_temp = 2 + 3 * X[i] + epsilon[i]
    Y1.append(Y1_temp)
Y1 = np.array(Y1)

# Method 2 vectorization
Y2 = 2 + 3 * X + epsilon

# Q3
'''
Estimate the parameters of the regression model Y = beta_0 + beta_1 * X + epsilon using OLS.
'''
Y = Y2
# add a constant term for the intercept
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
print(model.summary())

# Data plot
plt.scatter(X[:,1], Y, color='blue', alpha=0.5) 
# Q4
# Prediciton
Y_pred = model.predict(X)  
plt.scatter(X[:,1], Y_pred, color='red')  
# Scatter plot of X and Y with fitted line
plt.title('Scatter plot of X and Y with fitted line')
plt.xlabel('X')
plt.ylabel('Y')
# Show Plt
plt.show()

#%% Lab Problem 2: Heteroscedasticity and Robust Standard Errors 
'''
This problem introduces the concept of heteroscedasticity and how to handle it using robust
standard errors.
'''
# Q1
'''
1. Generate a vector of 500 observations from a standard normal distribution 
and assign it as the independent variable X.
'''
# X
#np.random.seed(0)
# mean
mu = 0
# std    
sigma = 1 
# number of observations
obs = 500

# Generate normal distirbution sample
X = abs(np.random.normal(loc=mu, scale=sigma, size=obs))
#X = sm.add_constant(X) 

# Epilson
#np.random.seed(100)
# mean
mu = 0
# std    
sigma = 1 
# number of observations
#obs = 500

epsilon = np.random.normal(loc=mu, scale=sigma, size=obs)

#%%
# Q2
Y1 = []
for i in range(len(X)):
    epsilon_heteroskedastic = X[i] * epsilon[i]
    Y1.append(2 + 3* X[i] + epsilon_heteroskedastic)
Y1 = np.array(Y1)

epsilon_heteroskedastic = X * epsilon
Y2 = 2 + 3 * X + epsilon_heteroskedastic
Y3 = 2 + 3 * X + epsilon

plt.figure()
plt.scatter(X, Y2, s = 5)

# Scatter plot of X vs Y
plt.title("Scatter plot of X vs Y")
plt.xlabel("X")
plt.ylabel("Y")

# 显示图形
plt.show()

#%% Q3
'''
Estimate the parameters of the regression model Y = beta_0 + beta_1 * X + epsilon using OLS.
'''
Y = Y2
X = sm.add_constant(X) 
results = sm.OLS(Y, X).fit()
print(results.summary())

#%%
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
# Perform Breusch-Pagan test
bp_test = het_breuschpagan(results.resid, X)
# Print results
'''
    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue : float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic
'''
print('Lagrange multiplier statistic = {}'.format(bp_test[0]))
print('p-value of LM test = {}'.format(bp_test[1]))
print('f-statistic of the hypothesis={}'.format(bp_test[2]))
print('p-value for f-test = {}'.format(bp_test[3]))
# Perform White Test
white_test = het_white(results.resid, X)
print('Lagrange multiplier statistic = {}'.format(white_test[0]))
print('p-value of LM test = {}'.format(white_test[1]))
print('f-statistic of the hypothesis={}'.format(white_test[2]))
print('p-value for f-test = {}'.format(white_test[3]))
# Perform White Test
#%%
# Refit the model with robust standard errors
robust_results = sm.OLS(Y, X).fit(cov_type='HC3')
results = sm.OLS(Y, X).fit()
print(robust_results.summary())
print(results.summary())

#%%
model_gls = sm.GLS(Y, X)
results_gls = model_gls.fit()
print(results_gls.summary())
#%%
model_robust = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
results_robust = model_robust.fit()
# results_robust
print(results_robust.summary())


#%% Lab Problem 3: MLE for the Binomial Logistic Regression Model
'''
This problem will help you understand the MLE method applied to a logistic regression model.
'''
from scipy.stats import logistic



# X
np.random.seed(0)
# mean
mu = 0
# std    
sigma = 1 
# number of observations
obs = 500

# Generate normal distirbution sample
X = np.random.normal(loc=mu, scale=sigma, size=obs)

# epsilon 
epsilon = logistic.rvs(size = obs)

# Y

Y = ((0.5 + 3*X - epsilon) > 0).astype(int)

# MLE Estimation
X = sm.add_constant(X)
results = sm.Logit(Y,X).fit()
print(results.summary())


#%% Compute model prediction probabilities
pred_prob = results.predict(X)

# Sort values for plotting
sort_idx = np.argsort(X[:,1])
x_sorted = X[sort_idx,1]
y_sorted = pred_prob[sort_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:,1], Y, alpha=0.5, s=10, label='Data points')
plt.plot(x_sorted, y_sorted, color='red', label='Logistic model')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.show()
#%% Lab Problem 4: OLS vs MLE
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy import stats 


# X
np.random.seed(0)
# mean
mu = 0
# std    
sigma = 1 
# number of observations
obs = 500

# Generate normal distirbution sample
X = np.random.normal(loc=mu, scale=sigma, size=obs)

# Epilson
#np.random.seed(100)
# mean
mu = 0
# std    
sigma = 1 
# Generate normal distirbution sample
obs = 500
epsilon = np.random.normal(loc=mu, scale=sigma, size=obs)

Y = 2 + 3 * X + epsilon
# Add constant to X for the intercept term
X = sm.add_constant(X)

# OLS estimation
model_ols = sm.OLS(Y, X)
results_ols = model_ols.fit()
print('OLS results:')
print(results_ols.summary())
#%%

# Define log-likelihood function
def minus_log_likelihood(beta, y, x):
    error = y - np.dot(x, beta)
    Log_likelihood = np.sum(stats.norm.logpdf(error, loc=0, scale=1))
    return - Log_likelihood

# Initial guess for beta
beta_init = [0, 0]
# Minimize the negative log-likelihood
result_mle = minimize(minus_log_likelihood, beta_init, args=(Y,X) )
print('\nMLE results:')
print('Coefficient estimates: ', result_mle.x)
# In MLE, standard error can be computed from the inverse of the Hessian matrix
print('Standard errors:', np.sqrt(np.diag(result_mle.hess_inv)))

#%%
def ordinary_least_square_error(beta, y, x):
    objective_function = np.sum((y - np.dot(x, beta))**2)
    return objective_function 

result_ols = minimize(ordinary_least_square_error, beta_init, args=(Y,X) )
print('\nOLS results:')
print('Coefficient estimates: ', result_ols.x)
# In OLS, standard error can be computed from the inverse of the Hessian matrix
print('Standard errors:', np.sqrt(np.diag(result_ols.hess_inv)))
