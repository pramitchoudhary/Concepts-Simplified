# Reference: 1. http://stats.stackexchange.com/questions/163034/bayesian-logit-model-intuitive-explanation
#            2. http://www.johnmyleswhite.com/notebook/2010/08/20/using-jags-in-r-with-the-rjags-package/

import logging
import scipy.stats as stats
from math import exp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def bayesian_lr_example():
    # create simulated data
    no_of_iterations = 10
    epsilon = stats.norm(0, 1).rvs(size=10)
    x = range(no_of_iterations)
    z = x + epsilon
    p = [1 / (1 + exp(-z[i])) for i in range(10)]
    # Bernoulli distribution to generate labels between 0,1
    y = [stats.bernoulli.rvs(p[i]) for i in range(10)]

    a = stats.norm(0, 0.0001)
    b = stats.norm(0, 0.0001)
    z_hat = [a.rvs() + b.rvs() * x[i] for i in range(10)]
    # Fitting through the logit function to generate probability [0,1]
    p_hat = [1 / (1 + exp(-z_hat[i])) for i in range(10)]
    # Selecting the labels over a bernoulli distribution
    y_hat = [stats.bernoulli.rvs(p_hat[i]) for i in range(10)]

    logger.info("Comparing the values")
    logger.info("Ground Labels %s", y)
    logger.info("Predicted Labels %s", y_hat)

if __name__ == '__main__':
    bayesian_lr_example()