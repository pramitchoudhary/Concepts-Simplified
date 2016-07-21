import pymc
import numpy as np
import pandas as pd
import matplotlib
import pymc.graph

from matplotlib import pylab

mtcars_data = pd.read_csv('./mtcars.csv')
b0 = pymc.Normal("b0", 0, 0.0003)
b1 = pymc.Normal("b1", 0, 0.0003)
err = pymc.Uniform("err", 0, 500)

# Visualize the features
matplotlib.use('Agg')
pylab.plot(mtcars_data['wt'], mtcars_data['mpg'], 'ok')
pylab.ylabel('mpg')
pylab.xlabel('weight of the car')
pylab.savefig("%s.pdf" % wt_vs_mpg, format='pdf')

x_weight = pymc.Normal("wt", 0, 1, value=np.array(mtcars_data["wt"]), observed=True)

@pymc.deterministic
def pred(alpha=b0, beta=b1, x=x_weight):
    return alpha + beta*x

y = pymc.Normal("y", pred, err, value=np.array(mtcars_data["mpg"]), observed=True)

model = pymc.Model([pred, b0, b1, y, err, x_weight])
graph = pymc.graph.graph(model)
# Visualize the model
graph.write_png('univariate.png')

mcmc = pymc.MCMC(model)
mcmc.sample(50000, 20000)

print np.mean(mcmc.trace('b1')[:])
pylab.hist(mcmc.trace('b1')[:], bins=50)
pylab.savefig("%s.pdf" % alpha_trace, format='pdf')


