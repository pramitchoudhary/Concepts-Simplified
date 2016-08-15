# Reference: https://blog.cordiner.net/2010/06/16/calculating-variance-and-mean-with-mapreduce-python/
import multiprocessing
from multiprocessing import Pool
import numpy


def compute_mean_varaiance(row):
    """Compute mean and variance for each row"""
    print multiprocessing.current_process().name
    return numpy.size(row), numpy.mean(row), numpy.var(row)

def combiner(row1, row2):
    """Combine stats taking 2 rows in a pair"""
    n_a, mean_a, var_a = row1
    n_b, mean_b, var_b = row2
    n_ab = n_a + n_b
    mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab
    var_ab = (((n_a * var_a) + (n_b * var_b)) / n_ab) + ((n_a * n_b) * ((mean_b - mean_a) / n_ab)**2)
    return n_ab, mean_ab, var_ab

def main():
    n_rows = 100
    n_samples_per_row = 500
    pool_instance = Pool(processes=2)
    x = numpy.random.rand(n_rows, n_samples_per_row)
    y = reduce(combiner, pool_instance.map(compute_mean_varaiance, x))
    print "n=%d, mean=%f, var=%f" % y


if __name__ == '__main__':
    main()