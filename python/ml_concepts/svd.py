import logging
from numpy import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reference: http://glowingpython.blogspot.com/2011/06/svd-decomposition-with-numpy.html
def svd_example():
    A = floor(random.rand(4,4)*20-6)
    logger.info("Matrix A:\n{0}".format(A))
    b = floor(random.rand(4,1)*20-6)
    logger.info("Matrix B:\n{0}".format(b))

    U,s,V_t = linalg.svd(A) # SVD decomposition of A
    logger.info("Matrix U:\n{0}".format(U))
    logger.info("Matrix S:\n{0}".format(s))
    logger.info("Matrix V(transpose\n:{0}".format(U))

    logger.info("Computing inverse using linalg.pinv")
    # Computing the inverse using pinv
    inv_pinv = linalg.pinv(A)
    logger.info("pinv:\n{0}".format(inv_pinv))

    # Computing inverse using matrix decomposition
    logger.info("Computing inverse using svd matrix decomposition")
    inv_svd = dot(dot(V_t.T, linalg.inv(diag(s))), U.T)
    logger.info("svd inverse:\n{0}".format(inv_svd))
    logger.info("comparing the results from pinv and svd_inverse:\n{0}".format(allclose(inv_pinv, inv_svd)))

    logger.info("Sol1: Solving x using pinv matrix... x=A^-1 x b")
    result_pinv_x = dot(inv_pinv, b)

    logger.info("Sol2: Solving x using svd_inverse matrix... x=A^-1 x b")
    result_svd_x = dot(inv_svd, b)

    if not allclose(result_pinv_x, result_svd_x):
        raise ValueError('Should have been True')


if __name__ == '__main__':
    svd_example()

