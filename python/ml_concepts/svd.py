import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reference: http://glowingpython.blogspot.com/2011/06/svd-decomposition-with-numpy.html
def svd_example():
    a = np.floor(np.random.rand(4, 4)*20-6)
    logger.info("Matrix A:\n %s", a)
    b = np.floor(np.random.rand(4, 1)*20-6)
    logger.info("Matrix B:\n %s", b)

    u, s, v_t = np.linalg.svd(a) # SVD decomposition of A
    logger.info("Matrix U:\n %s", u)
    logger.info("Matrix S:\n %s", s)
    logger.info("Matrix V(transpose:\n %s", u)

    logger.info("Computing inverse using linalg.pinv")
    # Computing the inverse using pinv
    inv_pinv = np.linalg.pinv(a)
    logger.info("pinv:\n %s", inv_pinv)

    # Computing inverse using matrix decomposition
    logger.info("Computing inverse using svd matrix decomposition")
    inv_svd = np.dot(np.dot(v_t.T, np.linalg.inv(np.diag(s))), u.T)
    logger.info("svd inverse:\n %s", inv_svd)
    logger.info("comparing the results from pinv and svd_inverse:\n %s",
                np.allclose(inv_pinv, inv_svd))

    logger.info("Sol1: Solving x using pinv matrix... x=A^-1 x b")
    result_pinv_x = np.dot(inv_pinv, b)

    logger.info("Sol2: Solving x using svd_inverse matrix... x=A^-1 x b")
    result_svd_x = np.dot(inv_svd, b)

    if not np.allclose(result_pinv_x, result_svd_x):
        raise ValueError('Should have been True')


if __name__ == '__main__':
    svd_example()
