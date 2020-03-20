"""
Non-negative matrix factorization extension (NFME).

A collection of tools from other libraries designed to
make working with non-negative matrix factorization a
little more friendly.

AUTHOR: Cooper M. Stansbury
"""

import random
import warnings
from operator import truediv as div
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import minmax_scale as sklearn_norm
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from skimage import io

import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from scipy import signal

PRINT_OPTIONS = np.set_printoptions(threshold=5)

class NMFE():
    """
    A class to make exploring non-negative matrix factorizations more
    efficient and fun. This class builds on various linear alegebra tools
    in numpy, scipy, and sklearn and is loosely inspired by some ideas
    in nimfa. Non-negative matrix factorization takes an input matrix (I)
    and estimates two matrices (W and H) whose product is approximately
    I. This approximation (E) will vary based on different parameters
    to the factorization. NMFE is designed to help explore these changes
    empirically in order to "sanity" check dimension reduction.

    NMFE expects data and fit parameters up-front on initialization.
    Fit parameters may be re-specified after the initial fit. All
    additional objects will require their own instance of NMFE.

    Example initilialization:
    >>> nmfe = NMFE(input_matrix=<your_data>, norm=True, n_components)
    """

    def __str__(self):
        printable = ', '.join(f"{k}={v}" for k,v in self.init_kwargs.items())
        return f"NMFE(input_matrix={type(self.I)},"
                f" norm={self.normalized}, {printable})"

    def __repr__(self):
        """
        NOTE: `np.arrays` have some trickery. Right now this is handled
        with the "global" variable `PRINT_OPTIONS` in header of this file.
        """
        printable = ', '.join(f"{k}={v}" for k,v in self.init_kwargs.items())
        return f"NMFE(input_matrix={self.I}>, "
                f"norm={self.normalized}, {printable})"

    def _scale(self, normalize, mat):
        """
        A dunder method to handling matrix normalization. This is called
        scaling here to avoid confusion with matrix norms.

        ARGS:
            - normalize (bool): if True, scale the `mat` arg to range [0,1]

            - mat (np.array): matrix to fit, will almost always be
                `self.I`

        RETURNS:
            - `np.array` normalized to N[0,1] scale
        """
        mat = mat.astype(float)
        if normalize:
            mat = sklearn_norm(mat,
                                feature_range=(0, 1),
                                axis=0,
                                copy=True)
        else:
            return mat
        return mat

    def _check_inputs(self, mat):
        """
        A dunder method to gracefully handle NMFE specific inputs.
        All other inputs are expected to be handled by the target.
        The primary reason for this class is to ensure `__init__()`
        is called appropriately. Raises `ValueError` with specific
        messages if `__init__()` is called without a matrix or
        `n_components` defined.

        ARGS:
            - mat (np.array): arg present to check
                input before attribute is set.

        RAISES:
            - `ValueError` with messages depending on the
                severity of the infraction.
        """
        if mat is None:
            raise ValueError("NMFE instantiated without an `input_matrix` "
                            "specified. Try: `NMFE(input_matrix=<np.array>)`")
        elif mat.min() < 0:
            raise ValueError("Detected negative values in `input_matrix`.")

        if not 'n_components' in self.init_kwargs:
            raise ValueError("NMFE instantiated without `n_components`"
                             " specified. Try: `NMFE(n_components=<int>)`")

    def _fit(self, mat, **kwargs):
        """
        A dunder method to calculate non-negative factorization.
        This method is a wrappar around `sklearn.decomposition.NMF`.
        This does not rely on inheritence to allow the user
        to call, to allow a user multiple fits on the same data.

        ARGS:
            - mat (np.array): matrix to fit, will almost always be
                `self.I`

            - kwargs (dict): parameters to pass to `NMF.fit_transform()`

        RETURNS:
            - 3 `np.array`, containing the loadings and estimated
                reconstruction matrix.
        """
        nmf = NMF(**kwargs)
        W = nmf.fit_transform(mat)
        H = nmf.components_
        E = np.dot(W, H)
        return (W, H, E)

    def _compute_residuals(self):
        """
        A method to compute the residuals based on current values
        of I and E. Note, the residuals can be computed in a
        variety of ways. This method is the simple element-wise
        differences between the input and the reconstructed matrix.
        More distances are available via the `compute_dist` method.

        RETURNS:
            - `np.array`
        """
        residuls = self.I - self.E
        return residuls

    def __init__(self, input_matrix=None, norm=False, **kwargs):
        """
        The initialization method for NMFE. This method differs
        from sklearn's NMF class in that it is less flexible.
        Initialization requires data input and does many of the
        factorization computations upfront. The factorization
        philosophy is to compute the simplest factorization as
        a default and allow the user to enhance the factorization
        method by specifying more arguments or re-fitting the input
        matrix.

        Initialization returns five primary martices as attributes.
        Some methods require symetric, invertible matrices. You
        can call `to_square()` in order to coerce your input into
        a square matrix via resampling. Note: these is information loss
        in this process.

        ARGS:
            - input_matrix (np.array): the input matrix on which
                to perform non-negative factorization.

            - norm (bool): should the input be rescaled to the
                [0,1] interval? Note: the default is `False`.

            - **kwargs (dict): additional keyword arguments to pass
                into `NMF.fit_transform()` from sklearn.

        RETURNS:
            - initializes the class and key attributes.
        """
        self.__version__ = "0.0.1"
        self.init_kwargs = kwargs
        self._check_inputs(input_matrix)
        self._n_components = kwargs['n_components']

        # most import attribute is the input, of "I"
        # handle scaling and normalization args
        self.I = self._scale(norm, input_matrix)
        self.normalized = norm

        # set SQUARE flag for graceful function
        # checks later on
        if self.I.shape[0] == self.I.shape[1]:
            self.issquare = True
        else:
            self.issquare = False

        # construct NMF matrices
        self.W, self.H, self.E = self._fit(self.I, **kwargs)

        # construct matrix properties
        self.I_rank = np.linalg.matrix_rank(self.I)
        self.I_ns = null_space(self.I)
        self.E_rank = np.linalg.matrix_rank(self.I)
        self.E_ns = null_space(self.I)

        # compute naive residuals
        self.R = self._compute_residuals()
        self.rss = None

    def refit(self, norm=False, **kwargs):
        """
        A method to recompute the factorization ON THE SAME INPUT.

        ARGS:
            - norm (bool): if True, scale the `mat` arg to range [0,1]

            - kwargs (dict): parameters to pass to `NMF.fit_transform()`

        RETURNS:
            - re-initializes the class and key attributes.
        """
        if len(kwargs) == 0:
            kwargs = self.init_kwargs
        I = self.I.copy()
        self.__init__(I, norm=norm, **kwargs)

    def to_square(self, in_place=True):
        """
        A method force input to a square matrix in order to
        make invertible matrix functions callable. This function
        calls `refit` to update the estimates for H, W and E.
        `reshape_to_square` will determine the axis with
        the larger dimensionality and DOWNSAMPLE that axis to
        the same number of cols/rows as the smaller axis.

        WARNING: this is generally a bad idea, as it distorts the
        input space. Still, it may be ok depending on the domain of
        interest.

        ARGS:
            - in_place (boolean): reset attributes? This will
                overwrite the current values for `W`, `H`, and `E`.

        RETURNS:
            - `np.array`: ONLY returns new loadings and
                and reconstruction estimation when
                `in_place=False`.
        """
        axis_0 = self.I.shape[0]
        axis_1 = self.I.shape[1]

        if axis_0 < axis_1:
            print(f"Reshaping from {self.I.shape} to {(axis_0, axis_0)}"
                  f" with `in_place={in_place}`")
            I_sq = np.apply_along_axis(signal.resample,
                                       axis=1,
                                       arr=self.I,
                                       num=axis_0)

        else:
            print(f"Reshaping from {self.I.shape} to {(axis_1, axis_1)}"
                  f" with `in_place={in_place}`")
            I_sq = np.apply_along_axis(signal.resample,
                           axis=0,
                           arr=self.I,
                           num=axis_1)

        # function specifically calls `_fit()` based on intialization
        # params. `refit()` can be used to change or update
        # the factorization but the flexibility is purposefully
        # limited here as the main goal of this method is input
        # translation, not re-computing
        new_W, new_H, new_E = self._fit(I_sq, **self.init_kwargs)

        if in_place:
            self.issquare = True
            self.W = new_W
            self.H = new_H
            self.E = new_E
            self.I = I_sq
        else:
            return I_sq, new_W, new_H, new_E

    def compute_rss(self, set_attr=True):
        """
        A method to fit residual sum of squares. Based loosley on `nimfa`.
        Note that this method will

        ARGS:
            - set_attr (boolean): should the `self.R` AND `self.rss` be updated?

        RETURNS:
            - `np.array` iff `set_attr=False`
        """
        if not self.issquare:
            raise ValueError("Won't compute `compute_rss()` "
                                "on a non-square matrix.")

        # recompute locally to avoid kwarg collisions (small overhead)
        residuals = self.I - self.E
        rss = np.square(residuals).sum()
        if set_attr:
            self.R = residuals
            self.rss = rss
        else:
            return rss

    def get_residual_norm(self, order='fro'):
        """
        A method to compute the norm of the difference
        between `I` and `E`. This is a method to wrap
        `numpy.linalg.norm` in order to quickly compute the
        "quality" of the factorization.

        ARGS:
            - order (string): the order of the matrix norm pass
                into `np.linalg.norm(ord=order)`. There are currently
                8 supported matrix norms. Please see
                `numpy.linalg.norm` docs for the full list.

        RETURNS:
            - `dict` with the order of the matrix and the value
                of the matrix norm (inspired by R programming)
        """
        if not self.issquare:
            raise ValueError("Won't compute `get_residual_norm()` "
                            "on a non-square matrix.")

        if self.R is None:
            raise ValueError(f"`self.R` is `None`. "
            "Call `compute_residuals(set_attr=True)`"
            " in order to compute and store residuals.")
        return {order: np.linalg.norm(self.R, ord=order)}

    def compute_dist(self, metric='seuclidean'):
        """
        A method to wrap `scipy.spatial.distance` functions.
        This gives the researcher the ability to quickly
        analyze the 'fit' of the NFM estimation.

        ARGS:
            - metric (string): there are a large number
                of valid inputs. Please see
                scipy.spatial.distance.cdist docs for full
                list. The boolean only metrics have been
                excluded from this method.

        RETURNS:
            - `dict` with the metric used to calculate distance
                (inspired by R programming)
        """
        valid_matrics = ["braycurtis", "canberra", "chebyshev",
                         "cityblock", "correlation", "cosine",
                         "euclidean", "jensenshannon", "mahalanobis",
                         "minkowski", "seuclidean", "sqeuclidean"]

        if not metric in valid_matrics:
            raise ValueError(f"`metric='{metric}'` is not supported."
                             f" Please choose from: {valid_matrics}")
        return {metric:cdist(self.I, self.E, metric=metric)}

    def compute_hausdorff(self, full=False):
        """
        A method to compute the Hausdorff distance between
        `I` and `E`.

        ARGS:
            - full (boolean): if true, returns a `tuple`
                containing the contribution points from `I`
                and `E` respectively. This is not what most
                users expect from a distance function, so
                generally we return only the first position,
                which is an `int`.

        RETURN:
            - `int` or `tuple` depending on the value `full`.
        """
        if full:
            return directed_hausdorff(self.I, self.E)
        else:
            return directed_hausdorff(self.I, self.E)[0]

    def compute_fit_error(self, metric='MSE'):
        """
        A cheeky set of functions to compute the fit metrics
        based on keyword input.

        ARGS:
            - metric (string): which of the metrics should
                be returned?

        RETURNS:
            - `dict` with metric and value
        """

        I = self.I.flatten()
        E = self.E.flatten()

        valid_matrics = {
            'EVAR': explained_variance_score(I, E),
            'MAE': mean_absolute_error(I, E),
            'MSE': mean_squared_error(I, E),
            'MSLE': mean_squared_log_error(I, E),
            'MEDAE': median_absolute_error(I,E),
            'R2': r2_score(I, E)
        }

        if not metric in valid_matrics:
            raise ValueError(f"`metric='{metric}'` is not supported."
                             f" Please choose from: {valid_matrics.keys()}")

        return {metric: valid_matrics[metric]}

    def plot(self, matrix="E", cmap='Greys', size=(5,5)):
        """
        A method to plot a given matrix. For example:

        ARGS:
            - matrix (char): a character indicating which
                matrix to plot. Valid choices currently are:
                ['I', 'W', 'H', 'E', 'R']

            - cmap (string): a valid color map.

            - size (tuple): the size of the output.

        RETURNS:
            - NA. This is an inline plotting method.
        """
        matrix_map = {
            "I" : {'name':'input_matrix','arr':self.I},
            "W": {'name':'W matrix','arr':self.W},
            "H": {'name':'H matrix','arr':self.H},
            "E": {'name':'estimated matrix','arr':self.E},
            "R": {'name':'residual matrix','arr':self.R}
        }

        if not matrix in matrix_map:
            raise ValueError(f"Cannot call `plot(matrix='{matrix}')`. "
                             f"Valid options are: {matrix_map.keys()}" )

        fig = plt.figure(figsize=size)
        imgplot = plt.imshow(matrix_map[matrix]['arr'], cmap=cmap)
        name = matrix_map[matrix]['name']
        plt.title(f'{name}', fontsize=15)
        plt.show()

    def plot_IE(self, cmap='Greys', size=(10,10)):
        """
        A method to plot `I` and `E` matrices for quick comparisons.

        ARGS:
            - cmap (string): a valid color map.

            - size (tuple): the size of the output.

        RETURNS:
            - NA. This is just an inline plotting method.
        """
        fig, (ax0, ax1) = plt.subplots(nrows=1,
                ncols=2,
                sharex=True,
                sharey=True)

        ax0.imshow(self.I, cmap=cmap)
        ax0.set_title(f'Original {self.I.shape}',
                      fontsize=15)
        ax1.imshow(self.E, cmap=cmap)
        ax1.set_title(f'W * H with n={self._n_components} {self.E.shape}',
                      fontsize=15)

        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        fig.tight_layout()
        plt.show()

    def plot_all(self, cmap='Greys', size=(10,10)):
        """
        A method to plot matrices I, W, H and E on
        a grid.

        ARGS:
            - cmap (string): a valid color map.

            - size (tuple): the size of the output.

        RETURNS:
            - NA. This is just an inline plotting method.
        """

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2,
                        ncols=2,
                        sharex=True,
                        sharey=True)

        ax0.imshow(self.I, cmap=cmap)
        ax0.set_title(f'Original {self.I.shape}',
                      fontsize=15)
        ax1.imshow(self.W, cmap=cmap)
        ax1.set_title(f'W Loadings {self.W.shape}',
                      fontsize=15)
        ax2.imshow(self.H, cmap=cmap)
        ax2.set_title(f'H Loadings {self.H.shape}',
                      fontsize=15)
        ax3.imshow(self.E, cmap=cmap)
        ax3.set_title(f'W * H with n={self._n_components} {self.E.shape}',
                      fontsize=15)

        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        fig.tight_layout()
        plt.show()
