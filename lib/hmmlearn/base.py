import logging
import string
import sys
from collections import deque, defaultdict
import inspect
import warnings

import numpy as np
from scipy import linalg, special
# from sklearn.base import BaseEstimator
# from sklearn.utils import check_array, check_random_state

from . import _hmmc, _utils
from .utils import normalize, log_normalize, iter_from_X_lengths, log_mask_zero, check_random_state


_log = logging.getLogger(__name__)
#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class ConvergenceMonitor:
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.

    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.

    >>> from hmmlearn.base import ConvergenceMonitor
    >>> from hmmlearn import hmm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = hmm.GaussianHMM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        # XXX we might want to check that ``logprob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))


class BaseEstimator:
    """Base class for all estimators in scikit-learn.
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    # def __repr__(self, N_CHAR_MAX=700):
    #     # N_CHAR_MAX is the (approximate) maximum number of non-blank
    #     # characters to render. We pass it as an optional parameter to ease
    #     # the tests.

    #     from .utils._pprint import _EstimatorPrettyPrinter

    #     N_MAX_ELEMENTS_TO_SHOW = 30  # number of elements to show in sequences

    #     # use ellipsis for sequences with a lot of elements
    #     pp = _EstimatorPrettyPrinter(
    #         compact=True, indent=1, indent_at_name=True,
    #         n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)

    #     repr_ = pp.pformat(self)

    #     # Use bruteforce ellipsis when there are a lot of non-blank characters
    #     n_nonblank = len(''.join(repr_.split()))
    #     if n_nonblank > N_CHAR_MAX:
    #         lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
    #         regex = r'^(\s*\S){%d}' % lim
    #         # The regex '^(\s*\S){%d}' % n
    #         # matches from the start of the string until the nth non-blank
    #         # character:
    #         # - ^ matches the start of string
    #         # - (pattern){n} matches n repetitions of pattern
    #         # - \s*\S matches a non-blank char following zero or more blanks
    #         left_lim = re.match(regex, repr_).end()
    #         right_lim = re.match(regex, repr_[::-1]).end()

    #         if '\n' in repr_[left_lim:-right_lim]:
    #             # The left side and right side aren't on the same line.
    #             # To avoid weird cuts, e.g.:
    #             # categoric...ore',
    #             # we need to start the right side with an appropriate newline
    #             # character so that it renders properly as:
    #             # categoric...
    #             # handle_unknown='ignore',
    #             # so we add [^\n]*\n which matches until the next \n
    #             regex += r'[^\n]*\n'
    #             right_lim = re.match(regex, repr_[::-1]).end()

    #         ellipsis = '...'
    #         if left_lim + len(ellipsis) < len(repr_) - right_lim:
    #             # Only add ellipsis if it results in a shorter repr
    #             repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]

    #     return repr_

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version='0.24.2')
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            if pickle_version != '0.24.2':
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, '0.24.2'),
                    UserWarning)
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

    def _more_tags(self):
        _DEFAULT_TAGS = {
            'non_deterministic': False,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            "allow_nan": False,
            'stateless': False,
            'multilabel': False,
            '_skip_test': False,
            '_xfail_checks': False,
            'multioutput_only': False,
            'binary_only': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64],
            'requires_y': False,
            'pairwise': False,
        }
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input.")

    # def _validate_data(self, X, y='no_validation', reset=True,
    #                    validate_separately=False, **check_params):
    #     """Validate input data and set or check the `n_features_in_` attribute.
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix, dataframe} of shape \
    #             (n_samples, n_features)
    #         The input samples.
    #     y : array-like of shape (n_samples,), default='no_validation'
    #         The targets.
    #         - If `None`, `check_array` is called on `X`. If the estimator's
    #           requires_y tag is True, then an error will be raised.
    #         - If `'no_validation'`, `check_array` is called on `X` and the
    #           estimator's requires_y tag is ignored. This is a default
    #           placeholder and is never meant to be explicitly set.
    #         - Otherwise, both `X` and `y` are checked with either `check_array`
    #           or `check_X_y` depending on `validate_separately`.
    #     reset : bool, default=True
    #         Whether to reset the `n_features_in_` attribute.
    #         If False, the input will be checked for consistency with data
    #         provided when reset was last True.
    #         .. note::
    #            It is recommended to call reset=True in `fit` and in the first
    #            call to `partial_fit`. All other methods that validate `X`
    #            should set `reset=False`.
    #     validate_separately : False or tuple of dicts, default=False
    #         Only used if y is not None.
    #         If False, call validate_X_y(). Else, it must be a tuple of kwargs
    #         to be used for calling check_array() on X and y respectively.
    #     **check_params : kwargs
    #         Parameters passed to :func:`sklearn.utils.check_array` or
    #         :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
    #         is not False.
    #     Returns
    #     -------
    #     out : {ndarray, sparse matrix} or tuple of these
    #         The validated input. A tuple is returned if `y` is not None.
    #     """

    #     if y is None:
    #         if self._get_tags()['requires_y']:
    #             raise ValueError(
    #                 f"This {self.__class__.__name__} estimator "
    #                 f"requires y to be passed, but the target y is None."
    #             )
    #         X = check_array(X, **check_params)
    #         out = X
    #     elif isinstance(y, str) and y == 'no_validation':
    #         X = check_array(X, **check_params)
    #         out = X
    #     else:
    #         if validate_separately:
    #             # We need this because some estimators validate X and y
    #             # separately, and in general, separately calling check_array()
    #             # on X and y isn't equivalent to just calling check_X_y()
    #             # :(
    #             check_X_params, check_y_params = validate_separately
    #             X = check_array(X, **check_X_params)
    #             y = check_array(y, **check_y_params)
    #         else:
    #             X, y = check_X_y(X, y, **check_params)
    #         out = X, y

    #     if check_params.get('ensure_2d', True):
    #         self._check_n_features(X, reset=reset)

    #     return out

    # @property
    # def _repr_html_(self):
    #     """HTML representation of estimator.
    #     This is redundant with the logic of `_repr_mimebundle_`. The latter
    #     should be favorted in the long term, `_repr_html_` is only
    #     implemented for consumers who do not interpret `_repr_mimbundle_`.
    #     """
    #     if get_config()["display"] != 'diagram':
    #         raise AttributeError("_repr_html_ is only defined when the "
    #                              "'display' configuration option is set to "
    #                              "'diagram'")
    #     return self._repr_html_inner

    # def _repr_html_inner(self):
    #     """This function is returned by the @property `_repr_html_` to make
    #     `hasattr(estimator, "_repr_html_") return `True` or `False` depending
    #     on `get_config()["display"]`.
    #     """
    #     return estimator_html_repr(self)

    # def _repr_mimebundle_(self, **kwargs):
    #     """Mime bundle used by jupyter kernels to display estimator"""
    #     output = {"text/plain": repr(self)}
    #     if get_config()["display"] == 'diagram':
    #         output["text/html"] = estimator_html_repr(self)
    #     return output


class _BaseHMM(BaseEstimator):
    r"""Base class for Hidden Markov Models.

    This class allows for easy evaluation of, sampling from, and
    maximum a posteriori estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        self.n_components = n_components
        self.params = params
        self.init_params = init_params
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        _utils.check_is_fitted(self, "transmat_")
        eigvals, eigvecs = linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def score_samples(self, X, lengths=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        # X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        posteriors = np.zeros((n_samples, self.n_components))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij

            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    def score(self, X, lengths=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        # X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij
        return logprob

    def _decode_viterbi(self, X):
        framelogprob = self._compute_log_likelihood(X)
        return self._do_viterbi_pass(framelogprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        # X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j])
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, lengths=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, lengths)
        return state_sequence

    def predict_proba(self, X, lengths=None):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.

        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        # X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_._reset()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        if (self.transmat_.sum(axis=1) == 0).any():
            _log.warning("Some rows of transmat_ have zero sum because no "
                         "transition from the state was ever observed.")

        return self

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_samples, n_components, log_mask_zero(self.startprob_),
            log_mask_zero(self.transmat_), framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_mask_zero(self.transmat_),
                       framelogprob, fwdlattice)
        with np.errstate(under="ignore"):
            return special.logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        _hmmc._backward(n_samples, n_components,
                        log_mask_zero(self.startprob_),
                        log_mask_zero(self.transmat_),
                        framelogprob, bwdlattice)
        return bwdlattice

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _needs_init(self, code, name):
        if code in self.init_params:
            if hasattr(self, name):
                _log.warning(
                    "Even though the %r attribute is set, it will be "
                    "overwritten during initialization because 'init_params' "
                    "contains %r", name, code)
            return True
        if not hasattr(self, name):
            return True
        return False

    def _get_n_fit_scalars_per_param(self):
        """Return a mapping of fittable parameter name (as in ``self.params``)
        to the number of corresponding scalar parameters that will actually be
        fitted.

        This is used to detect whether the user did not pass enough data points
        for a non-degenerate fit.
        """

    def _init(self, X, lengths):
        """Initializes model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        init = 1. / self.n_components
        if self._needs_init("s", "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if self._needs_init("t", "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components),
                                     init)
        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        if n_fit_scalars_per_param is not None:
            n_fit_scalars = sum(
                n_fit_scalars_per_param[p] for p in self.params)
            if X.size < n_fit_scalars:
                _log.warning(
                    "Fitting a model with %d free scalar parameters with only "
                    "%d data points will result in a degenerate solution.",
                    n_fit_scalars, X.size)

    def _check(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_.sum(axis=1)))

    def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """

    def _generate_sample_from_state(self, state, random_state=None):
        """Generates a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.

        random_state: RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.

        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.

        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            _hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_),
                                      bwdlattice, framelogprob,
                                      log_xi_sum)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def _do_mstep(self, stats):
        """Performs the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # If a prior is < 1, `prior - 1 + starts['start']` can be negative.  In
        # that case maximization of (n1+e1) log p1 + ... + (ns+es) log ps under
        # the conditions sum(p) = 1 and all(p >= 0) show that the negative
        # terms can just be set to zero.
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        if 's' in self.params:
            startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                    0)
            self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
            self.transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
            normalize(self.transmat_, axis=1)
