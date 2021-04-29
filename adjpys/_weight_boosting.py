"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _check_sample_weight
from scipy.special import xlogy
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.optimize import *
from scipy.optimize import fmin_slsqp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import collections

from ._base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_array, check_random_state, check_X_y, _safe_indexing
from ..utils.extmath import softmax
from ..utils.extmath import stable_cumsum
from ..metrics import accuracy_score, r2_score
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_sample_weight
from ..utils.validation import has_fit_parameter
from ..utils.validation import _num_samples

__all__ = [
    'AdaBoostClassifier',
    'AdaBoostRegressor',
]

#VV
def new_auc(y_test,pred):
    conma=confusion_matrix(y_test , pred) # y_test부분에 실제값, 2번째 파라미터로 인자값
    TPR = conma[1,1]/(conma[1,0]+conma[1,1])
    FPR = conma[0,1]/(conma[0,0]+conma[0,1])
    return (1+TPR-FPR)/2

class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _validate_data(self, X, y=None):

        # Accept or convert to these sparse matrix formats so we can
        # use _safe_indexing
        accept_sparse = ['csr', 'csc']
        if y is None:
            ret = check_array(X,
                              accept_sparse=accept_sparse,
                              ensure_2d=False,
                              allow_nd=True,
                              dtype=None)
        else:
            ret = check_X_y(X, y,
                            accept_sparse=accept_sparse,
                            ensure_2d=False,
                            allow_nd=True,
                            dtype=None,
                            y_numeric=is_regressor(self))
        return ret

    def fit(self, X, y, sample_weight=None,epochs=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y)

        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum() # 이거는 뭔진 모르겠는데 일단 (1/넣은 자료수) 이다
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.auc=False
        self.newfit=False
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64) # tree에 부가할 weight이고
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64) # 에러담을 항

        random_state = check_random_state(self.random_state)
        y_ca=to_categorical(y)
        for iboost in range(self.n_estimators):
            # Boosting step
            #for마다 데이터 분리필요

            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y_ca,
                sample_weight,
                random_state,t_y=y,epochs=epochs)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight) #sample weight 정규화.. 움.. 데이터 분할할거면 이거 해결해야함 아니면 과적합 난다.

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        X = self._validate_data(X)

        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])


class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.

    See Also
    --------
    AdaBoostRegressor
        An AdaBoost regressor that begins by fitting a regressor on the
        original dataset and then fits additional copies of the regressor
        on the same dataset but where the weights of instances are
        adjusted according to the error of the current prediction.

    GradientBoostingClassifier
        GB builds an additive model in a forward stage-wise fashion. Regression
        trees are fit on the negative gradient of the binomial or multinomial
        deviance loss function. Binary classification is a special case where
        only a single regression tree is induced.

    sklearn.tree.DecisionTreeClassifier
        A non-parametric supervised learning method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.feature_importances_
    array([0.28..., 0.42..., 0.14..., 0.16...])
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.983...
    """ #VV
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None,epochs=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight,epochs)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state,t_y,epochs=None):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state,t_y,epochs)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state,epochs)

    def _boost_real(self, iboost, X, y, sample_weight, random_state,t_y,epochs=None): # t_y는 카테고리화 되지않은 것
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state) # 여기가 복제로 만드는 곳 같음
        #print(iboost,"번째 estimator fit")
        estimator.fit(X, y, sample_weight=sample_weight,epochs=epochs,verbose=0)
        # 만약에 여기서 loss ft을 바꾸게 되면 categorical을 못쓰게 됨.

        y_predict_proba = estimator.predict(X) #NN.softmax는 그냥하면 확률나옴
        #print(y_predict_proba)
        if iboost == 0:
            self.classes_ = np.array([0,1])
            self.n_classes_ =  len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != t_y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))
        # Stop if classification is perfect
#        if estimator_error <= 0:
#            return sample_weight, 1., 0. 이게 원래 있던 부분 이거를 애초에 여기서 weight를 조절하는 방식으로 하자

        #c_weight = (1/2)*np.log((1-estimator_error)/estimator_error)
        #print("estimator_error",estimator_error)
        
        if estimator_error <= 0:
            return sample_weight, 1, 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == t_y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, y_predict_proba).sum(axis=1))
        # estimaotr_weight는 .. 음.. sampleweight에 적용할 sample weight 변화 정도라고 보는 것이 맞다.
        #print("estimator_weight",estimator_weight)
        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))
        #print("sample_weight",sample_weight)
            
        return sample_weight, 1, estimator_error # 원래 여기 부분 c_weight가 그냥 1임

    def fit2(self,X,y):
        check_is_fitted(self)
        # Check data
        #X = self._validate_X_predict(X) #V 트리만들 수 있는 자료인지 확인하는 것인듯 비교가 불가능하면 트리로 구성이 불가능하니깐
        self.newfit=True
        # Assign chunk of trees to jobs
        #n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        all_proba = []
        for estimator in self.estimators_:
            all_proba.append(estimator.predict(X))

        x0=np.array([1/len(all_proba)]*len(all_proba))
        #VV
        def g(x):
            return sum(x)-1
        def h(x):
            # 모든 x[i]들이 0이상(따로안적어도 댐) 1이하 여야함
            return 1-x
        def k(x):
            return x
        def f(x):
            out= [np.zeros((X.shape[0], 2), dtype=np.float64)] # 여기서 만약에 X때문에 안대면 all_proba에서 행가져오기
            for i in range(len(all_proba)):
                out += all_proba[i]*x[i]
            pred_proba = out[0][:,1] # 1이 그... 부도에 해당하는 기업 # 여기서 만약에 AUC_ROC내용이 좀 바뀌면 0,1기준으로 바꿔서 roc_auc값 구하기
            # roc바뀌면 여기서 out에서 큰숫자 인덱스를 고르는 방식으로 해가지고 해야할 듯  그거가지고 내가만든 auc함 댐.
            #print(roc_auc_score(y, pred_proba))
            return 1 - roc_auc_score(y, pred_proba) # test값은 받아야함
        self.new_weight=fmin_slsqp(f, x0, eqcons=[g],ieqcons=[h,k],epsilon=0.1)
        return self
    '''
        def f(x):
            out= [np.zeros((X.shape[0], 2), dtype=np.float64)] # 여기서 만약에 X때문에 안대면 all_proba에서 행가져오기
            for i in range(len(all_proba)):
                out += all_proba[i]*x[i]
            pred = self.classes_.take(np.argmax(out, axis=1),
                                       axis=0)
            new_auc(y,pred)
            print(roc_auc_score(y, pred_proba))
            return 1 - roc_auc_score(y, pred_proba) # test값은 받아야함

    '''





    def fit3(self,X,y): # AUC weight
        check_is_fitted(self)
        # Check data
        #X = self._validate_X_predict(X) #V 트리만들 수 있는 자료인지 확인하는 것인듯 비교가 불가능하면 트리로 구성이 불가능하니깐
        self.auc=True
        # Assign chunk of trees to jobs
        #n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        self.aucw = []
        for estimator in self.estimators_:
            self.aucw.append(roc_auc_score(y, estimator.predict(X)[:,1])) # AUC들을가지고 있게 되겠지?
        self.aucw /= sum(self.aucw)
        return self

        
    def predict_proba(self,X): # 0,1에 대한 확률을 반환
        all_proba = []
        for estimator in self.estimators_:
            all_proba.append(estimator.predict(X))
        out= [np.zeros((X.shape[0], 2), dtype=np.float64)] # 여기서 만약에 X때문에 안대면 all_proba에서 행가져오기
        for i in range(len(all_proba)):
            out += all_proba[i]*self.new_weight[i]
        print("내가 바꾼 proba") # 여기서 안돌아감
        return out

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X, Threshold): # X -> X_all 로 변경
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        #------------edit
        self.xlsx =X # 아니 이렇게 해도 주소로 같나? 왜저러지?
        #X = X_all.iloc[:,4:11] #데이터 분리
        X = self._validate_data(X)
        '''
        for i in range(self.n_estimators):
            pred = self.estimators_[i].predict(X) #pred에 0일 확률이랑 1일 확률 들어가는지확인
            for j in range(xlsx.shape[0]): # 전체 -1로 초기화
                xlsx['predict%d' %(i+1)] = -1
            for k in range(X.shape[0]):  # 0 1 확률 비교해서 넣어줌
                if pred[k][0] > pred[k][1]:
                    xlsx.at[X.iloc[k][0]-1,'predict%d' %(i+1)]= 0
                else:
                    xlsx.at[X.iloc[k][0]-1,'predict%d' %(i+1)]= 1
        xlsx.to_excel('result.xlsx', sheet_name = 'predict')
        '''
        

        pred = self.decision_function(X) #여기서 개별것들 확률을 넘겨주는 듯 0,1에 대한 각 확률
        ''' 이 부분이 excel print 결과물 부분일 것
        for i in range(self.n_estimators):
            proba = self.estimators_[i].predict(X) #pred에 0일 확률이랑 1일 확률 들어가는지확인
            print(proba)
            y_predict = self.classes_.take(np.argmax(proba, axis=1),
                                       axis=0)
            #for j in range(self.xlsx.shape[0]): # 전체 -1로 초기화
            self.xlsx['predict%d' %(i+1)] = y_predict
        #---------------
        '''
        if self.n_classes_ == 2: # 이진분류하는 듯
            return self.classes_.take(pred > Threshold, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        #VV
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        self.estimator_weights_ /= sum(self.estimator_weights_)
        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            #pred = sum(_samme_proba(estimator, n_classes, X)
                      # for estimator in self.estimators_)
            if self.newfit:
                pred=sum(self.estimators_[i].predict(X)*self.new_weight[i] for i in range(len(self.new_weight)))
                #이 바로 윗줄에서 new_weight[i]를 하느냐 아니면 저기 위에서 정규화한 estimator_weights_를 하느냐에 따라서 학습방식이 달라지는 것은 맞음
                #근데 만약 후자를 택한다면 실은 AUCRoC를 줄이기 위해서 한것은 아니고 그냥 그때그때 잘맞도록 한것임. 그때그떄 상황에 따라 오답률에 맞게 변화시킨 것
                #AUC_ROC를 위해선 new_weight를 짜야함
                #print(pred)
            elif self.auc: # VV auc weight
                pred=sum(self.estimators_[i].predict(X)*self.aucw[i] for i in range(len(self.aucw)))
            else :
                pred=sum(estimator.predict(X) for estimator in self.estimators_)
                pred /= len(self.estimator_weights_)


        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        
        return pred[:,1]
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        """Compute probabilities from the decision function.

        This is based eq. (4) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
        #("computedecision",decision) #VV
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
            #print(decision)
        else:
            decision /= (n_classes - 1)
        return softmax(decision, copy=False)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        n_classes = self.n_classes_
        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        X = self._validate_data(X)

        n_classes = self.n_classes_

        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        X = self._validate_data(X)
        return np.log(self.predict_proba(X))


class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        ``DecisionTreeRegressor(max_depth=3)``.

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.feature_importances_
    array([0.2788..., 0.7109..., 0.0065..., 0.0036...])
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...

    See also
    --------
    AdaBoostClassifier, GradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (real numbers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check loss
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)), size=_num_samples(X), replace=True,
            p=sample_weight
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == 'square':
            masked_error_vector **= 2
        elif self.loss == 'exponential':
            masked_error_vector = 1. - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1. - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        -------
        y : generator of array, shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
