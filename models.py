"""Numbered model registry. MATLAB-style names are approximations, not replicas."""
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, RationalQuadratic
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from imblearn.ensemble import RUSBoostClassifier

NAMES = {1:'Fine DT', 2:'Medium DT', 3:'Coarse DT', 4:'Linear DA', 5:'Quadratic DA',
    6:'Gaussian NB', 7:'Kernel NB', 8:'Linear SVM', 9:'Quadratic SVM', 10:'Cubic SVM',
    11:'Fine Gaussian SVM', 12:'Medium Gaussian SVM', 13:'Coarse Gaussian SVM',
    14:'Fine KNN', 15:'Medium KNN', 16:'Coarse KNN', 17:'Cosine KNN', 18:'Cubic KNN',
    19:'Weighted KNN', 20:'Boosted Tree EC', 21:'Bagged Tree EC',
    22:'Subspace Discriminant EC', 23:'Subspace KNN EC', 24:'RUS Boosted Tree EC',
    25:'Squared exponential GP', 26:'LR', 27:'Robust linear',
    28:'Interaction linear', 29:'Matern GP', 30:'Rational quadratic GP', 31:'Exponential GP'}
REG_IDS = [1, 2, 3, *range(8, 22), 23, 25, 26, 27, 28, 29, 30, 31]
GP_IDS = [25, 29, 30, 31]


class KernelNB(ClassifierMixin, BaseEstimator):
    """Naive Bayes: product of independent one-dimensional Gaussian KDEs."""
    def __init__(self, bandwidth=.3):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_prior_ = np.log(counts / counts.sum())
        self.densities_ = [[KernelDensity(bandwidth=self.bandwidth).fit(X[y == c, j:j+1])
                            for j in range(X.shape[1])] for c in self.classes_]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'densities_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Incorrect number of predictors')
        logp = np.column_stack([prior + sum(kde.score_samples(X[:, j:j+1])
                    for j, kde in enumerate(kdes))
                    for prior, kdes in zip(self.log_prior_, self.densities_)])
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def inverse_square(distances):
    """Exact matches get all weight; otherwise use inverse squared distance."""
    zero = distances == 0
    weights = 1 / np.maximum(distances, 1e-12)**2
    rows = zero.any(axis=1)
    weights[rows] = zero[rows]
    return weights


def make_model(number, task, seed=42, p=5):
    classification = task == 'Facies'
    if not classification and number not in REG_IDS:
        raise ValueError(f'{NAMES[number]} has no direct regression counterpart')
    tree = DecisionTreeClassifier if classification else DecisionTreeRegressor
    knn = KNeighborsClassifier if classification else KNeighborsRegressor
    bag = BaggingClassifier if classification else BaggingRegressor
    if number in (1, 2, 3):
        model = tree(max_leaf_nodes={1:101, 2:21, 3:5}[number], random_state=seed)
    elif number == 4:
        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif number == 5:
        model = QuadraticDiscriminantAnalysis(reg_param=.05)
    elif number == 6:
        model = GaussianNB()
    elif number == 7:
        model = KernelNB()
    elif number in range(8, 14):
        kernel = 'linear' if number == 8 else 'poly' if number <= 10 else 'rbf'
        scale = np.sqrt(p) * {11:.25, 12:1, 13:4}.get(number, 1)
        kwargs = dict(kernel=kernel, C=1, degree=2 if number == 9 else 3,
                      coef0=1 if kernel == 'poly' else 0, gamma=1/(2*scale**2))
        model = SVC(**kwargs) if classification else SVR(**kwargs, epsilon=.05)
    elif number in range(14, 20):
        k = {14:1, 15:10, 16:100, 17:10, 18:10, 19:10}[number]
        model = knn(n_neighbors=k, metric='cosine' if number == 17 else 'minkowski',
                    p=3 if number == 18 else 2,
                    weights=inverse_square if number == 19 else 'uniform')
    elif number == 20:
        model = (AdaBoostClassifier(estimator=tree(max_leaf_nodes=21, random_state=seed),
                     n_estimators=40, learning_rate=.1, random_state=seed) if classification
                 else GradientBoostingRegressor(n_estimators=80, max_depth=3,
                     learning_rate=.05, random_state=seed))
    elif number == 21:
        model = bag(estimator=tree(random_state=seed), n_estimators=40,
                    random_state=seed, n_jobs=1)
    elif number in (22, 23):
        base = (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                if number == 22 else knn(n_neighbors=10))
        model = bag(estimator=base, n_estimators=30, max_features=.6,
                    bootstrap=False, random_state=seed, n_jobs=1)
    elif number == 24:
        model = RUSBoostClassifier(estimator=tree(max_leaf_nodes=21, random_state=seed),
                    n_estimators=40, learning_rate=.1, random_state=seed)
    elif number in GP_IDS:
        kernels = {25: RBF(np.sqrt(p), length_scale_bounds='fixed'),
                   29: Matern(np.sqrt(p), length_scale_bounds='fixed', nu=1.5),
                   30: RationalQuadratic(np.sqrt(p), alpha=1., length_scale_bounds='fixed', alpha_bounds='fixed'),
                   31: Matern(np.sqrt(p), length_scale_bounds='fixed', nu=.5)}
        kernel = ConstantKernel(1., constant_value_bounds='fixed') * kernels[number]
        model = (GaussianProcessClassifier(kernel=kernel, optimizer=None, random_state=seed)
                 if classification else GaussianProcessRegressor(
                     kernel=kernel + WhiteKernel(.05, noise_level_bounds='fixed'),
                     optimizer=None, random_state=seed))
    elif number == 26:
        model = LinearRegression()
    elif number == 27:
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
    elif number == 28:
        model = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), LinearRegression())
    else:
        raise ValueError(number)
    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             StandardScaler(), model)
    if classification:
        return pipeline
    # Target scaling is also learned only inside each training fold.
    regressor = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    if task == 'Permeability':
        regressor = TransformedTargetRegressor(regressor=regressor, func=np.log10, inverse_func=pow10)
    return regressor


def pow10(x):
    return np.power(10., x)
