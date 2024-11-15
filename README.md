# SurvBETA: Ensemble-Based Survival Models Using Beran Estimators and Several Attention Mechanisms

Many ensemble-based models have been proposed to solve machine learning problems in the survival analysis framework, including random survival forests, the gradient boosting machine with weak survival models, ensembles of the Cox models. To extend the set of models, a new ensemble-based model called SurvBETA (the Survival Beran estimator Ensemble using Three Attention mechanisms) is proposed where the Beran estimator is used as a weak learner in the ensemble. The Beran estimator can be regarded as a kernel regression model taking into account the relationship between instances. Outputs of weak learners in the form of conditional survival functions are aggregated with attention weights taking into account the distance between the analyzed instance and prototypes of all bootstrap samples. The attention mechanism is used three times: for implementation of the Beran estimators, for determining specific prototypes of bootstrap samples and for aggregating the weak model predictions. The proposed model is presented in two forms: in a general form requiring to solve a complex optimization problem for its training; in a simplified form by considering a special representation of the attention weights by means of the imprecise Huber’s contamination model which leads to solving simple optimization problem. Numerical experiments illustrate properties of the model on synthetic data and compare the model with other survival models on real data. A code implementing the proposed model is publicly available.

## Installation

```
git clone https://github.com/NTAILab/SurvBETA.git
cd SurvBETA
pip install .
```

## Package Contents


- `survbeta` – package for working with the survBETA model. Contains classes with Beran (beran.py) model and SurvBETA (ensemble_beran.py).


## Usage

```
from survbeta import Beran, EnsembleBeran

import numpy as np

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

X, y = load_veterans_lung_cancer()
X = OneHotEncoder().fit_transform(X)
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ensemble = EnsembleBeran()
ensemble.fit(X_train, y_train)
H, S, T = ensemble.predict(X_test)
```

Parametrs of EnsembleBeran:

- `omega: float = 1.0` - temperature of global softmax;
- `tau: float = 1.0` - temperature of Beran estimators. Used if C-index and MAY are not optimized;
- `maximum_number_of_pairs: int = 10` - the maximum number of pairs for each sample element to optimize the C-index;
- `n_estimators: int = 10` - number of Beran estimators;
- `size_bagging: float = 0.4` - subsample share for each Beran estimator. Must be in [0.0, 1.0];
- `epsilon: float = 0.5` - the contamination parameter in optimisation. Must be in [0.0, 1.0]. If epsilon = 0, then only softmax is used;
- `lr: float = 1e-1` - learning rate in gradient descent;
- `const_in_div: float = 100.0` - scale parameter in the sigmoid in optimization;
- `num_epoch: int = 100` - number of epochs of gradient descent;
- `MAE_optimisation: bool = False` - should use MAE optimization? True or False;
- `epsilon_optimisation: bool = False` - should use epsilon optimization? True or False;
- `c_index_optimisation: bool = True` - should use C-index optimization? True or False;
- `mode: str = 'gradient'` - optimization mode. If 'gradient', then gradient descent will be used for optimization. If 'linear', then the linear programming problem will be solved.

**Warning:** y_train must be structured array with 2 fields – Status: boolean indicating whether the endpoint has been reached or the event time is right censored. Survival_in_days: total length of follow-up.

## Citation

Will be available later