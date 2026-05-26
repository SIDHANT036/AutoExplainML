import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def trained_rf(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


@pytest.fixture
def trained_lr(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


@pytest.fixture
def trained_dt(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


@pytest.fixture
def trained_gb(cancer_data):
    X_train, X_test, y_train, y_test = cancer_data
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


@pytest.fixture
def small_df():
    """Minimal DataFrame for fast tests."""
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_c": [10, 20, 30, 40, 50],
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def numpy_array():
    return np.random.rand(20, 4)
