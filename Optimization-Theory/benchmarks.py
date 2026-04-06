# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris

updated on Sun Feb 9 06:05:50 2025
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import math

def tuning_rf(solution): 
    if not hasattr(tuning_rf, "X_train"):
        data = pd.read_csv("ObesityDataSet.csv")

        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])

        y = data['NObeyesdad']
        X = data.drop(columns=['NObeyesdad'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tuning_rf.X_train = X_train
        tuning_rf.X_test = X_test
        tuning_rf.y_train = y_train
        tuning_rf.y_test = y_test

    X_train = tuning_rf.X_train
    X_test = tuning_rf.X_test
    y_train = tuning_rf.y_train
    y_test = tuning_rf.y_test

    n_estimators = int(solution[0])
    max_depth = int(solution[1])
    min_samples_split = int(solution[2])
    min_samples_leaf = int(solution[3])
    
    max_features_options = ['sqrt', 'log2', None]
    max_features = max_features_options[int(solution[4])]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy

def tuning_svm(solution):
    if not hasattr(tuning_svm, "X_train"):
        data = pd.read_csv("ObesityDataSet.csv")

        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])

        y = data['NObeyesdad']
        X = data.drop(columns=['NObeyesdad'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tuning_svm.X_train = X_train
        tuning_svm.X_test = X_test
        tuning_svm.y_train = y_train
        tuning_svm.y_test = y_test

    X_train = tuning_svm.X_train
    X_test = tuning_svm.X_test
    y_train = tuning_svm.y_train
    y_test = tuning_svm.y_test

    C = 10 ** solution[0]
    gamma = 10 ** solution[1]
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = kernel_options[int(solution[2])]

    model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy

def tuning_knn(solution):
    if not hasattr(tuning_knn, "X_train"):
        data = pd.read_csv("ObesityDataSet.csv")
        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        y = data['NObeyesdad']
        X = data.drop(columns=['NObeyesdad'])
        tuning_knn.X_train, tuning_knn.X_test, tuning_knn.y_train, tuning_knn.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = tuning_knn.X_train
    X_test = tuning_knn.X_test
    y_train = tuning_knn.y_train
    y_test = tuning_knn.y_test

    n_neighbors = int(solution[0])
    weights_options = ['uniform', 'distance']
    weights = weights_options[int(solution[1])]
    p = int(solution[2])

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return 1 - accuracy

def tuning_dt(solution):
    if not hasattr(tuning_dt, "X_train"):
        data = pd.read_csv("ObesityDataSet.csv")
        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        y = data['NObeyesdad']
        X = data.drop(columns=['NObeyesdad'])
        tuning_dt.X_train, tuning_dt.X_test, tuning_dt.y_train, tuning_dt.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = tuning_dt.X_train
    X_test = tuning_dt.X_test
    y_train = tuning_dt.y_train
    y_test = tuning_dt.y_test

    max_depth = int(solution[0])
    min_samples_split = int(solution[1])
    min_samples_leaf = int(solution[2])
    criterion_options = ['gini', 'entropy']
    criterion = criterion_options[int(solution[3])]

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return 1 - accuracy

def optimize_centers(solution):
    if not hasattr(optimize_centers, "X"):
        df = pd.read_csv("Iris.csv")
        X_raw = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
        y_true = df['Species'].astype('category').cat.codes.values  # encode labels
        scaler = StandardScaler()
        optimize_centers.X = scaler.fit_transform(X_raw)
        optimize_centers.y_true = y_true

    X = optimize_centers.X
    y_true = optimize_centers.y_true
    
    k = 3
    dim = 4
    centers = np.array(solution).reshape((k, dim))

    y_pred = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis = 1)

    
    score = adjusted_rand_score(y_true, y_pred)
    
    if score < 0:
        return 1.0

    return 1 - score

def fs(solution):
    if not hasattr(fs, "X_train"):
        data = pd.read_csv("ObesityDataSet.csv")#https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
        
        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])

        y = data['NObeyesdad']
        X = data.drop(columns=['NObeyesdad'])

        fs.X_train, fs.X_test, fs.y_train, fs.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = fs.X_train    
    X_test = fs.X_test    
    y_train = fs.y_train    
    y_test = fs.y_test

    binary_solution = np.array(solution) >= 0.5

    if np.count_nonzero(binary_solution) == 0:
        return 1.0

    selected_X_train = X_train.iloc[:, binary_solution]
    selected_X_test = X_test.iloc[:, binary_solution]

    model = LogisticRegression()
    model.fit(selected_X_train, y_train)
    y_pred = model.predict(selected_X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy


def fs_tuning_rf(solution):
    
    if not hasattr(fs_tuning_rf, "X_train"):
        data = pd.read_csv("diabetes_1000_rows.csv")

        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])

        y = data['Diabetes_012']
        X = data.drop(columns=['Diabetes_012'])

        fs_tuning_rf.X_train, fs_tuning_rf.X_test, \
        fs_tuning_rf.y_train, fs_tuning_rf.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    X_train = fs_tuning_rf.X_train
    X_test = fs_tuning_rf.X_test
    y_train = fs_tuning_rf.y_train
    y_test = fs_tuning_rf.y_test

   
    num_features = X_train.shape[1]
    fs_solution = np.array(solution[:num_features]) >= 0.5

    if np.count_nonzero(fs_solution) == 0:
        return 1.0   # invalid solution

    X_train_fs = X_train.iloc[:, fs_solution]
    X_test_fs = X_test.iloc[:, fs_solution]

 
    idx = num_features

    n_estimators = int(solution[idx])
    max_depth = int(solution[idx + 1])
    min_samples_split = int(solution[idx + 2])
    min_samples_leaf = int(solution[idx + 3])

    max_features_options = ['sqrt', 'log2', None]
    max_features = max_features_options[int(solution[idx + 4])]

    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_test_fs)

    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy


from sklearn.metrics import accuracy_score, precision_score

def fs_tuning_rf_ws(solution):
    # -------------------------------------------------
    # Load & cache dataset
    # -------------------------------------------------
    if not hasattr(fs_tuning_rf_ws, "X_train"):
        data = pd.read_csv("diabetes_1000_rows.csv")

        for col in data.select_dtypes(include='object').columns:
            data[col] = LabelEncoder().fit_transform(data[col])

        y = data['Diabetes_012']
        X = data.drop(columns=['Diabetes_012'])

        fs_tuning_rf_ws.X_train, fs_tuning_rf_ws.X_test, \
        fs_tuning_rf_ws.y_train, fs_tuning_rf_ws.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    X_train = fs_tuning_rf_ws.X_train
    X_test = fs_tuning_rf_ws.X_test
    y_train = fs_tuning_rf_ws.y_train
    y_test = fs_tuning_rf_ws.y_test

   
    num_features = X_train.shape[1]
    fs_solution = np.array(solution[:num_features]) >= 0.5

    selected_count = np.count_nonzero(fs_solution)

    if selected_count == 0:
        return 1.0   # invalid solution

    X_train_fs = X_train.iloc[:, fs_solution]
    X_test_fs = X_test.iloc[:, fs_solution]

    feature_ratio = selected_count / num_features  # f3

    
    idx = num_features

    n_estimators = int(solution[idx])
    max_depth = int(solution[idx + 1])
    min_samples_split = int(solution[idx + 2])
    min_samples_leaf = int(solution[idx + 3])

    max_features_options = ['sqrt', 'log2', None]
    max_features = max_features_options[int(solution[idx + 4])]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

   
    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_test_fs)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)

   
    w1 = 0.4   # accuracy weight
    w2 = 0.4   # precision weight
    w3 = 0.2   # feature reduction weight

    objective = (
        w1 * (1 - accuracy) +
        w2 * (1 - precision) +
        w3 * feature_ratio
    )

    return objective

# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def fs_tuning_rf_credit(solution):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    if not hasattr(fs_tuning_rf_credit, "X_train"):
        data = pd.read_csv("UCI_Credit_Card.csv").sample(n=5000, random_state=42)
        y = data["default.payment.next.month"]
        X = data.drop(columns=["default.payment.next.month"])

        fs_tuning_rf_credit.X_train, fs_tuning_rf_credit.X_test, \
        fs_tuning_rf_credit.y_train, fs_tuning_rf_credit.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    X_train = fs_tuning_rf_credit.X_train
    X_test  = fs_tuning_rf_credit.X_test
    y_train = fs_tuning_rf_credit.y_train
    y_test  = fs_tuning_rf_credit.y_test

    num_features = X_train.shape[1]
    fs_mask = np.array(solution[:num_features]) >= 0.5

    if np.count_nonzero(fs_mask) == 0:
        return 1.0

    X_train_fs = X_train.iloc[:, fs_mask]
    X_test_fs  = X_test.iloc[:, fs_mask]

    # ✅ SAFE INDEXING
    idx = len(solution) - 5

    n_estimators      = int(solution[idx])
    max_depth         = int(solution[idx + 1])
    min_samples_split = int(solution[idx + 2])
    min_samples_leaf  = int(solution[idx + 3])

    max_features_opts = ['sqrt', 'log2', None]
    mf_idx = max(0, min(int(solution[idx + 4]), 2))
    max_features = max_features_opts[mf_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth <= 0 else max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_test_fs)

    return 1 - accuracy_score(y_test, y_pred)

def fs_tuning_rf_credit_ws(solution):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    if not hasattr(fs_tuning_rf_credit_ws, "X_train"):
        data = pd.read_csv("UCI_Credit_Card.csv").sample(n=5000, random_state=42)
        y = data["default.payment.next.month"]
        X = data.drop(columns=["default.payment.next.month"])

        fs_tuning_rf_credit_ws.X_train, fs_tuning_rf_credit_ws.X_test, \
        fs_tuning_rf_credit_ws.y_train, fs_tuning_rf_credit_ws.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    X_train = fs_tuning_rf_credit_ws.X_train
    X_test  = fs_tuning_rf_credit_ws.X_test
    y_train = fs_tuning_rf_credit_ws.y_train
    y_test  = fs_tuning_rf_credit_ws.y_test

    num_features = X_train.shape[1]
    fs_mask = np.array(solution[:num_features]) >= 0.5
    selected = np.count_nonzero(fs_mask)

    if selected == 0:
        return 1.0

    X_train_fs = X_train.iloc[:, fs_mask]
    X_test_fs  = X_test.iloc[:, fs_mask]

    feature_ratio = selected / num_features

    # ✅ SAFE INDEXING
    idx = len(solution) - 5

    n_estimators      = int(solution[idx])
    max_depth         = int(solution[idx + 1])
    min_samples_split = int(solution[idx + 2])
    min_samples_leaf  = int(solution[idx + 3])

    max_features_opts = ['sqrt', 'log2', None]
    mf_idx = max(0, min(int(solution[idx + 4]), 2))
    max_features = max_features_opts[mf_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth <= 0 else max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_test_fs)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    return (
        0.5 * (1 - acc) +
        0.3 * (1 - f1) +
        0.2 * feature_ratio
    )






def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {

        "tuning_rf": ["tuning_rf", [50,5,2,1,0], [500,30,20,20,2], 5],
        "tuning_svm": ["tuning_svm", [-2, -5, 0], [3, 1, 3], 3],
        "tuning_knn": ["tuning_knn", [1, 0, 1], [20, 1, 2], 3],
        "tuning_dt": ["tuning_dt", [3, 2, 1, 0], [30, 20, 20, 1], 4],
        "optimize_centers": ["optimize_centers", 1, 10, 12],
        "fs": ["fs", 0, 1, 16],

        # -------------------------------------------------
        # Credit Dataset – Feature Selection + RF
        # -------------------------------------------------
        "fs_tuning_rf_credit": [
            "fs_tuning_rf_credit",
            [0]*23 + [50, 5, 2, 1, 0],
            [1]*23 + [500, 30, 20, 20, 2],
            28
        ],

        "fs_tuning_rf_credit_ws": [
            "fs_tuning_rf_credit_ws",
            [0]*23 + [50, 5, 2, 1, 0],
            [1]*23 + [500, 30, 20, 20, 2],
            28
        ]
    }

    return param.get(a, "nothing")

    
