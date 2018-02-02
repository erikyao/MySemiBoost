import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from MySemiBoost.semi_booster import SemiBooster
from MySemiBoost.similarity_matrix import SimilarityMatrix


def test_1():
    # data, label = load_breast_cancer(return_X_y=True)
    bc = load_breast_cancer()

    features = pd.DataFrame(bc['data'])
    features.columns = bc['feature_names']
    labels = pd.Series(bc['target']).replace(to_replace=0, value=-1)

    X = MinMaxScaler().fit_transform(X=features)
    S = SimilarityMatrix.compute(X)

    train_test_splitter = StratifiedKFold(n_splits=10, random_state=1031)
    labeled_unlabeled_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1031)
    for train_idx, test_idx in train_test_splitter.split(features, labels):
        train_X = features.iloc[train_idx, :]
        train_y = labels.iloc[train_idx]
        test_X = features.iloc[test_idx, :]
        test_y = labels.iloc[test_idx]

        labeled_idx, unlabeled_idx = next(labeled_unlabeled_splitter.split(train_X, train_y))
        labeled_train_X = train_X.iloc[labeled_idx, :]
        labeled_train_y = train_y.iloc[labeled_idx]
        unlabeled_train_X = train_X.iloc[unlabeled_idx, :]
        # unlabeled_train_y = train_y.iloc[unlabeled_idx]  # Not necessary for SemiBoost

        lr_config = dict(penalty='l2', C=1.0, class_weight=None, random_state=1337,
                         solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)
        lr = LogisticRegression(**lr_config)

        lr.fit(train_X, train_y)
        lr_pred_test_y = pd.Series(lr.predict(test_X), index=test_X.index)
        lr_accuracy = lr.score(test_X, test_y)

        print("Logistic Regression Acc: {}".format(lr_accuracy))

        # new_sigma = S.tenth_percentiles()[0]

        sb = SemiBooster(unlabeled_feat=unlabeled_train_X,
                         sigma=S.sigma,
                         S=S,
                         T=2,
                         sample_percent=0.1,
                         base_classifier=lr)
        sb.fit(labeled_feat=labeled_train_X,
               labels=labeled_train_y)

        sb_pred_test_y = sb.predict(test_X)

        sb_accuracy = sum(sb_pred_test_y == test_y) / len(test_y)
        print("SemiBoost Acc: {}".format(sb_accuracy))

        print("Size of test_y: {}; Same pred: {}".format(len(test_y), sum(sb_pred_test_y == lr_pred_test_y)))


def test_2():
    bc = load_breast_cancer()

    features = pd.DataFrame(bc['data'])
    features.columns = bc['feature_names']
    labels = pd.Series(bc['target']).replace(to_replace=0, value=-1)

    unlabeled_X = features.iloc[0:200, ]
    # unlabeled_y = labels[0:200]

    labeled_X = features.iloc[200:, ]
    labeled_y = labels[200:]

    cv = StratifiedKFold(n_splits=10, random_state=1031)

    lr_config = dict(penalty='l2', C=1.0, class_weight=None, random_state=1337,
                     solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)
    lr = LogisticRegression(**lr_config)

    # lr_scores = cross_val_score(estimator=lr, X=labeled_X, y=labeled_y,
    #                             cv=cv, n_jobs=1, verbose=0, fit_params=None)
    # print(lr_scores)

    X = MinMaxScaler().fit_transform(X=pd.concat([labeled_X, unlabeled_X]))
    S = SimilarityMatrix.compute(X)
    # new_sigma = S.tenth_percentiles()[0]

    # DO NOT set `base_classifier=lr` because cloning the fitted `lr` would lead to overheads
    sb = SemiBooster(unlabeled_feat=unlabeled_X,
                     sigma=S.sigma,
                     S=S,
                     T=20,
                     sample_percent=0.1,
                     base_classifier=LogisticRegression(**lr_config))

    sb_scores = cross_val_score(estimator=sb, X=labeled_X, y=labeled_y,
                                cv=cv, n_jobs=1, verbose=0)

    print(sb_scores)

    pass


if __name__ == '__main__':
    test_1()

    # lst = list(test_1())
    # df = pd.DataFrame(lst)
    # df.columns = ["Logistic Regression", "Semiboosted Linear Regression"]
    # print(df)
    # df.to_csv("result.tsv", sep="\t", index=False)
