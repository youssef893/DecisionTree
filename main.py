import pandas as pd
import sklearn.metrics as metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def generate_data():
    features, labels = make_classification(n_samples=500, n_features=4, n_informative=3,
                                           n_classes=3, weights=[0.2, 0.3, 0.5], n_redundant=0, n_clusters_per_class=1,
                                           random_state=584)
    return features, labels


def add_features_name(features, labels):
    features_name = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
    features_df = pd.DataFrame(features, columns=features_name)
    labels_df = pd.DataFrame(labels, columns=['labels'])
    dataset = pd.concat([features_df, labels_df], axis=1, join='inner', sort=False)
    return dataset


def split_dataset(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=584)
    return x_train, x_test, y_train, y_test


def dtree_classifier(x_train, x_test, y_train, y_test):
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=4)
    dtree.fit(x_train, y_train)
    prediction = dtree.predict(x_test)
    print('Accuracy =', metrics.accuracy_score(y_test, prediction) * 100,'%')


def main():
    features_in, labels_in = generate_data()
    x_train, x_test, y_train, y_test = split_dataset(features_in, labels_in)
    dtree_classifier(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
