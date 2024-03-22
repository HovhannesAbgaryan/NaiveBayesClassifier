import numpy as np


class NaiveBayes:
    # region Constructor

    def __init__(self, smoothing=False):
        self.smoothing = smoothing

    # endregion Constructor

    # region Functions

    def calculate_priors(self):
        # region Summary
        """
        Hint: store priors in a pandas Series or a list
        :return: recall: prior is P(label=l_i)
        """
        # endregion Summary

        # region Body

        priors = ((self.y_train.value_counts() + 1 * self.smoothing) /
                  (len(self.y_train) + self.y_train.nunique() * self.smoothing))

        return priors

        # endregion Body

    def calculate_likelihoods(self):
        # region Summary
        """
        Hint: store likelihoods in a data structure like dictionary:
                    feature_j = [likelihood_k]
                    likelihoods = {label_i: [feature_j]}
              Where j implies iteration over features, and k implies iteration over different values of feature j.
              Also, i implies iteration over different values of label.
              Likelihoods, is then a dictionary that maps different label values to its corresponding likelihoods with
              respect to feature values (list of lists).

              NB: The above pseudocode is for the purpose of understanding the logic, but it could also be implemented
              as it is. You are free to use any other data structure or way that is convenient to you!

              More Coding Hints: You are encouraged to use Pandas as much as possible for all these parts as it comes
              with flexible and convenient indexing features which makes the task easier.
        :return: recall: likelihood is P(feature=f_j|label=l_i)
        """
        # endregion Summary

        # region Body

        X_train = self.X_train
        y_train = self.y_train
        labels = y_train.unique()
        nr_labels = len(labels)
        smoothing = self.smoothing
        nr_features = X_train.shape[1]
        likelihoods = {}
        for label in labels:
            for col_id in range(nr_features):
                feature = X_train.iloc[:, col_id]
                levels = feature.unique()
                for level in levels:
                    label_mask = y_train == label
                    likelihoods[f'{col_id}={level}|{label}'] = ((((feature == level) & label_mask).sum() + 1 * smoothing)
                                                                / (label_mask.sum() + nr_labels * smoothing))
        return likelihoods

        # endregion Body

    def fit(self, X_train, y_train):
        # region Summary
        """
        Use this method to learn the model. If you feel it is easier to calculate priors and likelihoods at the same
        time, then feel free to change this method.
        :param X_train:
        :param y_train:
        :return:
        """
        # endregion Summary

        # region Body

        self.X_train = X_train
        self.y_train = y_train
        self.priors = self.calculate_priors()
        self.likelihoods = self.calculate_likelihoods()

        # endregion Body

    def predict(self, X_test):
        # region Summary
        """
        recall: posterior is P(label_i|feature_j)
        hint: Posterior probability is a matrix of size m*n (m samples and n labels).
              Our prediction for each instance in data is the class that has the highest posterior probability.
              You do not need to normalize your posterior, meaning that for classification, prior and likelihood are
              enough and there is no need to divide by evidence. Think why!
        :param X_test:
        :return: a list of class labels (predicted)
        """
        # endregion Summary

        # region Body

        likelihoods = self.likelihoods
        priors = self.priors
        labels = self.y_train.unique()
        nr_test = X_test.shape[0]
        prediction = []
        for i in range(nr_test):
            instance = X_test.iloc[i, :]
            probabilities = []
            for label in labels:
                probability = priors[priors.index == label]
                for idx, feature in enumerate(instance):
                    probability *= likelihoods[f'{idx}={feature}|{label}']
                probabilities.append(probability)
            prediction.append(labels[np.argmax(probabilities)])
        return np.array(prediction)

        # endregion Body

    # endregion Functions
