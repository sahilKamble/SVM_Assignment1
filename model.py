import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # calculate the covariance matrix
        cov = np.cov(X.T)

        # calculate the eigenvectors and eigenvalues of the covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(cov)

        # sort the eigenvectors based on their eigenvalues
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # select the top n_components eigenvectors
        if self.n_components <= len(eig_vals):
            self.components = np.stack([eig_pairs[i][1] for i in range(self.n_components)], axis=1)
        else:
            self.components = np.stack([eig_pairs[i][1] for i in range(len(eig_vals))], axis=1)

    
    def transform(self, X) -> np.ndarray:
        # project the data onto the top n_components eigenvectors
        return X @ self.components
    
    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def _svm_loss(self, X, y, C) -> float:
        # calculate the hinge loss
        return np.sum(np.maximum(0, 1 - y * (X @ self.w + self.b))) + C * np.sum(self.w ** 2)

    def _svm_grad(self, X, y, C) -> np.ndarray:
            # calculate the gradient of the hinge loss with respect to the parameters
            margin = y * (X @ self.w + self.b)
            mask = margin <= 1
            dLdw = -y[mask] @ X[mask] + 2 * C * self.w
            dLdb = -np.sum(y[mask])
            
            # take the real part of dLdw
            dLdw = dLdw.real

            return dLdw, dLdb


    def fit(
            self, X:np.ndarray, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            idx = np.random.randint(X.shape[0])
            xi, yi = X[idx], y[idx]

            # calculate the gradient of the loss function
            dLdw, dLdb = self._svm_grad(xi, yi, C)

            # update the parameters using the gradient descent update rule
            self.w -= learning_rate * dLdw
            self.b -= learning_rate * dLdb

    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.sign(X @ self.w + self.b)

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)



class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
    # first preprocess the data to make it suitable for the 1-vs-rest SVM model
    # then train the 10 SVM models using the preprocessed data for each class
        for i in range(self.num_classes):
            # preprocess the target labels to make it a binary classification problem
            y_i = np.where(y == i, 1, -1)

            # train the ith SVM model
            svm_i = self.models[i]
            svm_i.fit(X, y_i, **kwargs)

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = []
        for i in range(self.num_classes):
            svm_i = self.models[i]
            scores_i = svm_i.predict(X)
            scores.append(scores_i)
        scores = np.array(scores)
        return np.argmax(scores, axis=0)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        true_positives = [1 for i in range(len(y)) if y[i] == y_pred[i] and y[i] == 1]
        predicted_positives = [1 for i in range(len(y)) if y_pred[i] == 1]
        if len(predicted_positives) == 0:
            return 0
        else:
            return sum(true_positives) / sum(predicted_positives)


    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        true_positives = [1 for i in range(len(y)) if y[i] == y_pred[i] and y[i] == 1]
        actual_positives = [1 for i in range(len(y)) if y[i] == 1]
        if len(actual_positives) == 0:
            return 0
        else:
            return sum(true_positives) / sum(actual_positives)
    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        
        if precision == 0 and recall == 0:
            return 0
        
        return 2 * precision * recall / (precision + recall)

