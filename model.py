import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None

    def fit(self, X) -> None:
    # fit the PCA model
        X_meaned = X - np.mean(X , axis = 0)
        cov_mat = np.cov(X_meaned , rowvar = False)
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        n_components = self.n_components #you can select any number of components.
        self.components = sorted_eigenvectors[:,0:n_components]
    
    # def fit(self, X) -> None:
    #     # calculate the covariance matrix
    #     cov = np.cov(X.T)

    #     # calculate the eigenvectors and eigenvalues of the covariance matrix
    #     eig_vals, eig_vecs = np.linalg.eig(cov)

    #     # sort the eigenvectors based on their eigenvalues
    #     eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    #     eig_pairs.sort(key=lambda x: x[0], reverse=True)

    #     # select the top n_components eigenvectors
    #     if self.n_components <= len(eig_vals):
    #         self.components = np.stack([eig_pairs[i][1] for i in range(self.n_components)], axis=1)
    #     else:
    #         self.components = np.stack([eig_pairs[i][1] for i in range(len(eig_vals))], axis=1)

    
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

    def _svm_loss(self, X, y, C=1) -> float:
        # calculate the hinge loss
        return np.sum(np.maximum(0, 1 - y * (X @ self.w + self.b))) + C * np.sum(self.w ** 2)

    def _svm_grad(self, X, y, C) -> np.ndarray:
        # calculate the gradient of the hinge loss with respect to the parameters
        margin = y * (X @ self.w + self.b)
        mask = margin <= 1
        dLdw = -y[mask] @ X[mask] + 2 * C * self.w
        dLdb = -np.sum(y[mask])
        
        # take the real part of dLdw
        dLdw = np.real(dLdw)
        
        return dLdw, dLdb

    def fit(
        self, X:np.ndarray, y,
        learning_rate: float,
        num_iters: int,
        C: float = 1.0,
    ) -> None:
        self.w = None
        self.b = None
        self._initialize(X)
        
        # np.random.seed(0)
        # fit the SVM model using stochastic gradient descent
        j = num_iters/10
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            # idx = np.random.randint(X.shape[0])
            idx = np.random.randint(60000)
            xi =  X[idx]
            yi =  y[idx]

            z = yi * (np.dot(self.w, xi) + self.b)
            if z < 1:
                if(np.iscomplex(yi)):
                    print("y is complex")
                    return
                if(np.iscomplexobj(self.w)):
                    print("w is complex")
                    return
                if(np.iscomplexobj(xi)):
                    print("x is complex")
                    return
                grad_w = (C * yi * xi) - self.w
                grad_b = C * yi
                if np.iscomplexobj(grad_w):
                    print("grad_w is complex")
                    return
                self.w += learning_rate * grad_w
                # self.w += learning_rate * np.real(grad_w) # why does this become complex? 
                self.b += learning_rate * grad_b
            

            # if i % j == 0:
            #     train_accuracy = self.accuracy_score(X, y)
            #     print(f'iteration {i}/{num_iters}: train_accuracy={train_accuracy}')   
            
            # calculate the gradient of the loss function
            # dLdw, dLdb = self._svm_grad(xi, yi, C)
            # # compute and display the training accuracy every 100 iterations
            # if i % 100 == 0:
            #     train_accuracy = self.accuracy_score(X, y)
            #     print(f'iteration {i}/{num_iters}: train_accuracy={train_accuracy}')       

            # # update the parameters using the gradient descent update rule
            # self.w -= learning_rate * dLdw
            # self.b -= learning_rate * dLdb

                
    
    
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
        # print( np.argmax(scores, axis=0))  
        return np.argmax(scores, axis=0)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        precisions = []
        for i in range(self.num_classes):
            true_positives = np.sum((y == i) & (y_pred == i))
            predicted_positives = np.sum(y_pred == i)
            if predicted_positives > 0:
                precision = true_positives / predicted_positives
                precisions.append(precision)
        return np.mean(precisions)


    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        recalls = []
        for i in range(self.num_classes):
            true_positives = np.sum((y == i) & (y_pred == i))
            actual_positives = np.sum(y == i)
            if actual_positives > 0:
                recall = true_positives / actual_positives
                recalls.append(recall)
        return np.mean(recalls)
    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        
        if precision == 0 and recall == 0:
            return 0
        
        return 2 * precision * recall / (precision + recall)

