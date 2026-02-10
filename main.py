import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ProfessionalLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, l2_penalty=0.1):
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _compute_loss(self, y_true, y_pred):
        m = len(y_true)
        # Mean Squared Error + L2 Regularization
        mse = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        reg_term = (self.l2_penalty / (2 * m)) * np.sum(np.square(self.weights))
        return mse + reg_term

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / m) * (np.dot(X.T, (y_pred - y)) + self.l2_penalty * self.weights)
            db = (1 / m) * np.sum(y_pred - y)

            # 3. Update Parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)



def load_and_prep_data():
    print("Downloading California Housing Dataset...")
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)

    df = df.dropna().select_dtypes(include=[np.number])

    X = df.drop("median_house_value", axis=1).values
    y = df["median_house_value"].values


    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prep_data()

    model = ProfessionalLinearRegression(learning_rate=0.1, iterations=1500, l2_penalty=0.01)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2_score = model.score(y_test, predictions)

    print("-" * 30)
    print(f"Final R2 Score on Test Set: {r2_score:.4f}")

    plt.plot(model.loss_history)
    plt.title("Convergence of Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE + L2)")

    plt.show()
