import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = 0
        denominator = 0

        for i in range(n):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
    
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.slope * x + self.intercept)
        return y_pred
    
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
lr = LinearRegression()
lr.fit(X, y)
print(lr.slope)  # Output: 0.6
print(lr.intercept)  # Output: 2.2
y_pred = lr.predict(X)
print(y_pred)  # Output: [2.8, 3.4, 4.0, 4.6, 5.2]


