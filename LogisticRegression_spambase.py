import numpy as np
import os
import math
import matplotlib.pyplot as plt


class LogisticRegression(object):

    def read_data(self):
        x = []
        y = []
        f = open(os.curdir + '/spambase.txt', 'r')
        for line in f:
            numbers = [float(x) for x in line.split(',')]
            x.append(numbers[:58])
            y.append(numbers[-1])

        x = np.array(x)
        y = np.array([y])
        y = y.T  # make y as a column vector
        return x, y

    def feature_normalize(self, x):  # normalize features using (x-mu)/std
        mu = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x-mu)/std
        return x, mu, std

    def padding(self, x, mu, std):
        m = x.shape[0]
        x = np.hstack((np.ones((m, 1), float), x))  # add intercept term for x
        mu = np.hstack((np.zeros(1), mu))  # add 1 to mean, will be used later on
        std = np.hstack((np.ones(1), std))  # add 0 to std
        return x, mu, std

    def init_vals(self, n):
        theta = np.zeros(n, float)
        alpha = 0.01  # learning rate
        convergence = 0.0001
        return theta, alpha, convergence

    def sigmoid(self, z):  # sigmoid function
        sig = float(1)/(1+math.e**(-z))
        return sig

    def cost_function(self, x, y, theta):
        m = x.shape[0]
        linear = np.dot(x, theta)
        h = self.sigmoid(linear)
        step1 = np.dot(y.T, np.log(h))
        step2 = np.dot((1-y.T), np.log(1-h))
        cost = -(step1+step2)/m  # cost = sum{-y*log(h) - (1-y)*log(1-h)} / m
        return cost

    def gradient(self, x, y, theta):
        m = x.shape[0]
        linear = np.dot(x, theta)
        h = np.array(self.sigmoid(linear))
        diff = h - np.squeeze(y)
        delta = np.dot(x.T, diff)  # gradient = sum((h-y) * x)/m
        return delta/m

    def grad_descent(self, x, y, theta, alpha, convergence):
        new_cost = 0.0
        cost = 1.0
        cost_itr = []
        itr = 0
        while math.fabs(new_cost - cost) > convergence or itr <= 1500:  # run GD until cost difference < 0.0001
            cost = new_cost
            new_cost = self.cost_function(x, y, theta)
            cost_itr.append(new_cost)
            theta = theta - (alpha * self.gradient(x, y, theta))  # calculate new theta values simultaneously
            itr += 1
        return theta, np.array(cost_itr), itr

    def predict(self, x, y, theta, mu, std):
        m = x.shape[0]
        x = np.hstack(((np.ones((m, 1), float)), x))  # add 1's to x
        x = (x-mu)/std  # normalize
        linear = np.dot(x, theta)
        h = self.sigmoid(linear)
        predict = np.where(h >= 0.5, 1, 0)  # predict as 1 if probability >= 0.5
        return predict

    def calc_accuracy(self, predict, y):
        output = np.where(predict == y, 1, 0)
        accuracy = np.mean(output)
        return accuracy * 100

    def plot_cost(self, itr, cost):
        x_axis = range(itr)
        plt.interactive(False)
        plt.plot(x_axis, cost)
        plt.show()

    def plot_decision_boundary(self, x, theta):
        plt.interactive(False)
        plt.scatter(self.x_orig[:, 0], self.x_orig[:, 1], c=self.y_orig, cmap='viridis')
        plt_x = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]))
        plt_y = -(theta[0] + theta[1] * plt_x) / theta[2]
        plt.plot(plt_x, plt_y)
        plt.show()

    def log_reg(self):
        x_orig, y = self.read_data()
        x, mu, std = self.feature_normalize(x_orig)
        x, mu, std = self.padding(x, mu, std)
        theta, alpha, convergence = self.init_vals(x.shape[1])

        init_cost = self.cost_function(x, y, theta)
        init_grad = self.gradient(x, y, theta)

        print("init cost", init_cost)
        print("init grad", init_grad)

        theta, cost, itr = self.grad_descent(x, y, theta, alpha, convergence)

        print("learned theta", theta)
        print("iterations", itr)

        predict = self.predict(x_orig, y, theta, mu, std)
        accuracy = self.calc_accuracy(predict, y)
        print("accuracy", accuracy)

        self.plot_cost(itr, cost)


log = LogisticRegression()
log.log_reg()
