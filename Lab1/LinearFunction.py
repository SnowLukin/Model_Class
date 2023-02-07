import numpy as np
import matplotlib.pyplot as plt


class Lab1:
    def __init__(self):
        # Table values
        self.x = np.array([3, 5, 7, 9, 11, 13])
        self.y = np.array([3.5, 4.4, 5.7, 6.1, 6.5, 7.3])
        # self.x = np.array([10, 20, 30, 40, 50, 60])
        # self.y = np.array([1.06, 1.33, 1.52, 1.68, 1.81, 1.91])
        # Mean values
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        pass

    def linear_function(self, show_plot=True):
        # Create the matrix of the linear system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x, np.ones(len(self.x))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, self.y, rcond=None)[0]
        # Predict y
        y_pred = a * self.x + b
        # Plot the data and the approximation
        self.plot_result(y_pred, color='red', show_plot=show_plot)

    def power_function(self, show_plot=True):
        # Calculate the logarithms of x and y
        x_log = np.log(self.x)
        y_log = np.log(self.y)
        # Create the matrix of the power system of equations
        # and solve it using least squares method
        matrix = np.vstack([x_log, np.ones(len(x_log))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, y_log, rcond=None)[0]
        # Predict y
        y_pred = np.exp(a * np.log(self.x) + b)
        # Plot the data and the approximation
        self.plot_result(y_pred, color='green', show_plot=show_plot)

    def exponential_function(self, show_plot=True):
        # Create the matrix of the exponential system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x, np.ones(len(self.x))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, np.log(self.y), rcond=None)[0]
        # Predict y
        y_pred = np.exp(a * self.x + b)
        # Plot the data and the approximation
        self.plot_result(y_pred, color='orange', show_plot=show_plot)

    def quadratic_function(self, show_plot=True):
        # Create the matrix of the quadratic system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x ** 2, self.x, np.ones(len(self.x))]).T
        # Calculate a, b and c
        a, b, c = np.linalg.lstsq(matrix, self.y, rcond=None)[0]
        # Predict y
        y_pred = a * self.x ** 2 + b * self.x + c
        # Plot the data and the approximation
        self.plot_result(y_pred, color='purple', show_plot=show_plot)

    def plot_result(self, y_pred, color='red', show_plot=True):
        # Plot the data and the approximation
        plt.plot(self.x, y_pred, color=color)
        if show_plot:
            self.plot_show_dots()

    def plot_show_dots(self):
        plt.scatter(self.x, self.y, color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Least squares approximation')
        plt.legend(['Linear', 'Power', 'Exponential', 'Quadratic', 'Data'])
        plt.show()

    def plot_all_results(self):
        self.linear_function(False)
        self.power_function(False)
        self.exponential_function(False)
        self.quadratic_function(False)
        self.plot_show_dots()
