import numpy as np
import matplotlib.pyplot as plt


class Lab1:
    def __init__(self):
        # Table values
        # self.x = np.array([1, 2, 3, 4, 5, 6])
        # self.y = np.array([1, 1.5, 3, 4.5, 7, 8.5])
        self.x = np.array([3, 5, 7, 9, 11, 13])
        self.y = np.array([3.5, 4.4, 5.7, 6.1, 6.5, 7.3])
        # self.x = np.array([10, 20, 30, 40, 50, 60])
        # self.y = np.array([1.06, 1.33, 1.52, 1.68, 1.81, 1.91])
        # Mean values
        # self.x_mean = np.mean(self.x)
        # self.y_mean = np.mean(self.y)
        pass

    def linear_function(self, show_plot=True):
        # Create the matrix of the linear system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x, np.ones(len(self.x))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, self.y, rcond=None)[0]
        print('a: {}, b: {}', a, b)
        # Predict y
        y_pred = a * self.x + b
        # Plot the data and the approximation
        self.plot_result(y_pred, color='red', show_plot=show_plot)
        # Print comparison
        self.task_compare(y_pred, name='Linear')
        return y_pred

    def power_function(self, show_plot=True):
        # Calculate the logarithms of x and y
        x_log = np.log(self.x)
        y_log = np.log(self.y)
        # Create the matrix of the power system of equations
        # and solve it using least squares method
        matrix = np.vstack([x_log, np.ones(len(x_log))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, y_log, rcond=None)[0]
        # print(a, b)
        print('a: {}, b: {}', a, np.exp(b))
        # Predict y
        y_pred = np.exp(a * np.log(self.x) + b)
        # Plot the data and the approximation
        self.plot_result(y_pred, color='green', show_plot=show_plot)
        # Print comparison
        self.task_compare(y_pred, name='Power')
        return y_pred

    def exponential_function(self, show_plot=True):
        # Create the matrix of the exponential system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x, np.ones(len(self.x))]).T
        # Calculate a and b
        a, b = np.linalg.lstsq(matrix, np.log(self.y), rcond=None)[0]
        print('a: {}, b: {}', a, np.exp(b))
        # Predict y
        y_pred = np.exp(a * self.x + b)
        # Plot the data and the approximation
        self.plot_result(y_pred, color='orange', show_plot=show_plot)
        # Print comparison
        self.task_compare(y_pred, name='Exponential')
        return y_pred

    def quadratic_function(self, show_plot=True):
        # Create the matrix of the quadratic system of equations
        # and solve it using least squares method
        matrix = np.vstack([self.x ** 2, self.x, np.ones(len(self.x))]).T
        # Calculate a, b and c
        a, b, c = np.linalg.lstsq(matrix, self.y, rcond=None)[0]
        print('a: {}, b: {}, c: {}', a, b, c)
        print(a, b, c)
        # Predict y
        y_pred = a * self.x ** 2 + b * self.x + c
        # Plot the data and the approximation
        self.plot_result(y_pred, color='purple', show_plot=show_plot)
        # Print comparison
        self.task_compare(y_pred, name='Quadratic')
        return y_pred

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

    def compare_mse(self):
        # Calculate the mean square error
        mse_linear = np.mean((self.linear_function(False) - self.y) ** 2)
        mse_power = np.mean((self.power_function(False) - self.y) ** 2)
        mse_exponential = np.mean((self.exponential_function(False) - self.y) ** 2)
        mse_quadratic = np.mean((self.quadratic_function(False) - self.y) ** 2)
        # Print the results
        print('\nMean square error:')
        print('Linear: ', mse_linear)
        print('Power: ', mse_power)
        print('Exponential: ', mse_exponential)
        print('Quadratic: ', mse_quadratic)
        print('The best approximation is: ', min(mse_linear, mse_power, mse_exponential, mse_quadratic))

    def compare_mae(self):
        # Calculate the mean absolute error
        mae_linear = np.mean(np.abs(self.linear_function(False) - self.y))
        mae_power = np.mean(np.abs(self.power_function(False) - self.y))
        mae_exponential = np.mean(np.abs(self.exponential_function(False) - self.y))
        mae_quadratic = np.mean(np.abs(self.quadratic_function(False) - self.y))
        # Print the results
        print('\nMean absolute error:')
        print('Linear: ', mae_linear)
        print('Power: ', mae_power)
        print('Exponential: ', mae_exponential)
        print('Quadratic: ', mae_quadratic)
        print('The best approximation is: ', min(mae_linear, mae_power, mae_exponential, mae_quadratic))

    def compare_mse_mae(self):
        self.compare_mse()
        self.compare_mae()

    def task_compare(self, y_pred, name):
        result = 0
        for i in range(len(self.y)):
            result += (self.y[i] - y_pred[i]) ** 2
        print('{} = {}', name, result)
