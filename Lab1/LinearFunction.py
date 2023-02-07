import numpy as np
import matplotlib.pyplot as plt


class Lab1:
    def __init__(self):
        # Table values
        self.x = np.array([3, 5, 7, 9, 11, 13])
        self.y = np.array([3.5, 4.4, 5.7, 6.1, 6.5, 7.3])

        # Mean values
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        pass

    def linear_function(self, show_plot=True):
        def find_a_b():
            # Calculate a and b
            a = np.sum((self.x - self.x_mean) * (self.y - self.y_mean)) / np.sum((self.x - self.x_mean) ** 2)
            b = self.y_mean - a * self.x_mean
            return a, b

        def predict_y():
            a, b = find_a_b()
            return a * self.x + b

        # Predict y
        y_pred = predict_y()
        # Plot the data and the approximation
        self.plot_result(y_pred, color='red', show_plot=show_plot)

    def power_function(self, show_plot=True):
        def find_a_b():
            # Calculate a and b
            n = len(self.x)
            log_x = np.log(self.x)
            log_y = np.log(self.y)
            log_x_sum = np.sum(log_x)
            log_y_sum = np.sum(log_y)
            log_x_squared_sum = np.sum(log_x ** 2)
            log_x_log_y_sum = np.sum(log_x * log_y)
            a = (n * log_x_log_y_sum - log_x_sum * log_y_sum) / (n * log_x_squared_sum - log_x_sum ** 2)
            b = (log_y_sum - a * log_x_sum) / n
            return a, b

        def predict_y():
            a, b = find_a_b()
            return np.exp(b) * self.x ** a

        # Predict y
        y_pred = predict_y()
        # Plot the data and the approximation
        self.plot_result(y_pred, color='green', show_plot=show_plot)

    def exponential_function(self, show_plot=True):
        def find_a_b():
            # Find the coefficients for exponential function using least squares method
            coefficients = np.polyfit(self.x, np.log(self.y), 1)
            a = np.exp(coefficients[1])
            b = coefficients[0]

            # Round the coefficients to 0.01
            a = round(a, 2)
            b = round(b, 2)
            return a, b

        def predict_y():
            a, b = find_a_b()
            return a * np.exp(b * self.x)

        # Predict y
        y_pred = predict_y()
        # Plot the data and the approximation
        self.plot_result(y_pred, color='orange', show_plot=show_plot)

    def quadratic_approximation(self, show_plot=True):
        def find_a_b_c():
            # Find the coefficients for quadratic function using least squares method
            a, b, c = np.polyfit(self.x, self.y, 2)

            # Round the coefficients to 0.01
            a = round(a, 2)
            b = round(b, 2)
            c = round(c, 2)
            return a, b, c

        def predict_y():
            a, b, c = find_a_b_c()
            return a * self.x ** 2 + b * self.x + c

        # Predict y
        y_pred = predict_y()
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
        self.quadratic_approximation(False)
        self.plot_show_dots()
