import pandas as pd
import matplotlib.pyplot as plt

class RegresionLineal:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.sales = self.data['Sales(million Euro)']
        self.advertising = self.data['advertising (million euro)']
        self.mean_x = self.advertising.mean()
        self.mean_y = self.sales.mean()
        self.diff_x = self.advertising - self.mean_x
        self.diff_y = self.sales - self.mean_y
        self.B1 = (self.diff_x * self.diff_y).sum() / (self.diff_x**2).sum()
        self.B0 = self.mean_y - self.B1 * self.mean_x

    def predict_sales(self, advertising_value):
        return self.B0 + self.B1 * advertising_value

    def plot_regression(self):
        plt.scatter(self.advertising, self.sales, label='Datos de Ventas')
        regression_line = self.B0 + self.B1 * self.advertising
        plt.plot(self.advertising, regression_line, color='red', label='Línea de Regresión')
        plt.xlabel('Advertising (millones de euros)')
        plt.ylabel('Sales (millones de euros)')
        plt.legend()
        plt.title('Regresión Lineal de Sales vs Advertising')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    data_path = "C:\\CaseBenettonLinealRegretion.csv"
    regresion = RegresionLineal(data_path)
    print(f"Ecuación de Regresión: ŷ = {regresion.B0} + {regresion.B1}x")

    advertising_values = [57, 58, 60, 63, 70]
    for value in advertising_values:
        predicted_sales = regresion.predict_sales(value)
        print(f"Para Advertising = {value} millones de euros, Sales = {predicted_sales} millones de euros")

    regresion.plot_regression()
