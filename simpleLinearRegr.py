import numpy as np
import matplotlib.pyplot as plt

# calculate the coefficients:b_0, b_1
def estimate_coef(x,y):
    n = np.size(x)

    m_x,m_y = np.mean(x),np.mean(y)

    SSxy = np.sum(x*y) - n*m_x*m_y
    SSxx = np.sum(x*x) - n*m_x*m_x

    b_1 = SSxy/SSxx
    b_0 = m_y - m_x*b_1

    return (b_0,b_1)

def plot_regression_line(x, y, b):
    # scatters
    plt.scatter(x, y, color='m', marker='o', s=30)#what is 's'?

    # generate the regression line
    y_pred = b[0]+b[1]*x

    # plot the line
    plt.plot(x, y_pred, color='g')
    plt.xlabel('x')
    plt.ylabel('y')

    # show
    plt.show()

def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 

    b = estimate_coef(x,y)

    print("Estinmate Coefficients:\
        \nb_0:{} \nb_1:{}".format(b[0],b[1]))
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()