
#def twoDimGaussian():
#    import matplotlib.pyplot as plt
#    import numpy as np
#    from mpl_toolkits.mplot3d import Axes3D

#    figure = plt.figure()
#    ax = Axes3D(figure)

#    theta = 2
#    x = np.arange(-10, 10, 0.25)
#    y = np.arange(-10, 10, 0.25)
#    x, y = np.meshgrid(x, y)
#    coeffient = 1 / (2 * np.pi * np.power(theta, 2))
#    power_index = -(np.power(x, 2) + np.power(y, 2)) / np.power(theta, 2)
#    z = coeffient * np.exp(power_index)
#    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
#    plt.show()