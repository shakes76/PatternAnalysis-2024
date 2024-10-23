"""A file with helper methods like plotting"""
import matplotlib.pyplot as plt

def save_plot(train, validation, num_epochs, save_to):
    
    x = [i+1 for i in range(num_epochs)]
    plt.plot(x, train)
    plt.plot(x, validation)

    plt.savefig(save_to)
    plt.clf()