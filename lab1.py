import torch
import numpy as np
import matplotlib.pyplot as plt


def task1(): 
    """
    Task 1 of Lab 1 for COMP3710.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # grid for computing image, subdivide the space
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    
    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    
    # transfer to the GPU device - Hah I wish (CPU)
    x = x.to(device)
    y = y.to(device)
    
    # Compute Gaussian
    gaussian = torch.exp(-(x**2+y**2)/2.0)
    # Compute 2D sine function
    sin1 = torch.sin(x*y) # Should it be this?
    sin2 = torch.sin(x+y) # Or this? - got both in case - this one is Gabor filter?
    
    z = gaussian * sin2 # can change which sin1/2 is used by setting it to equal z.

    
    plt.imshow(z.cpu().numpy())
    plt.tight_layout()
    plt.show()
    
def processFractal(a):
    """
    Display an array of iteration counts as a colorful picture of a fractal.
    """
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
        30+50*np.sin(a_cyclic), 53155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

    
def task2():
    """
    Task2 of Lab 1 for COMP3710. 
    Uncomment/comment different sections to switch configurations and fractal types.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    # Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005] # Default resolution
    # Y, X = np.mgrid[-1.3:1.3:0.001, -2:1:0.001] # Higher resolution
    # Y, X = np.mgrid[-1:1:0.002, -1.5:0.5:0.002] # Slightly hihger res
    # Y, X = np.mgrid[-0.8:-0.5:0.0001, -0.05:0.15:0.0001] # Higher res + zoom
    # Y, X = np.mgrid[-1.3/2.5:1.3/2.5:0.001, -2/2.5:1/2.5:0.001] # Zoom and high res
    Y, X = np.mgrid[-1.5:1.5:0.001, -1.5:1.5:0.001] # Julie set range
    
    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y) #important!
    zs = z.clone()
    ns = torch.zeros_like(z)
    # c = torch.tensor(-0.6 - 0.4j) # Julia set figures 1 and 2
    c = torch.tensor(-0.1 + 0.9j) # figure 3
    
    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)
    c = c.to(device) # Julia set stuff
    
    # Mandelbrot Set
    # for i in range(200):
    #     # Compute the new values of z: z^2 + x
    #     zs_ = zs*zs + z
    #     # Have we diverged with this new value?
    #     not_diverged = torch.abs(zs_) < 4.0
    #     # Update variables to compute
    #     ns += not_diverged
    #     zs = zs_
        
    # Julia set calc
    for i in range(200):
        zs_ = zs*zs + c  # Use constant c
        not_diverged = torch.abs(zs_) < 4.0
        ns += not_diverged
        zs = zs_
        
    # plot
    fig = plt.figure(figsize=(16,10))
    
    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


def task3():
    """
    Task 3 for Lab 1 of COMP3710.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # grid for computing image, subdivide the space
    Y, X = np.mgrid[-1:1:0.002, -1:1:0.002] # 1-1 test with range for now? or normalise coords later?

    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y) # needed?
    
    # transfer to the GPU device
    x = x.to(device)
    y = y.to(device)
    
    # Dragon curve fractal functions - https://en.wikipedia.org/wiki/Dragon_curve
    def f1(z):
        return ((1 + 1j) * z) / 2
    
    def f2(z):
        return (1 - (1 - 1j) * z) / 2
    
    
    
    
def main():
    # task1()
    # task2()
    task3()
    
if __name__ == "__main__":
    main()
