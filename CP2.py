import matplotlib.pyplot as plt
import cv2
import numpy as np

def convert_to_grayscale(path):
    """
    Convert an image to grayscale and scale it to 256x256 if larger.
    
    Args:
        path (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The grayscale image.
    """
    img = cv2.imread(path)
    
    # Scale the image to 256x256 if larger
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv2.resize(img, (256, 256))
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def add_noise(img, loc, scale):
    """
    Add random noise to an image.
    
    Args:
        img (numpy.ndarray): The input image.
        loc (float): The mean of the normal distribution used to generate the random noise.
        scale (float): The standard deviation of the normal distribution used to generate the random noise.
        
    Returns:
        numpy.ndarray: The image with added random noise.
    """
    img_with_noise = img + np.random.normal(loc, scale, img.shape)
    return img_with_noise

def plot_original_image(img, axs):
    """
    Plot the original image.

    Parameters:
    img (numpy.ndarray): The original image.
    axs (numpy.ndarray): The axes to plot the image on.
    """
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

def regular_blur(img, kernel_size, sigma):
    """
    Apply regular Gaussian blur to the input image.

    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: The blurred image.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def rungeKutta_method(img, k, lambda_, h):
    """
    Applies the Runge-Kutta method to perform numerical integration for Perona-Malik diffusion.

    Args:
        img (numpy.ndarray): The input image.
        k (float): The diffusion coefficient.
        lambda_ (float): The edge enhancement parameter.
        h (float): The step size for numerical integration.

    Returns:
        numpy.ndarray: The image after applying the Runge-Kutta method for diffusion.
    """
    k1 = h * perona_malik_diffusion(img, k, lambda_)
    k2 = h * perona_malik_diffusion(img + 0.5 * k1, k, lambda_)
    k3 = h * perona_malik_diffusion(img + 0.5 * k2, k, lambda_)
    k4 = h * perona_malik_diffusion(img + k3, k, lambda_)
    img = img + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return img
    
def adamsBashforth_method(img, prev_derivative, k, lambda_, h):
    """
    Applies the Adams-Bashforth method to perform image processing using Perona-Malik diffusion.

    Args:
        img (ndarray): The input image.
        prev_derivative (ndarray): The derivative of the previous image.
        k (float): The diffusion coefficient.
        lambda_ (float): The edge enhancement parameter.
        h (float): The time step size.

    Returns:
        tuple: A tuple containing the updated image and the current derivative.
    """
    current_derivative = perona_malik_diffusion(img, k, lambda_)
    img = img + h * (1.5 * current_derivative - 0.5 * prev_derivative)

    return img, current_derivative
    

def psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
    img1 (numpy.ndarray): The first image.
    img2 (numpy.ndarray): The second image.

    Returns:
    float: The PSNR value.

    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def get_sobel_x(size_n):
    """
    Calculate the custom Sobel filter for the x gradient.

    Args:
        size_n (int): The size of the filter.

    Returns:
        numpy.ndarray: The custom Sobel filter for the x gradient.
    """
    custom_sobel = np.ndarray((size_n,size_n))
    for i in range(size_n):
        for j in range(size_n):
            if j != size_n//2:
                custom_sobel[i,j] = (j-size_n//2)/((i-size_n//2)**2 + (j-size_n//2)**2)
            else:
                custom_sobel[i,j] = 0
    return custom_sobel

def get_sobel_y(size_n):
    """
    Returns a custom Sobel filter for calculating the y gradient.

    Parameters:
    size_n (int): The size of the filter matrix.

    Returns:
    numpy.ndarray: The custom Sobel filter for the y gradient.
    """
    custom_sobel = np.ndarray((size_n,size_n))
    for i in range(size_n):
        for j in range(size_n):
            if i != size_n//2:
                custom_sobel[i,j] = (i-size_n//2)/((i-size_n//2)**2 + (j-size_n//2)**2)
            else:
                custom_sobel[i,j] = 0
    return -1*custom_sobel

def perona_malik_diffusion(img, k, lambda_):
    """
    Apply Perona-Malik diffusion to the input image.

    Args:
        img (numpy.ndarray): Input image.
        k (float): Diffusion coefficient.
        lambda_ (float): Time step.

    Returns:
        numpy.ndarray: Diffused image.
    """
    img = img.astype(np.float32)

    # Compute gradients in x and y directions
    grad_x = cv2.filter2D(src=img, ddepth=-1, kernel=get_sobel_x(3)) # this is more hand written than the below
    grad_y = cv2.filter2D(src=img, ddepth=-1, kernel=get_sobel_y(3)) # this is more hand written than the below
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1) # comment this to use the above
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1) # comment this to use the above
        
    # Compute the magnitude of the gradient
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Compute the diffusion coefficient
    c = np.exp(-(grad_mag / k)**2) # charbonnier function

    
    # Compute divergence of the gradient
    div_x = cv2.filter2D(src=c * grad_x, ddepth=-1, kernel=get_sobel_x(3))  # this is more hand written than the below
    div_y = cv2.filter2D(src=c * grad_y, ddepth=-1, kernel=get_sobel_y(3))  # this is more hand written than the below
    div_x = cv2.Sobel(c * grad_x, cv2.CV_32F, 1, 0, ksize=1)  # comment this to use the above
    div_y = cv2.Sobel(c * grad_y, cv2.CV_32F, 0, 1, ksize=1)  # comment this to use the above
    div = div_x + div_y
    
    # div = divergence(c, grad_x, grad_y)

    # Apply boundary conditions
    div[0, :] = 0  # Top boundary
    div[-1, :] = 0  # Bottom boundary
    div[:, 0] = 0  # Left boundary
    div[:, -1] = 0  # Right boundary

    # Update the image using the diffusion equation
    img = img + lambda_ * div

    return img





def plot_images(noisyimg, grayimg, k, lambda_, h, iterations):
    """
    Plots a series of images and their truncation errors during an iterative process.

    Parameters:
    noisyimg (numpy.ndarray): The original image.
    grayimg (numpy.ndarray): The grayscale version of the original image.
    k (float): The diffusion coefficient.
    lambda_ (float): The edge enhancement parameter.
    h (float): The time step size.
    iterations (int): The number of iterations.
    
    """
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))  # Create a figure and a set of subplots

    # Original Image
    plot_original_image(noisyimg, axs)
    rungeKutta_image = noisyimg.copy()
    adamsBashforth_image = noisyimg.copy()
    blured_image = noisyimg.copy()

    rungeKutta_errors = []
    adamsBashforth_errors = []
    iteration_steps = [] 

    prev_derivative = perona_malik_diffusion(noisyimg, k, lambda_)  # For Adams-Bashforth

    
    for i in range(iterations + 1):
        if not plt.fignum_exists(fig.number):
            break  
        
        rungeKutta_image = rungeKutta_method(rungeKutta_image, k, lambda_, h)
        adamsBashforth_image, prev_derivative = adamsBashforth_method(adamsBashforth_image, prev_derivative, k, lambda_, h)
        blured_image = regular_blur(blured_image, 5, 0.5)
        

        if i % 10 == 0:
            rungeKutta_error = psnr(grayimg, rungeKutta_image)
            adamsBashforth_error = psnr(grayimg, adamsBashforth_image)
            rungeKutta_errors.append(rungeKutta_error)
            adamsBashforth_errors.append(adamsBashforth_error)
            iteration_steps.append(i)

            # Update Perona-Malik Image
            axs[0, 1].imshow(rungeKutta_image, cmap='gray')
            axs[0, 1].set_title(f'Runge Kutta - Iteration {i}')
            axs[0, 1].axis('off')
            # Update Adams-Bashforth Image
            axs[0, 2].imshow(adamsBashforth_image, cmap='gray')
            axs[0, 2].set_title(f'Adams-Bashforth - Iteration {i}')
            axs[0, 2].axis('off')
            # Update Regular Blur Image
            axs[1, 0].imshow(blured_image, cmap='gray')
            axs[1, 0].set_title(f'Regular Blur - Iteration {i}')
            axs[1, 0].axis('off')
            
            
            axs[1, 1].plot(iteration_steps, rungeKutta_errors, color='blue', linewidth=0.5, marker='o', markersize=2)
            axs[1, 1].set_title('Runge Kutta Truncation Error')
            axs[1, 1].set_xlabel('Iteration')
            axs[1, 1].grid(True)

            axs[1, 2].plot(iteration_steps, adamsBashforth_errors, color='green', linewidth=0.5, marker='o', markersize=2)
            axs[1, 2].set_title('Adams-Bashforth Truncation Error')
            axs[1, 2].set_xlabel('Iteration')
            axs[1, 2].grid(True)
            plt.draw()
            plt.pause(0.03)  # Pause briefly to allow the plot to update

    plt.show()
    print(f'Runge-Kutta Truncation Error: {rungeKutta_errors[-1]}')
    print(f'Adams-Bashforth Truncation Error: {adamsBashforth_errors[-1]}')

# Example usage
k = 100  # Edge threshold parameter
lambda_ = 0.5  # Time step
h = 0.005  # Larger step size
grayImage = convert_to_grayscale('./4.1.08.tiff')
noisyImage = add_noise(grayImage, 0, 20)
plot_images(noisyImage, grayImage, k, lambda_, h=h, iterations=500)
