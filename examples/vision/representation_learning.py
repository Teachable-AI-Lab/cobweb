import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from tqdm import tqdm

from cobweb.cobweb_continuous import CobwebContinuousTree
from cobweb.visualize import visualize

rng = np.random.default_rng()


def plot_centroids(centroids, flat_images, patch_size):
    n_clusters = centroids.shape[0]
    width = int(np.sqrt(n_clusters))
    if n_clusters != width * width:
        width += 1
    pca = PCA(n_components=2)
    pca_images = pca.fit_transform(flat_images)
    pca_centroids = pca.transform(centroids)

    # Plotting
    fig = plt.figure(figsize=(16, 6))
    # grid = plt.GridSpec(patch_size, 2*patch_size, figure=fig)
    grid = plt.GridSpec(
        width, 2 * width, figure=fig, wspace=0.0, hspace=0.01
    )  # Adjusted spacing

    # Scatter plot of the first principal component of centroids
    ax_scatter = fig.add_subplot(grid[0:width, 0:width])
    ax_scatter.scatter(pca_images[:, 0], pca_images[:, 1], alpha=0.2, color="blue")
    ax_scatter.scatter(
        pca_centroids[:, 0], pca_centroids[:, 1], marker="x", color="red", s=100
    )
    ax_scatter.set_title("Data and Centroids using the first two principal components")
    ax_scatter.grid(True)

    # Transform centroids back to original space for visualization as images
    # centroids_original = pca.inverse_transform(centroids_pca) * np.sqrt(std*std + 10) + mean

    # Creating a grid of subplots for centroid images on the right side
    for i in range(n_clusters):
        # if i >= patch_size * patch_size:
        #     break
        ax_image = fig.add_subplot(
            grid[i % width, width + i // width]
        )  # Create subplots in the last two columns
        # ax_image.imshow(centroids_original[i].reshape(patch_size, patch_size, 3))
        # ax_image.imshow(centroids_pca[i].reshape(patch_size, patch_size, 1))
        ax_image.imshow(centroids[i].reshape(patch_size, patch_size), cmap="gray")

        ax_image.axis("off")

    plt.show()


def plot_images(X, patch_size, n_channels):
    # Reshape the data to (n_samples, patch_size, patch_size, n_channels)
    images = X.reshape(-1, patch_size, patch_size, n_channels)

    # Randomly select 15 images
    num_images_to_select = 15

    # Use the generator to select indices
    selected_indices = rng.choice(images.shape[0], num_images_to_select, replace=False)
    selected_images = images[selected_indices]

    # Plot the selected images in a grid
    fig, axes = plt.subplots(3, 5, figsize=(5, 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if n_channels == 1:
            ax.imshow(selected_images[i], cmap="grey")
        else:
            ax.imshow(selected_images[i])
        ax.axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()


def zca_whitening(images, epsilon=1e-5):
    # Step 1: Flatten images
    original_shape = images.shape  # Save the original shape of images
    images_flattened = images.reshape(
        images.shape[0], -1
    )  # Reshape to (n_samples, width*height)

    # Step 2: Mean and Covariance
    mean_image = np.mean(images_flattened, axis=0)
    images_centered = images_flattened - mean_image
    covariance_matrix = np.cov(
        images_centered, rowvar=False
    )  # Use 'rowvar=False' for (features, samples)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = linalg.eigh(covariance_matrix)

    # Step 4: Construct the whitening matrix
    whitening_matrix = (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
    )

    # Step 5: Apply the whitening transformation and return
    return (images_centered @ whitening_matrix).reshape(original_shape)


if __name__ == "__main__":
    # Define the transformation to convert the images to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts images to PyTorch tensors
        ]
    )

    # Load the MNIST training dataset
    mnist_train = datasets.MNIST(
        root="./datasets/MNIST", train=True, download=True, transform=transform
    )

    # Convert the dataset to NumPy arrays
    train_images = mnist_train.data.numpy()
    train_labels = mnist_train.targets.numpy()

    # For the test dataset (if needed)
    mnist_test = datasets.MNIST(
        root="./datasets/MNIST", train=False, download=True, transform=transform
    )
    test_images = mnist_test.data.numpy()
    test_labels = mnist_test.targets.numpy()

    # Shuffle the data
    indices = rng.permutation(len(train_images))
    train_images = train_images[indices]
    train_labels = train_labels[indices]
    # plot_images(train_images, 28, 1)

    # Get Sample
    n_images = 60000
    train_images = 1.0 * train_images[:n_images]

    # Normalize the images
    mean = train_images.mean(axis=0)
    std = train_images.std(axis=0)
    train_images = (train_images - mean) / np.sqrt(std * std + 10)
    # plot_images(train_images, 28, 1)

    # Whiten the images
    train_images = zca_whitening(train_images)
    # plot_images(train_images, 28, 1)

    # Flatten the data
    flat_images = train_images.reshape(train_images.shape[0], -1)

    # Norm the data
    norms = np.linalg.norm(flat_images, axis=1, keepdims=True)
    flat_images /= norms

    # Build the cobweb tree
    t = CobwebContinuousTree(flat_images.shape[1])
    # t = CobwebTorchTree(flat_images.shape[1:])

    for i in tqdm(range(flat_images.shape[0])):
        t.ifit(flat_images[i])
        # t.ifit(torch.from_numpy(flat_images[i]))

    centroids = []
    queue = [t.root]
    max_clusters = 64
    while len(queue) > 0:
        curr = queue.pop(0)
        for child in curr.children:
            if len(centroids) >= max_clusters:
                break
            centroids.append(child.mean)
            queue.append(child)
        # break

        # for c2 in child.children:
        #     centroids.append(c2.mean - child.mean)
    centroids = np.stack(centroids)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= norms

    # Render the learned filters
    plot_centroids(centroids, flat_images[:n_images], 28)

    visualize(t)
