import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import glob
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  # For progress bars

# --- Configuration (to be set by user prompts later) ---
TARGET_NET_DIM = 256  # Example, will be prompted
CHANNELS = 3  # Example, will be prompted
LATENT_DIM_FACTOR = 4  # Example, will be prompted
EPOCHS = 50
BATCH_SIZE = 32  # Adjusted for potentially larger images
MODEL_SAVE_PATH_TEMPLATE = "cae_{}x{}_lf{}.weights.h5"
COMPRESSED_FILE_EXTENSION = ".npz"


# --- 1. Define the Autoencoder Model ---
def build_autoencoder(input_shape, latent_dim_channels):
    # --- Encoder ---
    input_img = Input(shape=input_shape)  # e.g., (256, 256, 3)

    # Block 1
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)  # input_shape/2

    # Block 2
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # input_shape/4

    # Bottleneck / Latent representation
    encoded = Conv2D(
        latent_dim_channels,
        (3, 3),
        activation="relu",
        padding="same",
        name="bottleneck_conv",
    )(x)

    # --- Decoder ---
    # Latent representation shape is (input_shape[0]//4, input_shape[1]//4, latent_dim_channels)

    # Block 1
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same")(
        encoded
    )
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    # Block 2
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu", padding="same")(
        x
    )
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    # Output layer
    decoded = Conv2D(input_shape[2], (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded, name="Autoencoder")
    encoder = Model(input_img, encoded, name="Encoder")

    encoded_input_shape = (
        input_shape[0] // 4,
        input_shape[1] // 4,
        latent_dim_channels,
    )
    encoded_input = Input(shape=encoded_input_shape, name="encoded_input")

    # Reuse decoder layers for standalone decoder
    # Find layers by name or rely on fixed indexing if model structure is stable
    # Using indexing from the end of the autoencoder model for decoder part
    # autoencoder.layers are: Input, C,P,C,P,BottleneckC, CT,C,CT,C,OutputC
    # -1 OutputC, -2 C, -3 CT, -4 C, -5 CT
    deco_x = autoencoder.layers[-5](
        encoded_input
    )  # First Conv2DTranspose from decoder part
    deco_x = autoencoder.layers[-4](deco_x)  # Conv2D
    deco_x = autoencoder.layers[-3](deco_x)  # Second Conv2DTranspose
    deco_x = autoencoder.layers[-2](deco_x)  # Conv2D
    decoded_output = autoencoder.layers[-1](deco_x)  # Output Conv2D

    decoder = Model(encoded_input, decoded_output, name="Decoder")

    return autoencoder, encoder, decoder


# --- 2. Load and Preprocess Data ---
def get_network_ready_image(image_path_or_pil, target_net_dims, channels):
    """Loads, resizes, and normalizes a single image for network input."""
    try:
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil)
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil
        else:
            raise ValueError("Input must be a file path or PIL Image object.")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_or_pil}")
        return None, None
    except UnidentifiedImageError:
        print(
            f"Error: Cannot identify image file {image_path_or_pil}. It may be corrupted or not an image."
        )
        return None, None
    except Exception as e:
        print(f"Error loading image {image_path_or_pil}: {e}")
        return None, None

    original_wh = img.size  # (width, height)

    mode = "RGB" if channels == 3 else "L"
    if img.mode != mode:
        img = img.convert(mode)

    img_resized_for_network = img.resize(target_net_dims, Image.LANCZOS)

    img_array = np.array(img_resized_for_network).astype("float32") / 255.0
    if channels == 1 and len(img_array.shape) == 2:  # Grayscale needs channel dim
        img_array = np.expand_dims(img_array, axis=-1)
    elif channels == 3 and len(img_array.shape) == 2:  # Mishap with mode conversion?
        print(
            f"Warning: Image {image_path_or_pil} loaded as 2D despite CHANNELS=3. Converting to RGB again."
        )
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Ensure the final array has the correct number of channels
    if img_array.shape[-1] != channels:
        if channels == 1 and img_array.shape[-1] == 3:  # RGB to Grayscale
            # Basic grayscale conversion if PIL didn't handle it as expected
            img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            img_array = np.expand_dims(img_array, axis=-1)
        elif channels == 3 and img_array.shape[-1] == 1:  # Grayscale to RGB
            img_array = np.concatenate([img_array] * 3, axis=-1)
        else:
            print(
                f"Warning: Mismatch in expected channels ({channels}) and loaded image channels ({img_array.shape[-1]}) for {image_path_or_pil}."
            )
            # Fallback: if expecting 3 channels and got 4 (RGBA), take RGB
            if channels == 3 and img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            # If still not matching, this could be an issue
            if img_array.shape[-1] != channels:
                print(f"Error: Could not conform image to {channels} channels.")
                return None, None

    return img_array, original_wh


def load_dataset_from_directory(
    dir_path, target_net_dims, channels, test_split_ratio=0.2
):
    """Loads all images from a directory, prepares them for the network."""
    image_paths = []
    supported_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff")
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))

    if not image_paths:
        print(f"No images found in {dir_path} with supported extensions.")
        return None, None, None  # x_train, x_val, paths

    dataset_arrays = []
    dataset_full_paths = []

    print(
        f"Loading images from {dir_path} and preparing for network ({target_net_dims[0]}x{target_net_dims[1]})..."
    )
    for path in tqdm(image_paths):
        img_array, _ = get_network_ready_image(path, target_net_dims, channels)
        if img_array is not None:
            dataset_arrays.append(img_array)
            dataset_full_paths.append(
                path
            )  # Keep track of path for potential later use

    if not dataset_arrays:
        print("No images could be loaded successfully.")
        return None, None, None

    dataset_arrays = np.array(dataset_arrays)
    indices = np.arange(dataset_arrays.shape[0])
    np.random.shuffle(indices)
    dataset_arrays = dataset_arrays[indices]
    # dataset_full_paths = [dataset_full_paths[i] for i in indices] # Shuffle paths accordingly

    if test_split_ratio > 0:
        split_idx = int(len(dataset_arrays) * (1 - test_split_ratio))
        x_train = dataset_arrays[:split_idx]
        x_val = dataset_arrays[split_idx:]
        # train_paths = dataset_full_paths[:split_idx]
        # val_paths = dataset_full_paths[split_idx:]
        print(
            f"Loaded {len(dataset_arrays)} images. Training: {len(x_train)}, Validation: {len(x_val)}"
        )
        return (
            x_train,
            x_val,
            dataset_full_paths,
        )  # Returning all paths, split can be inferred
    else:
        print(
            f"Loaded {len(dataset_arrays)} images for training (no validation split)."
        )
        return dataset_arrays, None, dataset_full_paths


def load_cifar10_data(target_net_dims, channels):
    print("Loading and preprocessing CIFAR-10 data...")
    (x_train, _), (x_test, _) = cifar10.load_data()

    # Normalize images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Resize CIFAR-10 if target_net_dims are different (and handle channels)
    if x_train.shape[1:3] != target_net_dims or x_train.shape[3] != channels:
        print(
            f"Resizing CIFAR-10 images to {target_net_dims} and {channels} channels..."
        )
        x_train_resized = []
        for img_array in tqdm(x_train):
            pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
            # Use get_network_ready_image to handle resizing and channel conversion
            processed_img_array, _ = get_network_ready_image(
                pil_img, target_net_dims, channels
            )
            if processed_img_array is not None:
                x_train_resized.append(processed_img_array)
        x_train = np.array(x_train_resized)

        x_test_resized = []
        for img_array in tqdm(x_test):
            pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
            processed_img_array, _ = get_network_ready_image(
                pil_img, target_net_dims, channels
            )
            if processed_img_array is not None:
                x_test_resized.append(processed_img_array)
        x_test = np.array(x_test_resized)

    print(f"CIFAR-10 x_train shape: {x_train.shape}")
    print(f"CIFAR-10 x_test shape: {x_test.shape}")
    return x_train, x_test


# --- 3. Training ---
def train_model(
    autoencoder_model,
    x_train_data,
    x_val_data,
    current_epochs,
    current_batch_size,
    model_path,
):
    autoencoder_model.compile(optimizer="adam", loss="mse")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-6
        ),
    ]

    print("Training autoencoder...")
    validation_data_param = (
        (x_val_data, x_val_data)
        if x_val_data is not None and len(x_val_data) > 0
        else None
    )

    history = autoencoder_model.fit(
        x_train_data,
        x_train_data,
        epochs=current_epochs,
        batch_size=current_batch_size,
        shuffle=True,
        validation_data=validation_data_param,
        callbacks=callbacks,
        verbose=1,
    )

    autoencoder_model.save_weights(model_path)
    print(f"Model weights saved to {model_path}")
    return history


# --- 4. Compression and Decompression ---
def compress_image(
    encoder_model, image_path, output_npz_path, target_net_dims, channels
):
    print(f"Compressing {image_path}...")
    img_net_array, original_wh = get_network_ready_image(
        image_path, target_net_dims, channels
    )
    if img_net_array is None:
        return False

    # Add batch dimension
    img_net_array_batch = np.expand_dims(img_net_array, axis=0)
    latent_representation = encoder_model.predict(img_net_array_batch)

    np.savez_compressed(
        output_npz_path, latent=latent_representation, original_shape_wh=original_wh
    )
    print(f"Compressed representation saved to {output_npz_path}")

    try:
        original_size_bytes = os.path.getsize(image_path)
        compressed_size_bytes = os.path.getsize(output_npz_path)
        if compressed_size_bytes > 0:
            ratio = original_size_bytes / compressed_size_bytes
            print(f"Original file size: {original_size_bytes / 1024:.2f} KB")
            print(
                f"Compressed file size: {compressed_size_bytes / 1024:.2f} KB (latent data + metadata)"
            )
            print(f"File Size Compression Ratio: {ratio:.2f}x")
        else:
            print(
                "Could not calculate file size compression ratio (compressed file size is 0)."
            )
    except Exception as e:
        print(f"Could not calculate file sizes: {e}")
    return True


def decompress_image(decoder_model, compressed_npz_path, output_image_path):
    print(f"Decompressing {compressed_npz_path}...")
    try:
        data = np.load(compressed_npz_path)
        latent_representation = data["latent"]
        original_wh = data["original_shape_wh"]  # (width, height)
    except Exception as e:
        print(f"Error loading compressed file {compressed_npz_path}: {e}")
        return False

    reconstructed_net_output_array = decoder_model.predict(latent_representation)

    # Remove batch dimension if present and rescale to 0-255
    if len(reconstructed_net_output_array.shape) == 4:
        reconstructed_net_output_array = reconstructed_net_output_array[0]

    reconstructed_img_for_pil = (reconstructed_net_output_array * 255).astype(np.uint8)

    # Convert to PIL Image (this is currently at target_net_dims)
    current_channels = reconstructed_img_for_pil.shape[-1]
    if current_channels == 1:
        # Squeeze if it's (H, W, 1) to (H, W) for 'L' mode
        reconstructed_pil_net_dims = Image.fromarray(
            reconstructed_img_for_pil.squeeze(axis=-1), mode="L"
        )
    elif current_channels == 3:
        reconstructed_pil_net_dims = Image.fromarray(
            reconstructed_img_for_pil, mode="RGB"
        )
    else:
        print(
            f"Warning: Reconstructed image has {current_channels} channels. Attempting to save."
        )
        # Fallback for unexpected channel count, try to save first channel as grayscale
        # or first 3 as RGB if possible. This part might need refinement based on typical issues.
        if current_channels >= 3:
            reconstructed_pil_net_dims = Image.fromarray(
                reconstructed_img_for_pil[..., :3], mode="RGB"
            )
        elif current_channels >= 1:
            reconstructed_pil_net_dims = Image.fromarray(
                reconstructed_img_for_pil[..., 0], mode="L"
            )
        else:
            print("Error: Reconstructed image has 0 channels.")
            return False

    # Resize back to original dimensions using stored original_wh
    final_reconstructed_pil = reconstructed_pil_net_dims.resize(
        tuple(original_wh), Image.LANCZOS
    )

    try:
        final_reconstructed_pil.save(output_image_path)
        print(f"Decompressed image saved to {output_image_path}")
        return True
    except Exception as e:
        print(f"Error saving decompressed image: {e}")
        return False


# --- 5. Evaluation and Visualization ---
def display_comparison_images(
    originals_pil_or_path,
    reconstructed_pil_or_path,
    n=1,
    titles=("Original", "Reconstructed"),
):
    """Displays original and reconstructed images side-by-side from PIL objects or paths."""
    if not isinstance(originals_pil_or_path, list):
        originals_pil_or_path = [originals_pil_or_path]
    if not isinstance(reconstructed_pil_or_path, list):
        reconstructed_pil_or_path = [reconstructed_pil_or_path]

    actual_n = min(n, len(originals_pil_or_path), len(reconstructed_pil_or_path))
    if actual_n == 0:
        print("No images to display.")
        if (
            len(reconstructed_pil_or_path) == 1 and originals_pil_or_path[0] is None
        ):  # only reconstructed
            img_rec = reconstructed_pil_or_path[0]
            if isinstance(img_rec, str):
                img_rec = Image.open(img_rec)
            plt.figure(figsize=(5, 5))
            plt.imshow(img_rec)
            plt.title(titles[1])
            plt.axis("off")
            plt.show()
        return

    plt.figure(figsize=(10, 4 * actual_n // 2 if actual_n > 1 else 5))
    for i in range(actual_n):
        img_orig = originals_pil_or_path[i]
        img_rec = reconstructed_pil_or_path[i]

        if isinstance(img_orig, str):
            img_orig = Image.open(img_orig)
        if isinstance(img_rec, str):
            img_rec = Image.open(img_rec)

        # Display original
        ax = plt.subplot(2, actual_n, i + 1)
        if img_orig:
            plt.imshow(img_orig)
            plt.title(titles[0])
        else:
            plt.text(0.5, 0.5, "Original N/A", ha="center", va="center")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, actual_n, i + 1 + actual_n)
        plt.imshow(img_rec)
        plt.title(titles[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


def display_network_images(original_net_arrays, reconstructed_net_arrays, n=5):
    """Displays network-input-sized original and reconstructed images."""
    plt.figure(figsize=(10, 4))
    for i in range(min(n, len(original_net_arrays))):
        # Display original (network input version)
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_net_arrays[i].squeeze())  # .squeeze() for grayscale
        plt.title("Original (Net Input)")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction (network output version)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_net_arrays[i].squeeze())  # .squeeze() for grayscale
        plt.title("Reconstructed (Net Out)")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def evaluate_reconstruction_metrics(
    original_norm_array, reconstructed_norm_array, channels
):
    """Calculates MSE, PSNR, SSIM between two normalized numpy arrays of the same shape."""
    if original_norm_array.shape != reconstructed_norm_array.shape:
        print("Error: Arrays must have the same shape for metric evaluation.")
        print(
            f"Original shape: {original_norm_array.shape}, Reconstructed shape: {reconstructed_norm_array.shape}"
        )
        return -1, -1, -1

    mse_val = np.mean((original_norm_array - reconstructed_norm_array) ** 2)
    psnr_val = psnr(original_norm_array, reconstructed_norm_array, data_range=1.0)

    # SSIM needs careful handling of win_size, esp for small images
    min_dim = min(original_norm_array.shape[0], original_norm_array.shape[1])
    win_size = min(
        7, min_dim if min_dim % 2 == 1 else min_dim - 1
    )  # Ensure odd and <= min_dim
    if win_size < 3:
        win_size = 3  # Minimum sensible win_size, though SSIM might be less meaningful

    if channels > 1:
        ssim_val = ssim(
            original_norm_array,
            reconstructed_norm_array,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
            multichannel=True,
        )
    else:  # Grayscale, ssim expects 2D or 3D with last dim 1. Squeeze if needed.
        # Scikit-image ssim since 0.19 prefers multichannel=True and channel_axis even for grayscale if it has a channel dim
        # If it's purely 2D, then multichannel=False (default)
        if original_norm_array.ndim == 3 and original_norm_array.shape[-1] == 1:
            ssim_val = ssim(
                original_norm_array,
                reconstructed_norm_array,
                data_range=1.0,
                channel_axis=-1,
                win_size=win_size,
                multichannel=True,
            )
        else:  # Assuming 2D grayscale
            ssim_val = ssim(
                original_norm_array.squeeze(),
                reconstructed_norm_array.squeeze(),
                data_range=1.0,
                win_size=win_size,
            )

    print(f"Mean Squared Error (MSE): {mse_val:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_val:.2f} dB")
    print(f"Structural Similarity Index (SSIM): {ssim_val:.4f}")
    return mse_val, psnr_val, ssim_val


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Convolutional Autoencoder for Image Compression ---")

    while True:
        try:
            TARGET_NET_DIM_input = int(
                input(
                    "Enter target square dimension for network processing (e.g., 128, 256, must be divisible by 4): "
                ).strip()
            )
            if TARGET_NET_DIM_input % 4 != 0 or TARGET_NET_DIM_input <= 0:
                print("Dimension must be a positive number divisible by 4.")
                continue
            TARGET_NET_DIM = TARGET_NET_DIM_input
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            CHANNELS_input = int(
                input("Enter number of channels (1 for grayscale, 3 for RGB): ").strip()
            )
            if CHANNELS_input not in [1, 3]:
                print("Channels must be 1 or 3.")
                continue
            CHANNELS = CHANNELS_input
            break
        except ValueError:
            print("Invalid input. Please enter 1 or 3.")

    while True:
        try:
            # This factor multiplies the number of input channels to get bottleneck channels
            # Overall tensor element compression factor will be 16 / LATENT_DIM_FACTOR
            LATENT_DIM_FACTOR_input = float(
                input(
                    f"Enter bottleneck channel multiplier (e.g., 1.0 means bottleneck channels = {CHANNELS} channels). \n"
                    f"A value of 4.0 means bottleneck has {CHANNELS*4} channels.\n"
                    f"Overall tensor element compression is roughly 16 / this_factor.\n"
                    f"  (e.g., factor 1.0 -> 16x compr.; factor 4.0 -> 4x compr.; factor 0.25 -> 64x compr.): "
                ).strip()
            )
            if LATENT_DIM_FACTOR_input <= 0:
                print("Factor must be positive.")
                continue
            LATENT_DIM_FACTOR = LATENT_DIM_FACTOR_input
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    input_shape = (TARGET_NET_DIM, TARGET_NET_DIM, CHANNELS)
    # The number of channels in the bottleneck.
    # Overall data compression of elements = 16 / LATENT_DIM_FACTOR
    latent_dim_channels = int(CHANNELS * LATENT_DIM_FACTOR)
    if latent_dim_channels < 1:
        latent_dim_channels = 1  # Ensure at least one channel
    print(f"Bottleneck will have {latent_dim_channels} channels.")
    print(
        f"This configuration results in an overall tensor element compression of roughly {16.0/LATENT_DIM_FACTOR:.2f}x."
    )

    autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_dim_channels)
    # autoencoder.summary() # Can be verbose

    MODEL_SAVE_PATH = MODEL_SAVE_PATH_TEMPLATE.format(
        TARGET_NET_DIM, TARGET_NET_DIM, str(LATENT_DIM_FACTOR).replace(".", "_")
    )
    print(f"Model weights will be saved/loaded from: {MODEL_SAVE_PATH}")

    while True:
        print("\n--- Choose Mode ---")
        print("1: Train model")
        print("2: Compress an image")
        print("3: Decompress an image")
        print("4: Train model AND evaluate on a test/validation set")
        print("5: Exit")
        mode = input("Enter selection (1-5): ").strip()

        if mode == "1" or mode == "4":
            x_train_data, x_val_data = None, None
            use_cifar = (
                input("Use CIFAR-10 demo data? (y/n, default n): ").strip().lower()
            )
            if use_cifar == "y":
                x_train_data, x_val_data = load_cifar10_data(
                    (TARGET_NET_DIM, TARGET_NET_DIM), CHANNELS
                )
            else:
                train_dir = input("Enter path to TRAINING image directory: ").strip()
                if not os.path.isdir(train_dir):
                    print(f"Error: Training directory '{train_dir}' not found.")
                    continue

                val_dir_choice = (
                    input(
                        "Use separate VALIDATION image directory? (y/n, default n for auto-split): "
                    )
                    .strip()
                    .lower()
                )
                if val_dir_choice == "y":
                    val_dir = input(
                        "Enter path to VALIDATION image directory: "
                    ).strip()
                    if not os.path.isdir(val_dir):
                        print(f"Error: Validation directory '{val_dir}' not found.")
                        continue
                    x_train_data, _, _ = load_dataset_from_directory(
                        train_dir,
                        (TARGET_NET_DIM, TARGET_NET_DIM),
                        CHANNELS,
                        test_split_ratio=0,
                    )
                    x_val_data, _, _ = load_dataset_from_directory(
                        val_dir,
                        (TARGET_NET_DIM, TARGET_NET_DIM),
                        CHANNELS,
                        test_split_ratio=0,
                    )
                else:
                    val_split = 0.2
                    try:
                        val_split_input = input(
                            "Enter validation split ratio from training data (e.g., 0.2 for 20%, default 0.2): "
                        ).strip()
                        if val_split_input:  # if not empty
                            val_split = float(val_split_input)
                        if not (0 <= val_split < 1):
                            print(
                                "Validation split must be between 0.0 and < 1.0. Using 0.2."
                            )
                            val_split = 0.2
                    except ValueError:
                        print("Invalid split ratio. Using 0.2.")
                        val_split = 0.2
                    x_train_data, x_val_data, _ = load_dataset_from_directory(
                        train_dir,
                        (TARGET_NET_DIM, TARGET_NET_DIM),
                        CHANNELS,
                        test_split_ratio=val_split,
                    )

            if x_train_data is None or len(x_train_data) == 0:
                print("No training data loaded. Cannot proceed with training.")
                continue

            if os.path.exists(MODEL_SAVE_PATH):
                if (
                    input(
                        f"Found existing model {MODEL_SAVE_PATH}. Load weights and continue training or retrain from scratch? (load/scratch, default: load): "
                    ).lower()
                    == "scratch"
                ):
                    print("Training new model from scratch...")
                    # Rebuild to ensure fresh weights if user chooses scratch on existing compiled model
                    autoencoder, encoder, decoder = build_autoencoder(
                        input_shape, latent_dim_channels
                    )
                else:
                    print(
                        f"Loading existing model weights from {MODEL_SAVE_PATH} to continue training..."
                    )
                    try:
                        autoencoder.load_weights(MODEL_SAVE_PATH)
                    except Exception as e:
                        print(
                            f"Error loading weights: {e}. Training from scratch instead."
                        )
                        autoencoder, encoder, decoder = build_autoencoder(
                            input_shape, latent_dim_channels
                        )  # Rebuild

            train_model(
                autoencoder,
                x_train_data,
                x_val_data,
                EPOCHS,
                BATCH_SIZE,
                MODEL_SAVE_PATH,
            )

            if mode == "4":
                print("\n--- Evaluating on Test/Validation Set ---")
                eval_data = x_val_data
                if (
                    use_cifar == "y"
                ):  # CIFAR-10 test set was loaded into x_val_data by load_cifar10_data
                    print("Using CIFAR-10 test set for evaluation.")
                elif x_val_data is None or len(x_val_data) == 0:
                    print("No validation data available for evaluation. Skipping.")
                    continue
                else:
                    print("Using custom validation set for evaluation.")

                if eval_data is not None and len(eval_data) > 0:
                    decoded_eval_imgs = autoencoder.predict(eval_data)
                    display_network_images(
                        eval_data[:10],
                        decoded_eval_imgs[:10],
                        n=min(10, len(eval_data)),
                    )
                    evaluate_reconstruction_metrics(
                        eval_data, decoded_eval_imgs, CHANNELS
                    )
                else:
                    print("No data to evaluate.")

        elif mode == "2":  # Compress
            if not os.path.exists(MODEL_SAVE_PATH):
                print(
                    f"Error: Model file '{MODEL_SAVE_PATH}' not found. Please train the model first."
                )
            else:
                autoencoder.load_weights(MODEL_SAVE_PATH)
                image_to_compress = input(
                    "Enter path to image for compression: "
                ).strip()
                if not os.path.exists(image_to_compress):
                    print(f"Error: Image file '{image_to_compress}' not found.")
                else:
                    base, ext = os.path.splitext(image_to_compress)
                    compressed_output_path = (
                        base + "_compressed" + COMPRESSED_FILE_EXTENSION
                    )
                    compress_image(
                        encoder,
                        image_to_compress,
                        compressed_output_path,
                        (TARGET_NET_DIM, TARGET_NET_DIM),
                        CHANNELS,
                    )

        elif mode == "3":  # Decompress
            if not os.path.exists(MODEL_SAVE_PATH):
                print(
                    f"Error: Model file '{MODEL_SAVE_PATH}' not found. Please train the model first."
                )
            else:
                autoencoder.load_weights(MODEL_SAVE_PATH)
                compressed_file = input(
                    f"Enter path to compressed file (e.g., image_compressed{COMPRESSED_FILE_EXTENSION}): "
                ).strip()
                if not os.path.exists(compressed_file):
                    print(f"Error: Compressed file '{compressed_file}' not found.")
                else:
                    base, _ = os.path.splitext(compressed_file)
                    if base.endswith("_compressed"):
                        base = base[: -len("_compressed")]

                    # Suggest a name for the reconstructed image based on the compressed file name
                    suggested_recon_name = base + "_reconstructed.png"
                    decompressed_output_path = input(
                        f"Enter path to save decompressed image (default: {suggested_recon_name}): "
                    ).strip()
                    if not decompressed_output_path:
                        decompressed_output_path = suggested_recon_name

                    if decompress_image(
                        decoder, compressed_file, decompressed_output_path
                    ):
                        # Try to find original for comparison
                        original_image_path_guess = None
                        common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
                        for ext_g in common_exts:
                            potential_orig_path = base + ext_g
                            if os.path.exists(potential_orig_path):
                                original_image_path_guess = potential_orig_path
                                break

                        original_pil_for_display = None
                        if original_image_path_guess:
                            print(
                                f"Found potential original image: {original_image_path_guess}"
                            )
                            try:
                                original_pil_for_display = Image.open(
                                    original_image_path_guess
                                )
                            except Exception as e:
                                print(f"Could not load original image for display: {e}")
                        else:
                            print(
                                f"Could not automatically find an original image matching base name '{base}'."
                            )

                        try:
                            reconstructed_pil_for_display = Image.open(
                                decompressed_output_path
                            )
                        except Exception as e:
                            print(
                                f"Could not load reconstructed image for display: {e}"
                            )
                            continue

                        print(
                            "\n--- Comparing Original (if found) and Reconstructed ---"
                        )
                        display_comparison_images(
                            (
                                original_pil_for_display
                                if original_pil_for_display
                                else "Original N/A"
                            ),  # Path or PIL
                            reconstructed_pil_for_display,  # Path or PIL
                        )

                        if original_pil_for_display:
                            print(
                                "\n--- Calculating Metrics (comparing versions processed for network) ---"
                            )
                            original_net_array, _ = get_network_ready_image(
                                original_pil_for_display,
                                (TARGET_NET_DIM, TARGET_NET_DIM),
                                CHANNELS,
                            )
                            # For metrics, re-process the saved reconstructed image to network dimensions
                            reconstructed_net_array, _ = get_network_ready_image(
                                reconstructed_pil_for_display,
                                (TARGET_NET_DIM, TARGET_NET_DIM),
                                CHANNELS,
                            )

                            if (
                                original_net_array is not None
                                and reconstructed_net_array is not None
                            ):
                                evaluate_reconstruction_metrics(
                                    original_net_array,
                                    reconstructed_net_array,
                                    CHANNELS,
                                )
                            else:
                                print(
                                    "Could not prepare images for metric calculation."
                                )
                        else:
                            print(
                                "Skipping metrics as original image was not available for comparison."
                            )

        elif mode == "5":
            print("Exiting program.")
            break
        else:
            print("Invalid mode selected. Please choose a number from 1 to 5.")

    print("Program finished.")
