import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf # type: ignore
from tqdm import tqdm
from PIL import Image
from evaluations.evaluator import Evaluator  
import argparse

# Disable TensorFlow 2.x behavior
tf.disable_v2_behavior()

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID statistics for a given dataset path.")
    parser.add_argument("--base_path", type=str, default="CelebA",
                        help="Base path to the dataset (e.g., CelebA)")
    # parser.add_argument("--dataset_type", type=str, choices=["train", "val", "test"], default="train",
    #                     help="Dataset type to use (e.g., train, val, test)")
    parser.add_argument("--dataset_name", type=str, default="church",
                        help="Name of the dataset (e.g., celeba)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images.")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to (e.g., 64 for 64x64).")
    return parser.parse_args()

def preprocess_image(img_path, image_size):
    with Image.open(img_path).convert("RGB") as img:
        img = img.resize((image_size, image_size), Image.BILINEAR)  
        img = np.array(img, dtype=np.uint8)  
        img = np.clip(img, 0, 255)  
    return img

def calculate_fid_statistics(image_paths, evaluator, batch_size, image_size):
    batches = [np.array([preprocess_image(p, image_size) for p in image_paths[i:i + batch_size]]).astype(np.uint8)
               for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches")]

    ref_acts = evaluator.compute_activations(batches)
    ref_stats, ref_stats_spatial = tuple(evaluator.compute_statistics(x) for x in ref_acts)
        
    ref_images = np.concatenate(batches, axis=0)
    if ref_images.shape[0] > 10000:
        indices = np.random.choice(ref_images.shape[0], 10000, replace=False)
        arr_0 = ref_images[indices]
    else:
        arr_0 = ref_images
        
    return ref_stats, ref_stats_spatial, arr_0

if __name__ == "__main__":
    args = parse_args()

    base_path = args.base_path
    # dataset_type = args.dataset_type
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    image_size = args.image_size

    # Validate the provided arguments
    if not base_path:
        raise ValueError("Base path is not specified. Please provide a valid base path to the dataset.")
    # if not dataset_type:
    #     raise ValueError("Dataset type is not specified. Please choose from 'train', 'val', or 'test'.")
    if not dataset_name:
        raise ValueError("Dataset name is not specified. Please provide a valid dataset name.")

    # Construct the full data path
    data_path = os.path.join(base_path)#, dataset_type)

    # Recursively find all images in the given dataset path, including subdirectories
    image_paths = glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True)
    print(f"Total images found: {len(image_paths)}")

    # Determine the output file name based on the selected dataset type and dataset name
    output_path = f"./VIRTUAL_{dataset_name}{image_size}.npz"

    with tf.Session() as sess:
        evaluator = Evaluator(sess)

        print("Calculating FID statistics using Evaluator...")
        ref_stats, ref_stats_spatial, arr_0 = calculate_fid_statistics(image_paths, evaluator, batch_size, image_size)

    mu = ref_stats.mu
    sigma = ref_stats.sigma
    mu_s = ref_stats_spatial.mu
    sigma_s = ref_stats_spatial.sigma
    arr_0 = arr_0.astype(np.uint8)
    
    print("Saving FID statistics...")
    np.savez_compressed(output_path, mu=mu, sigma=sigma, mu_s=mu_s, sigma_s=sigma_s, arr_0=arr_0)
    print(f"FID statistics saved to {output_path}")
