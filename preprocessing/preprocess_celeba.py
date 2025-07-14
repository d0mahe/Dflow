import os
import torch
import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}
ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 
                    'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 
                   'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 
                   'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17,
                   'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 
                   'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 
                   'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 
                   'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 
                   'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 
                   'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}

IMG_SIZE = 64
IX_TO_ATTR_DICT = {v: k for k, v in ATTR_TO_IX_DICT.items()}

def preprocess_images(args):
    # Ensure output directory for the partition exists
    partition_dir = os.path.join(args.out_dir, args.partition, 'img')  # Add 'img' subfolder to prevent ImageFolder errors
    os.makedirs(partition_dir, exist_ok=True)

    # Load data and attributes
    print('Preprocessing partition {}'.format(args.partition))
    transform = transforms.Compose([transforms.CenterCrop(140), transforms.Resize(IMG_SIZE)])
    eval_data = load_eval_partition(args.partition, args.data_dir)
    attr_data = load_attributes(eval_data, args.partition, args.data_dir)

    # Save each image individually
    print('Starting conversion and saving...')
    labels = []  # to save labels for each image

    # Adding progress bar using tqdm
    for i in tqdm.tqdm(range(len(eval_data)), desc=f"Processing {args.partition} images"):
        img_path = os.path.join(args.data_dir, 'img_align_celeba', eval_data[i])
        
        with Image.open(img_path) as img:
            img = transform(img)
            img = img.convert("RGB")  # Ensure the image is in RGB format

            # Save image as .jpg or .png in the partition folder
            output_img_path = os.path.join(partition_dir, f'{i:06d}.png')  # Save as 000000.png, 000001.png, etc.
            img.save(output_img_path)
            
            # Collect the label information
            labels.append(attr_data[i].tolist())  # Convert tensor to list for saving

    # Save labels as a CSV file
    label_file_path = os.path.join(args.out_dir, f'{args.partition}_labels.csv')
    with open(label_file_path, 'w') as f:
        header = ','.join([IX_TO_ATTR_DICT[i] for i in range(len(ATTR_TO_IX_DICT))]) + '\n'
        f.write(header)
        for label in labels:
            f.write(','.join(map(str, label)) + '\n')

    print(f"Preprocessing for {args.partition} completed. Images and labels saved.")

def load_eval_partition(partition, data_dir):
    eval_data = []
    with open(os.path.join(data_dir, 'list_eval_partition.txt')) as fp:
        rows = fp.readlines()
        for row in rows:
            path, label = row.strip().split(' ')
            label = int(label)
            if label == VALID_PARTITIONS[partition]:
                eval_data.append(path)
    return eval_data

def load_attributes(paths, partition, data_dir):
    attr_data = []
    with open(os.path.join(data_dir, 'list_attr_celeba.txt')) as fp:
        rows = fp.readlines()
        for row in rows[2:]:  # Skip the first two lines
            row = row.strip().split()
            path, attrs = row[0], row[1:]
            if path in paths:
                attrs = np.array(attrs).astype(int)
                attrs[attrs < 0] = 0
                attr_data.append(attrs)
    attr_data = np.vstack(attr_data).astype(np.int64)
    attr_data = torch.from_numpy(attr_data).float()
    return attr_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/', type=str, help='Path to downloaded CelebA dataset (e.g. ./data)')
    parser.add_argument('--out_dir', default='./CelebA/', type=str, help='Destination of output images and labels')
    parser.add_argument('--partition', default='train', type=str, help='Partition to process: train, val, test')
    args = parser.parse_args()
    preprocess_images(args)
