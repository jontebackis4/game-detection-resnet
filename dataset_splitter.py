import os
import csv
import random
from collections import defaultdict

def split_dataset_by_labels(folder_path, train_ratio=0.7, val_ratio=0.15):
    images_by_label = defaultdict(list)

    for file_name in os.listdir(folder_path):
        label = file_name.split('_')[0]
        images_by_label[label].append(file_name)

    train_set, val_set, test_set = [], [], []

    # Split the data into training, validation, and test sets
    for label, images in images_by_label.items():
        random.shuffle(images)
        num_images = len(images)
        print(label, 'has', num_images, 'images')
        
        train_end = int(train_ratio * num_images)
        val_end = train_end + int(val_ratio * num_images)

        train_set.extend([(img, label) for img in images[:train_end]])
        val_set.extend([(img, label) for img in images[train_end:val_end]])
        test_set.extend([(img, label) for img in images[val_end:]])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set

# All images in your dataset should be stored in one folder
# The label has to be the first part of the file name with '_' as separator
# Three CSV files with image file names will be created for the training, validation, and test sets
def save_sets_to_csv_by_label(dataset_path, train_csv='dataset/train_set.csv', val_csv='dataset/val_set.csv', test_csv='dataset/test_set.csv'):
    train_set, val_set, test_set = split_dataset_by_labels(dataset_path)

    with open(train_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(train_set)

    with open(val_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(val_set)

    with open(test_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(test_set)

    print(f'Saved train set to {train_csv}, validation set to {val_csv}, and test set to {test_csv}')

save_sets_to_csv_by_label('dataset/frames')