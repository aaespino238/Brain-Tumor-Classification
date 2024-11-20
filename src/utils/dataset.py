import pandas as pd
import numpy as np
import os

dataset_path = '../dataset/'

def get_class_paths():
    classes = []
    class_paths = []

    path = os.path.join(dataset_path, "allData")
    
    for label in os.listdir(path):
        label_path = os.path.join(path, label)

        if os.path.isdir(label_path):
            for image in os.listdir(label_path):
                image_path = os.path.join(label_path, image)

                classes.append(label)
                class_paths.append(image_path)
    
    df = pd.DataFrame({
        'Class Path': class_paths,
        'Class': classes
    })

    return df

