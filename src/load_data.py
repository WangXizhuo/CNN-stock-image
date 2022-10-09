from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
from torchvision.transforms import transforms

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 60
FORWARD_DAY = 5
BATCH_SIZE = 128
YEAR_START = 1993
YEAR_END = 1993
IMAGE_DIR = '../monthly_20d'
LOG_DIR = 'logs'

class StockImage(Dataset):
    def __init__(self, root_dir, start_year, end_year, transform=None):
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year
        self.transform = transform
        img_array = []

        label_array = np.array([])


        for year in range(start_year, end_year+1):
            img_path = os.path.join(root_dir,
                                    f"20d_month_has_vb_[20]_ma_{year}_images.dat")
            label_path= os.path.join(root_dir,
                                     f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")
            images = np.memmap(img_path,
                               dtype=np.uint8,
                               mode='r').reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH))
            for i in range(len(images)):
                img_array.append(np.array(images[i]))

            label_df = pd.read_feather(label_path)
            label_array = np.concatenate([label_array,
                                         np.where(label_df[f"Ret_{FORWARD_DAY}d"] > 0, 1, 0)])
        self.label = label_array
        self.image = np.array(img_array)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = np.eye(2)[int(self.label.item(idx))]
        # label = [self.label[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image)

# check sample dataset
if __name__ == '__main__':
    # To pass to dataloader
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    sample_data = StockImage(IMAGE_DIR,
                             YEAR_START,
                             YEAR_END,
                             transforms)

    sample_data = DataLoader(sample_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             drop_last=True)

    writer = SummaryWriter(LOG_DIR)
    batch_num = 1

    for data in sample_data:
        for i in range(128):
            images, targets = data
            print(images[0], targets[0])
            writer.add_image(f"sample data in batch{batch_num}",
                             images[i],
                             i,
                             dataformats='CHW')
        batch_num += 1
        print(f"Batch {batch_num} finished loading")

# load image: type "tensorboard --logdir=LOG_DIR --port=6007"  in terminal



