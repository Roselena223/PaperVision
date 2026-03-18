import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from args import get_args
from utils import resize_box_xyxy


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        args = get_args()
        # TODO 1: Get the row number idx from dataframe
        # your code here
        row = self.df.iloc[idx]

        img = Image.open(row["images"]).convert("RGB")
        w, h = img.size
        
        image = img.resize((args.image_size, args.image_size))
        
        image = to_tensor(img)

        boxes, labels = [], []
        with open(row["labels"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                
                x1, y1, x2, y2 = resize_box_xyxy((x1, y1, x2, y2), w, h, args.image_size, args.image_size)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        # TODO 2: Return what you need from this class
        # your code here
        return image, target
