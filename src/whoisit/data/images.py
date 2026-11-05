"""Load the photos as tensors for training."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_DIR = Path("data/aligned/celeba/images")  # made by preprocess.py
SIZE = 112
MEAN = (0.485, 0.456, 0.406)  # the values the ImageNet network expects
STD = (0.229, 0.224, 0.225)


def make_transform(train):
    steps = [transforms.Resize((SIZE, SIZE))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class FaceDataset(Dataset):
    def __init__(self, photos, image_dir=IMAGE_DIR, train=True):
        # photos: table with the columns image and person
        # preprocess.py saves the photo as images/<person>/<name>.jpg
        self.paths = [Path(image_dir) / str(person) / (Path(name).stem + ".jpg")
                      for name, person in zip(photos["image"], photos["person"])]
        people = sorted(photos["person"].unique())
        label_of = {person: i for i, person in enumerate(people)}
        self.labels = [label_of[person] for person in photos["person"]]
        self.num_people = len(people)
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]
