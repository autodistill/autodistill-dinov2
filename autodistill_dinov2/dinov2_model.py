import json
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
from autodistill.detection import DetectionTargetModel
from PIL import Image
from sklearn import svm
from tqdm import tqdm

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_image = T.Compose(
    [T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def compute_embeddings(files: list, dinov2_vits14) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14(load_image(file).to(DEVICE))

            all_embeddings[file] = (
                np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
            )

    return all_embeddings


@dataclass
class DINOv2(DetectionTargetModel):
    def __init__(self):
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinov2_vits14.to(device)

        self.dinov2_model = dinov2_vits14

    def predict(self, input: str) -> sv.Detections:
        embedding = compute_embeddings([input], self.dinov2_model)

        return self.model.predict(np.array(embedding[input]).reshape(-1, 384))

    def train(self, dataset_location: str):
        dataset = sv.ClassificationDataset.from_multiclass_folder_structure(
            dataset_location
        )

        clf = svm.SVC(gamma="scale")

        classes = dataset.classes
        images = list(dataset.images.keys())[:500]
        annotations = dataset.annotations

        images = [file for file in images if file.endswith(".jpg")]

        embeddings = compute_embeddings(images, self.dinov2_model)

        with open("embeddings.json", "w") as f:
            json.dump(embeddings, f)

        y = [classes[annotations[file].class_id[0]] for file in images]

        embedding_list = [embeddings[file] for file in images]

        # svm needs at least 2 classes
        unqiue_classes = list(set(y))

        if len(unqiue_classes) == 1:
            raise ValueError("Only one class in dataset")

        # DINOv2 has 384 dimensions
        clf.fit(np.array(embedding_list).reshape(-1, 384), y)

        self.model = clf
