import json
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
from autodistill.detection import CaptionOntology
from autodistill.classification import ClassificationBaseModel
from PIL import Image
from sklearn import svm
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
class DINOv2(ClassificationBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinov2_vits14.to(device)

        self.dinov2_model = dinov2_vits14
        self.ontology = ontology

    def predict(self, input: str) -> sv.Classifications:
        embedding = compute_embeddings([input], self.dinov2_model)

        class_id = self.model.predict(np.array(embedding[input]).reshape(-1, 384))

        return sv.Classifications(
            class_id=np.array([self.ontology.classes().index(class_id)]),
            confidence=np.array([1]),
        )

    def train(self, dataset_location: str):
        dataset = sv.ClassificationDataset.from_folder_structure(dataset_location)

        clf = svm.SVC(gamma="scale")

        classes = dataset.classes
        images = list(dataset.images.keys())
        annotations = dataset.annotations

        all_images = []

        for image in images:
            class_label = classes[annotations[image].class_id[0]]

            all_images.append(os.path.join(dataset_location, class_label, image))

        embeddings = compute_embeddings(all_images, self.dinov2_model)

        with open("embeddings.json", "w") as f:
            json.dump(embeddings, f)

        y = [
            classes[annotations[os.path.basename(file)].class_id[0]]
            for file in all_images
        ]

        embedding_list = [embeddings[file] for file in all_images]

        # svm needs at least 2 classes
        unqiue_classes = list(set(y))

        if len(unqiue_classes) == 1:
            raise ValueError("Only one class in dataset")

        # DINOv2 has 384 dimensions
        clf.fit(np.array(embedding_list).reshape(-1, 384), y)

        self.model = clf
