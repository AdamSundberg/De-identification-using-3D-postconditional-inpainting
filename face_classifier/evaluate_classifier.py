import os

import hydra
import torch
from classifier import Classifier
from dataset import VolumeDataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_score, recall_score
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


@hydra.main(version_base=None, config_path="../config", config_name="classifier_eval_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    full_dataset = VolumeDataset(
        cfg.dataset.faces_root_dir
    )

    batch_size = cfg.batch_size

    eval_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    model = Classifier()
    model.to(device)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(eval_dataloader, desc="Evaluation", total=len(eval_dataloader)):
            inputs = inputs.to(device)
            labels = labels.long()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    test_loss /= len(eval_dataloader)
    test_accuracy = 100 * correct / total
    test_precision = precision_score(true_labels, predicted_labels, average="macro")
    test_recall = recall_score(true_labels, predicted_labels, average="macro")

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}"
    )


if __name__ == "__main__":
    run()
