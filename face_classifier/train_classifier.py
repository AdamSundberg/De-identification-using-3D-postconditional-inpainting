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


# Training loop
def train(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, writer, device
):
    best_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs), desc="Epoch", total=num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []

        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
            inputs = inputs.to(device)
            labels = labels.long()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_precision = precision_score(
            true_labels, predicted_labels, average="macro"
        )
        train_recall = recall_score(true_labels, predicted_labels, average="macro")

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)
        writer.add_scalar("train_precision", train_precision, epoch)
        writer.add_scalar("train_recall", train_recall, epoch)

        print(
            f"Train Epoch: [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        )

        val_loss, val_accuracy, val_precision, val_recall = validate(
            model, val_loader, criterion, writer, epoch, device
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved with best validation loss.")


# Validation loop
def validate(model, val_loader, criterion, writer, epoch, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", total=len(val_loader)):
            inputs = inputs.to(device)
            labels = labels.long()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_precision = precision_score(true_labels, predicted_labels, average="macro")
    val_recall = recall_score(true_labels, predicted_labels, average="macro")

    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_accuracy", val_accuracy, epoch)
    writer.add_scalar("val_precision", val_precision, epoch)
    writer.add_scalar("val_recall", val_recall, epoch)

    print(
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}"
    )
    return val_loss, val_accuracy, val_precision, val_recall


def test(model, test_loader, criterion, writer, device):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test", total=len(test_loader)):
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

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    test_precision = precision_score(true_labels, predicted_labels, average="macro")
    test_recall = recall_score(true_labels, predicted_labels, average="macro")

    writer.add_scalar("test_loss", test_loss)
    writer.add_scalar("test_accuracy", test_accuracy)
    writer.add_scalar("test_precision", test_precision)
    writer.add_scalar("test_recall", test_recall)

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}"
    )


@hydra.main(version_base=None, config_path="../config", config_name="classifier_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    full_dataset = VolumeDataset(
        cfg.dataset.faces_root_dir, cfg.dataset.no_faces_root_dir
    )
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset, [0.8, 0.1, 0.1]
    )

    batch_size = cfg.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Classifier()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = cfg.epochs
    train(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs,
        writer,
        device,
    )

    test(model, test_dataloader, criterion, writer, device)

    writer.close()


if __name__ == "__main__":
    run()
