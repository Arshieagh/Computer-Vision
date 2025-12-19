import torch
from tqdm import tqdm
from data.cifar100 import load_data
from models.baseline import BaseLineModel

lambda_p = 0.5


def train(model, train_dataloader, lr=0.001, num_epochs=20, device=None):
    print(f"Using device: {device}")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")

        running_loss = 0.0
        for _, (data, targets) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            fine_targets, coarse_targets = targets
            fine_targets = fine_targets.to(device)
            coarse_targets = coarse_targets.to(device)

            # Forward pass
            logits_fine, logits_coarse = model(data)
            loss_fine = criterion(logits_fine, fine_targets)
            loss_coarse = criterion(logits_coarse, coarse_targets)
            loss = loss_fine + lambda_p * loss_coarse

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")


def evaluate(model, test_dataloader, device=None):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.to(device)
            fine_targets, coarse_targets = targets
            fine_targets = fine_targets.to(device)

            logits_fine, _ = model(data)
            _, predicted = torch.max(logits_fine.data, 1)
            total += fine_targets.size(0)
            correct += (predicted == fine_targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")


def main():
    batch_size = 128
    num_epochs = 20  # 50
    learning_rate = 0.001

    train_dataloader, test_dataloader = load_data(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaseLineModel()
    model = model.to(device)

    train(
        model,
        train_dataloader,
        lr=learning_rate,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate(model, test_dataloader, device=device)


if __name__ == "__main__":
    main()
