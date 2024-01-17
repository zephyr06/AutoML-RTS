import time
import torch
import torch.nn as nn

from .Cifar10DataIO import data_loader
from .ResNet import ResNet, ResidualBlock, device, get_resnet_blocks
from .Hyperparameters import Hyperparameters


def evaluate_resnet(model, hyperparameters):
    data_size_test = hyperparameters.data_size_test
    batch_size = hyperparameters.batch_size

    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              test_only=True, test_data_size=data_size_test)

    with torch.no_grad():
        start_inference_test_time = time.time()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        end_inference_test_time = time.time()
        average_inference_time = (
            end_inference_test_time - start_inference_test_time) / total
        final_accuracy = 100 * correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(
            total, final_accuracy))
        print(f"Average running time per image during inference: ",
              average_inference_time, "seconds")
    return final_accuracy, average_inference_time


def fine_tune_resnet(model, hp):
    model.to(device)
    data_size_train = hp.data_size_train
    num_epochs = hp.num_epochs
    batch_size = hp.batch_size
    learning_rate = hp.learning_rate

    # CIFAR10 dataset
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             training_data_size=hp.data_size_train,
                                             training_noise_chance=hp.training_noise)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

    # Train the model
    import gc
    final_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the {} validation images: {} %'.format(data_size_train,
                                                                                     100 * correct / total))

    return model


def train_and_evaluate_resnet(model, hp):
    start_all_time = time.time()

    model = fine_tune_resnet(model=model, hp=hp)

    final_accuracy, average_inference_time = evaluate_resnet(
        model, hp)

    end_all_time = time.time()
    total_run_time = end_all_time - start_all_time
    print(f"Total running time: ", total_run_time, "seconds")
    return final_accuracy, round(average_inference_time * 1000, 3)


def load_train_evaluate_resnet(hyperparameters):
    layer_num = 18
    num_classes = 10
    layer_dist = get_resnet_blocks(num_layers=layer_num)
    model = ResNet(ResidualBlock, layer_dist, prune_ratio=0, quant_type="none",
                   num_classes=num_classes).to(device)
    train_and_evaluate_resnet(hyperparameters, model)


def print_layer_channels(layer):
    print(layer[0].conv1.out_channels, layer[1].conv1.out_channels)


def print_resnet18_structure(model):
    # print(model.conv1.in_channels)
    print("ResNet structure (in case of resnet18):")
    print_layer_channels(model.layer1)
    print_layer_channels(model.layer2)
    print_layer_channels(model.layer3)
    print_layer_channels(model.layer4)


if __name__ == "__main__":
    # training max: 5e4
    # testing max: 1e4
    hyperparameters = {"data_size_train": 128, "data_size_test": 1024, "prune_ratio": 0.5, "quant_type": "none",
                       "layer_num": 18, "num_epochs": 2}
    # hyperparameters = {"data_size_train": 128, "data_size_test": 10000, "prune_ratio": 0.8, "quant_type": "qint8",
    #                    "layer_num": 34, "num_epochs": 2}
    # hyperparameters = {"data_size_train": 128, "data_size_test": 10000, "prune_ratio": 0.8, "quant_type": "none",
    #                    "layer_num": 34, "num_epochs": 2}
    load_train_evaluate_resnet(hyperparameters)
