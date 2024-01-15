import time
import torch
import torch.nn as nn

from .Cifar10DataIO import data_loader
from .ResNet import ResNet, ResidualBlock, device


def get_resnet_blocks(num_layers):
    # Dictionary mapping the total number of layers to the number of layer blocks in each stage
    resnet_configs = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        # Add more configurations as needed
    }

    # Check if the provided number of layers is in the dictionary
    if num_layers in resnet_configs:
        return resnet_configs[num_layers]
    else:
        # Redistribute layers for other cases
        base_blocks = [3, 4, 6, 3]  # Base configuration for redistribution
        total_blocks = sum(base_blocks)
        redistributed_blocks = [
            int(round(b * (num_layers - 2) / total_blocks)) for b in base_blocks]

        # Adjust to ensure the total number of layers is exactly num_layers
        diff = num_layers - 2 - sum(redistributed_blocks)
        # Add the difference to the first stage
        redistributed_blocks[0] += diff

        return redistributed_blocks


def evaluate_resnet(model, hyperparameters):
    data_size_test = hyperparameters.get("data_size_test", 1000)
    num_classes = hyperparameters.get("num_classes", 10)
    batch_size = hyperparameters.get("batch_size", 128)

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
            # del images, labels, outputs

        end_inference_test_time = time.time()
        average_inference_time = (
            end_inference_test_time - start_inference_test_time) / total
        final_accuracy = 100 * correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(
            total, final_accuracy))
        print(f"Average running time per image during inference: ",
              average_inference_time, "seconds")
    return average_inference_time, final_accuracy


def fine_tune_resnet(model, hyperparameters):
    model.to(device)
    data_size_train = hyperparameters.get("data_size_train", 1000)
    num_epochs = hyperparameters.get("num_epochs", 15)
    batch_size = hyperparameters.get("batch_size", 128)
    learning_rate = hyperparameters.get("learning_rate", 0.01)

    # CIFAR10 dataset
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size, training_data_size=data_size_train)

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


def train_and_evaluate_resnet(model, hyperparameters):
    start_all_time = time.time()

    # extract hyper-parameters
    data_size_train = hyperparameters.get("data_size_train", 1000)
    data_size_test = hyperparameters.get("data_size_test", 1000)
    num_epochs = hyperparameters.get("num_epochs", 15)
    batch_size = hyperparameters.get("batch_size", 128)
    learning_rate = hyperparameters.get("learning_rate", 0.01)

    # ************************************* Optimization variables *************************************
    prune_ratio = hyperparameters.get("prune_ratio", 0)
    quant_type = hyperparameters.get("quant_type", "qint8")
    # layer_num = hyperparameters.get("layer_num", 34)

    model = fine_tune_resnet(model=model, hyperparameters=hyperparameters)

    average_inference_time, final_accuracy = evaluate_resnet(
        model, hyperparameters)

    end_all_time = time.time()
    total_run_time = end_all_time - start_all_time
    print(f"Total running time: ", total_run_time, "seconds")
    return final_accuracy, round(average_inference_time * 1000, 3)


def load_train_evaluate_resnet(hyperparameters):
    layer_num = hyperparameters.get("layer_num", 34)
    num_classes = hyperparameters.get("num_classes", 10)
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
