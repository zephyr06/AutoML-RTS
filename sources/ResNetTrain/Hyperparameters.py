class Hyperparameters:
    def __init__(self):
        self.data_size_train = 1024
        self.data_size_test = 1024
        self.num_epochs = 15
        self.batch_size = 128
        self.learning_rate = 0.01
        self.num_classes = 10
        self.training_poison_chance = 0.0
        self.testing_data_noise = 0.0

    def print_hyperparameters(self):
        print("Hyperparameters:")
        print("data_size_train:", self.data_size_train)
        print("data_size_test:", self.data_size_test)
        print("num_epochs:", self.num_epochs)
        print("batch_size:", self.batch_size)
        print("learning_rate:", self.learning_rate)
        print("num_classes:", self.num_classes)
        print("training_poison_chance:", self.training_poison_chance)
        print("testing_data_noise:", self.testing_data_noise)


def get_hp_formal_cifar10():
    hp = Hyperparameters()
    hp.data_size_train = 50000
    hp.data_size_test = 10000
    hp.num_epochs = 10
    hp.batch_size = 128
    hp.learning_rate = 0.01
    hp.num_classes = 10
    return hp


def get_hp_formal_cifar10_resnet50():
    hp = Hyperparameters()
    hp.data_size_train = 50000
    hp.data_size_test = 10000
    hp.num_epochs = 20
    hp.batch_size = 32
    hp.learning_rate = 0.01
    hp.num_classes = 10
    return hp


def get_hp_test_cifar10_fast():
    hp = Hyperparameters()
    hp.data_size_train = 128
    hp.data_size_test = 128
    hp.num_epochs = 1
    hp.batch_size = 128
    hp.learning_rate = 0.01
    hp.num_classes = 10
    return hp


def get_hp_test_cifar10():
    hp = Hyperparameters()
    hp.data_size_train = int(1e4)
    hp.data_size_test = int(2e3)
    hp.num_epochs = 1
    hp.batch_size = 128
    hp.learning_rate = 0.01
    hp.num_classes = 10
    return hp
