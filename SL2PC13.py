""" 
Group C: Assignment No. 13
Assignment Title: MNIST Handwritten Character Detection using PyTorch,
Keras and Tensorflow
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import keras
import tensorflow as tf

def pytorch_mnist():
    # Hyperparameters
    input_size = 784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = x.reshape(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = NeuralNet(input_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

def keras_mnist():
    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=100)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc*100, "%")

def tensorflow_mnist():
    # Load the dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=100)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc*100, "%")

def main():
    while True:
        print("\nChoose an option:")
        print("1. PyTorch MNIST")
        print("2. Keras MNIST")
        print("3. TensorFlow MNIST")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            pytorch_mnist()
        elif choice == "2":
            keras_mnist()
        elif choice == "3":
            tensorflow_mnist()
        elif choice == "4":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
