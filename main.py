import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Draw:
    def __init__(self, master, data):
        self.master = master
        self.canvas_width = 280
        self.canvas_height = 280
        self.scale_to = 28

        self.data = data

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left")

        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_canvas)
        self.save_button.pack(side="left")

        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.exit)
        self.exit_button.pack(side="left")

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_handle = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 7
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw_handle.ellipse([x-r, y-r, x+r, y+r], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_handle = ImageDraw.Draw(self.image)

    def save_canvas(self):
        resized_image = self.image.resize((self.scale_to, self.scale_to))
        image_data = np.array(resized_image) / 255.0
        self.data.append(image_data)
        plt.imshow(image_data, cmap='gray')
        plt.show()

    def exit(self, event=None):
        print("Exiting...")
        self.master.quit()

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        # images should be [N, 28, 28], labels [N]
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# A simple CNN for MNIST: Conv -> ReLU -> Conv -> ReLU -> FC layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28x28 -> 14x14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14x14 -> 7x7
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Load MNIST data
    with gzip.open('mnist.pkl.gz', 'rb') as file:
        train_set, valid_set, test_set = pickle.load(file, encoding='latin1')

    train_images, train_labels = train_set
    test_images, test_labels = test_set

    train_images = train_images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    batch_size = 128
    learning_rate = 0.001
    epochs = 7

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    success_rates = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        success_rate = 100 * correct / total
        success_rates.append(success_rate)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Success Rate: {success_rate:.2f}%")

    print(f"Final Success Rate: {success_rates[-1]:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), success_rates, marker='o', label="Success Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate (%)")
    plt.title("Training Success Rate per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig("success_rate.png")

    # Drawing app
    data = []
    root = tk.Tk()
    app = Draw(root, data)
    root.mainloop()

    # Evaluate drawn images
    if data:
        model.eval()
        for idx, image in enumerate(data):
            # image is 28x28 numpy array
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
            predicted_label = torch.argmax(output, dim=1).item()

            print(f"Drawing {idx+1}:")
            print(f"Predicted digit is {predicted_label}")
            for i, prob in enumerate(probabilities):
                print(f"{i} - {prob * 100:.2f}%")
