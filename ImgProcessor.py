import os
import numpy as np
from PIL import Image
import json


class MazeDataset:
    def __init__(self, directory):
        self.directory = directory
        self.file_list = os.listdir(self.directory)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.file_list[idx])

        # Load the image
        img = Image.open(img_path)

        # Resize the image
        new_size = (img.width // 20, img.height // 20)
        img = img.resize(new_size, Image.ANTIALIAS)

        # Convert the image to grayscale
        img = img.convert('L')

        # Binarize the image
        threshold = 128
        img = img.point(lambda p: 0 if p > threshold else 1)

        # Convert the image to a NumPy array
        img_array = np.array(img)

        return img_array, idx

    def print_maze(self, maze_array, idx):
        print(f"Maze {idx}:")
        print("maze = np.array([")
        for row in maze_array:
            print("    " + str(row.tolist()) + ",")
        print("])  # 0 = free, 1 = occupied")


def write_mazes_to_json(mazes, filename):
    with open(filename, 'w') as f:
        json.dump({"mazes": mazes}, f)


mazes = []
dataset = MazeDataset(directory=r'C:\Users\safem\Desktop\Fall 2023\CSCE5214\MazeSolver\Dataset\archive\NothingButMazes')
for idx in range(len(dataset)):
    maze_array, label = dataset[idx]
    mazes.append(maze_array.tolist())

write_mazes_to_json(mazes, 'all_mazes.json')

