from flask import Flask, render_template, url_for
import random
from PIL import ImageOps, Image
import numpy as np
app = Flask(__name__)

MAZE_SIZE = (21, 21)


def read_maze_from_file(filename):
    with Image.open(filename) as im:
        im = im.resize(MAZE_SIZE)
        # im.show()
    return np.array(ImageOps.grayscale(ImageOps.invert(im)))/255


def convert_maze_to_list(maze):
    return list(list(int(y) for y in x) for x in maze)


@app.route('/')
def display_maze():
    maze_id = random.randrange(21)
    maze = read_maze_from_file('archive/rectangular_mazes_10x10/'+str(maze_id)+'.png')
    maze = convert_maze_to_list(maze)
    return render_template("amazement.html", maze=str(maze), maze_id=maze_id)
