import csv

filename = "optimal_actions.csv"
maze_number = 21


def generate_grid_from_csv(filename, maze_number):
    grid = [[5 for _ in range(21)] for _ in range(21)]
    maze_found = False  # Flag to check if maze_number exists in the CSV

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            maze_idx, state_str, action = int(row[0]), row[1], int(row[2])
            if maze_idx == maze_number:
                maze_found = True
                x, y = map(int, state_str.strip('"')[1:-1].split(", "))
                grid[y][x] = action
                grid[19][20] = 9

    if not maze_found:
        print("maze not found!")
        return None

    return grid


grid = generate_grid_from_csv(filename, maze_number)

if grid is not None:
    for row in grid:
        print(' '.join(map(str, row)))
