import csv


def generate_grid_from_csv(filename, maze_number):
    grid = [[5 for _ in range(21)] for _ in range(21)]

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            maze_idx, state_str, action = int(row[0]), row[1], int(row[2])
            if maze_idx == maze_number:
                x, y = map(int, state_str.strip('"')[1:-1].split(", "))
                grid[y][x] = action
    grid[19][20] = 9

    return grid


filename = "optimal_actions.csv"
maze_number = 22
grid = generate_grid_from_csv(filename, maze_number)

# Print the grid to terminal
for row in grid:
    print(' '.join(map(str, row)))
