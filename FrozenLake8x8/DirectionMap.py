import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

# 'Enum'
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


OFFSET = 0.5
LEFT_DX = -1 # dx
DOWN_DY = -1 # dy
RIGHT_DX = 1 # dx
UP_DY = 1 # dy
NO_D = 0


'''
INDEX MAP
0   1   2   3   4   5   6   7
8   9   10  11  12  13  14  15
16  17  18  19  20  21  22  23
24  25  26  27  28  29  30  31
I'm not typing the rest of this
it just goes till it hits '63'.
You get the idea.
'''



# plot a direction map so you know what the best move is for each tile
def save_map(positions, qvalues, map_dim, name):
    directions = []
    arrows = []
    count = 0

    for qval in qvalues:
        directions += [np.argmax(qval)]

    # -1 for zero index
    x_dim = map_dim[1] # x == columns
    y_dim = map_dim[0] # y == rows

    for pos in positions:
        # gotta switch (0,0) from top left to bottom left
        x = pos%x_dim
        y = int((map_dim[0]-1) - int(np.floor(pos/y_dim)))

        if directions[count] == LEFT:
            arrows.append([x+RIGHT_DX, y+OFFSET, LEFT_DX, NO_D])
        elif directions[count] == DOWN:
            arrows.append([x+OFFSET, y+UP_DY, NO_D, DOWN_DY])
        elif directions[count] == RIGHT:
            arrows.append([x, y+OFFSET, RIGHT_DX, NO_D])
        elif directions[count] == UP:
            arrows.append([x+OFFSET, y, NO_D, UP_DY])
        else:  # assume left to prevent breaking the program
            arrows.append([x+RIGHT_DX, y+OFFSET, LEFT_DX, NO_D])
            print('ERROR DETERMINING DIRECTION')

        count += 1

    fig, ax = plt.subplots()
    for x, y, dx, dy in arrows:
        ax.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.05, head_length=0.1, length_includes_head=True)
        #ax.text(x+OFFSET, y+OFFSET, str(strength))

    spacing = 1
    majorLocator = plticker.MultipleLocator(spacing)
    ax.xaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_locator(majorLocator)
    ax.grid(markevery=2)
    plt.xlim(xmin=0, xmax=x_dim)
    plt.ylim(ymin=0, ymax=y_dim)
    plt.savefig('./tile_maps/' + name + '.png')
    plt.close()


if __name__ == '__main__':
    vals = []
    walkable = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14, 15]

    for i in walkable:
        vals += [[0, 0, 0, 1]]

    save_map(positions=walkable, qvalues=vals, map_dim=(4, 4), name='DQN10')
