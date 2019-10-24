import os


grid_size = 10

dir_graph = open("graph_"+str(grid_size)+"_dir", "w")

for row in range(0,grid_size):
    for col in range(0,grid_size-1):
        dir_graph.write(str(grid_size*row+col))
        dir_graph.write(" ")
        dir_graph.write(str(grid_size*row+col+1))
        dir_graph.write("\n")

for row in range(0, grid_size-1):
    for col in range(0, grid_size):
        dir_graph.write(str(grid_size*row+col))
        dir_graph.write(" ")
        dir_graph.write(str(grid_size*(row+1)+col))
        dir_graph.write("\n")
