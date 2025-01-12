import pandas as pd
from pyamaze import maze, agent
from algorithims import *
import random

def printer(path, effi, title):
    length = len(path)
    print(title)
    print(f'\npath length = {length}\nefficiency = {effi}')

def calculate_efficiency(execution_time, path_length, cells_searched, weights=None):
    if weights is None:
        weights = {'time': 1, 'path': 1, 'cells': 1}

    normalized_time = execution_time / (execution_time + 1)
    normalized_path = path_length / (path_length + 1)
    normalized_cells = cells_searched / (cells_searched + 1)

    efficiency_score = (
        weights['time'] * normalized_time +
        weights['path'] * normalized_path +
        weights['cells'] * normalized_cells
    )
    
    return efficiency_score

# Initialize dataframe
column = ['sizex', 'sizey', 'total_size', 'goalx', 'goaly', 'startx', 'starty', 'loop', 'pattern',
          'DFS-Effi', 'BFS-Effi', 'A*-Effi', 'Bi-Effi', 'dijkstra-Effi', 'GBest-Effi']
data = pd.DataFrame(columns=column)

# Number of iterations
num_iterations = 10

# While loop
iteration = 0
while iteration < num_iterations:
    # Generate random maze parameters
    sizex = random.randint(10, 100)
    sizey = random.randint(10, 100)
    goalx = random.randint(5, sizex)
    goaly = random.randint(5, sizey)
    startx = random.randint(1, sizex)
    starty = random.randint(1, sizey)
    loop = random.randint(0, 100)
    pat = 'h' if random.randint(0, 1) == 0 else 'v'

    # Create maze, agent, and searcher
    m = maze(sizex, sizey)
    m.CreateMaze(x=goalx, y=goaly, loopPercent=loop, pattern=pat)
    a = agent(m, x=startx, y=starty, footprints=True, filled=True)

    # Perform searches
    dfs = DFS(m,a)
    bfs = BFS(m,a)
    astar = Astar(m,a)
    bi = BiDirectionalSearch(m,a)
    dijik = dijkstra(m,a)
    gbes = GreedyBestFirstSearch(m,a)
    dfsPath, dsearched, dfseffi = dfs.fwdPath,dfs.searched_cells,dfs.efficiency
    bfsPath, bsearched, bfseffi = bfs.fwdPath,bfs.cells_searched,bfs.efficiency
    aPath, asearched, aeffi = astar.fwdPath,astar.cells_searched,astar.efficiency
    biPath, bisearched, bieffi = bi.fwdPath,bi.cells_searched,bi.efficiency
    dijkPath, dijikstrasearched, dijikstraeffi = dijik.fwdPath,dijik.cells_searched,dijik.efficiency
    gbestPath, gbessearched, gbesteffi = gbes.fwdPath,gbes.cells_searched,gbes.efficiency

    # Calculate efficiencies
    dfseffi = calculate_efficiency(dfseffi, len(dfsPath), dsearched)
    bfseffi = calculate_efficiency(bfseffi, len(bfsPath), bsearched)
    aeffi = calculate_efficiency(aeffi, len(aPath), asearched)
    bieffi = calculate_efficiency(bieffi, len(biPath), bisearched) # bidirectional returns number of cells
    dijikstraeffi = calculate_efficiency(dijikstraeffi, len(dijkPath),dijikstrasearched)
    gbesteffi = calculate_efficiency(gbesteffi, len(gbestPath),gbessearched)
    
    efficient = {'DFS-Effi': dfseffi,
        'BFS-Effi': bfseffi,
        'A*-Effi': aeffi,
        'Bi-Effi': bieffi,
        'dijkstra-Effi': dijikstraeffi,
        'GBest-Effi': gbesteffi}
    best_efficient = min(efficient,key=efficient.get)

    # Collect data into a dictionary
    collector = {
        'sizex': sizex,
        'sizey': sizey,
        'total_size': sizex * sizey,
        'goalx': goalx,
        'goaly': goaly,
        'startx': startx,
        'starty': starty,
        'loop': loop,
        'pattern': pat,
        'DFS-Effi': dfseffi,
        'BFS-Effi': bfseffi,
        'A*-Effi': aeffi,
        'Bi-Effi': bieffi,
        'dijkstra-Effi': dijikstraeffi,
        'GBest-Effi': gbesteffi,
        'Best-Effi' : best_efficient
    }

    # Append to the dataframe
    data = data._append(collector, ignore_index=True)

    # Increment iteration
    iteration += 1

    # Print results for this iteration
    print(iteration)
# Final dataframe
print(data)
data.to_csv('trial2.csv',index=True)