from pyamaze import *
from queue import PriorityQueue
import time

class searcher():
    """
    A class containing building blocks of all algorithims.

    Attributes:
        Maze: An instance of the maze.
        Agent: The agent navigating the maze.
    """
    def __init__(self, Maze, Agent):
        """
        Initializes the searcher class with the given maze and agent.

        Args:
            Maze: An instance of the maze.
            Agent: The agent navigating the maze.
        """
        self.Maze = Maze
        self.Agent = Agent
        self.start = (self.__Agent.x,self.__Agent.y)
        self.goal = self.__Maze._goal
    @property
    def Maze(self):
        return self.__Maze
    @Maze.setter
    def Maze(self,Maze):
        if(isinstance(Maze,maze)):
            self.__Maze = Maze
        else:
            raise ValueError('Maze must be a pyamaze maze object')
    @property
    def Agent(self):
        return self.__Agent
    @Agent.setter
    def Agent(self,Agent):
        if(isinstance(Agent,agent)):
            self.__Agent = Agent
        else:
            raise ValueError('Agent must be a pyamaze Agent object')
    
    def fwdpather(self, path): 
        """
        Reconstructs the path from the given dictionary path.

        Args:
            path (dict): Dictionary containing the path from goal to start.

        Returns:
            fwdPath (dict): The path from start to goal.
        """
        cell = self.goal
        fwdPath = {}
        while cell != self.start:
            fwdPath[path[cell]] = cell
            cell = path[cell]
        return fwdPath
    @staticmethod
    def h(cell, goal): 
        """
        Computes the heuristic for A* search.

        Args:
            cell (tuple): The current cell position.
            goal (tuple): The goal cell position.

        Returns:
            int: The heuristic value.
        """
        x1, y1 = cell
        xg, yg = goal
        return abs(x1 - xg) + abs(y1 - yg)
    @staticmethod
    def move_in_direction(currCell, d):
        if d == 'E':
            childCell = (currCell[0], currCell[1] + 1)
        elif d == 'S':
            childCell = (currCell[0] + 1, currCell[1])
        elif d == 'N':
            childCell = (currCell[0] - 1, currCell[1])
        elif d == 'W':
            childCell = (currCell[0], currCell[1] - 1)
        return childCell
    @staticmethod
    def rotate_direction_left(Direction):
        K = list(Direction.keys())
        V = list(Direction.values())
        V_rotated = [V[-1]] + V[:-1]
        return dict(zip(K, V_rotated))
    @staticmethod
    def rotate_direction_right(Direction):
        K = list(Direction.keys())
        V = list(Direction.values())
        V_rotated = V[1:] + [V[0]]
        return dict(zip(K, V_rotated))
    @staticmethod
    def move_Forward(cell, Direction):
        if Direction["Forward"] == "E":
            return (cell[0], cell[1] + 1), "E"
        elif Direction["Forward"] == "W":
            return (cell[0], cell[1] - 1), "W"
        elif Direction["Forward"] == "N":
            return (cell[0] - 1, cell[1]), "N"
        elif Direction["Forward"] == "S":
            return (cell[0] + 1, cell[1]), "S"
class DFS(searcher):
    """
    Performs Depth-First Search (DFS) to find a path in the maze.

    Returns:
        fwdPath (dict): The shortest path from the start to the goal.
        dfsPath (dict): All explored paths during the DFS search.
    """
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        self.fwdPath = None
        self.cells_searched = None
        self.efficiency = None
        self.perform_search()

    def perform_search(self):
        
        start_time = time.time()
        start = (self.Agent.x, self.Agent.y)  # Starting position of the agent.
        explored = [start]  # List to keep track of explored cells.
        frontier = [start]  # Stack to manage the DFS frontier.
        dfsPath = {}  # To keep track of the path from one cell to the next.

        while len(frontier) > 0:
            currCell = frontier.pop()  # Get the last cell added to the frontier.
            if currCell == self.goal:  # Check if the goal is reached.
                break
            for d in 'ESNW':  # Check all possible directions.
                if self.Maze.maze_map[currCell][d] == True:  # If there is a valid path in the direction.
                    childCell = self.move_in_direction(currCell,d)
                    if childCell in explored:
                        continue  # Skip if the cell is already explored.
                    else:
                        frontier.append(childCell)  # Add the cell to the frontier.
                        explored.append(childCell)  # Mark it as explored.
                        dfsPath[childCell] = currCell   # Record the path. It is actually all cells searched by algorithim
                        # and contains path in reverse order i.e from goal to start
        self.fwdPath = self.fwdpather(dfsPath)
        self.cells_searched = len(dfsPath)
        self.efficiency = time.time() - start_time
    def __call__(self):
        """Return the results directly when the object is called."""
        return self.fwdPath, self.cells_searched, self.efficiency
    def __repr__(self):
        return f'Dept First Search:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'
class BFS(searcher):
    """
    Performs Breadth-First Search (BFS) to find a path in the maze.

    Returns:
        fwdPath (dict): The shortest path from the start to the goal.
        bfsPath (dict): All explored paths during the BFS search.
        efficiency : Time taken by function
    """
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        self.fwdPath = None
        self.cells_searched = None
        self.efficiency = None
        self.perform_search()
    def perform_search(self):
        start_time = time.time()
        start = (self.Agent.x, self.Agent.y)  # Starting position of the agent.
        explored = [start]  # List to keep track of explored cells.
        frontier = [start]  # Queue to manage the BFS frontier.
        bfsPath = {}  # To keep track of the path from one cell to the next.

        while len(frontier) > 0:
            currCell = frontier.pop(0)  # Get the first cell in the frontier.
            if currCell == self.goal:  # Check if the goal is reached.
                break
            for d in 'EWSN':  # Check all possible directions.
                if self.Maze.maze_map[currCell][d] == True:  # If there is a valid path in the direction.
                    childCell = self.move_in_direction(currCell,d)
                    if childCell in explored:
                        continue  # Skip if the cell is already explored.
                    else:
                        frontier.append(childCell)  # Add the cell to the frontier.
                        explored.append(childCell)  # Mark it as explored.
                        bfsPath[childCell] = currCell  # Record the path. It is actually all cells searched by algorithim
                        # and contains path in reverse order i.e from goal to start

        self.fwdPath = self.fwdpather(bfsPath)
        end_time = time.time()
        self.efficiency = end_time - start_time
        self.cells_searched = len(bfsPath)
    def __call__(self):
        return self.fwdPath,len(self.cells_searched),self.efficiency
    def __repr__(self):
        return f'Breadth First Search:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'
class Astar(searcher):
    """
        Performs A* search algorithm to find a path in the maze.

        Returns:
            fwdPath (dict): The shortest path from the start to the goal.
            apath (dict): All explored paths during the A* search.
    """
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        self.fwdPath = None
        self.cells_searched = None
        self.efficiency = None
        self.perform_search()

    def perform_search(self):
        start_time = time.time()
        start = (self.Agent.x, self.Agent.y)  # Starting position of the agent.
        g_score = {cell:float('inf') for cell in self.Maze.grid} # sets initial cost of each
        # cell to infinity
        g_score[start] = 0 #sets the g score of starting point as 0 as we are already at starting point
        f_score = {cell:float('inf') for cell in self.Maze.grid}
        f_score[start] = searcher.h(start,self.goal) #the f cost of starting will be as f(n) = g(n) + h(n) here g(n) is 0 so f(n) = h(n)
        apath = {}
        open = PriorityQueue()
        open.put((f_score[start],f_score[start],start))

        while not open.empty():
            currCell = open.get()[2]
            if currCell == self.Maze._goal:
                break
            else:
                for d in 'EWSN':  # Check all possible directions.
                    if self.Maze.maze_map[currCell][d] == True:  # If there is a valid path in the direction.
                        childCell = self.move_in_direction(currCell,d)
                        temp_g_score = g_score[currCell]+1
                        temp_h_score = searcher.h(childCell,(self.Maze._goal))
                        temp_f_score = temp_g_score + temp_h_score
                        if temp_f_score < f_score[childCell]:
                            g_score[childCell] = temp_g_score
                            f_score[childCell] = temp_f_score
                            open.put((temp_f_score,temp_h_score,childCell))
                            apath[childCell] = currCell
        self.fwdPath = self.fwdpather(apath)
        end_time = time.time()
        self.efficiency = end_time - start_time
        self.cells_searched = len(apath)
    def __call__(self):
        return self.fwdPath,len(self.Apath),self.efficiency
    def __repr__(self):
        return f'A*:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'
    # Dijkstra's algorithm implementation
class dijkstra(searcher):
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        self.fwdPath = None
        self.cells_searched = None
        self.efficiency = None
        self.perform_search()
    def perform_search(self):
        start_time = time.time()
        unvisited = {n:float('inf') for n in self.Maze.grid}
        unvisited[self.start] = 0
        visited = {}
        dijkstraPath = {}
        while unvisited:
            currCell = min(unvisited,key=unvisited.get)
            visited[currCell] = unvisited[currCell]
            if currCell == self.goal:
                break
            for d in 'EWNS':
                if self.Maze.maze_map[currCell][d] == True:  # If there is a valid path in the direction.
                    childCell = self.move_in_direction(currCell,d)
                    if childCell in visited:
                        continue
                    tempDist = unvisited[currCell] + 1
                    if tempDist < unvisited[childCell]:
                        unvisited[childCell] = tempDist
                        dijkstraPath[childCell] = currCell 
            unvisited.pop(currCell)
        self.fwdPath = self.fwdpather(dijkstraPath)
        end_time = time.time()
        self.efficiency = end_time - start_time
        self.cells_searched = len(dijkstraPath)
    def __call__(self):
        return self.fwdPath,len(self.cells_searched),self.efficiency
    def __repr__(self):
        return f'Dijikstra:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'

class BiDirectionalSearch(searcher):
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        self.perform_search()

    def perform_search(self):
        start_time = time.time()
        start = (self.Agent.x, self.Agent.y)
        goal = self.Maze._goal

        # Initialize frontiers and visited sets
        frontier_start = [start]
        frontier_goal = [goal]
        visited_start = {start: None}
        visited_goal = {goal: None}
        searchpath_start = {start}
        searchpath_goal = {goal}

        # Explore both sides simultaneously
        while frontier_start and frontier_goal:
            # Forward direction from start
            currCell_start = frontier_start.pop(0)
            for direction in 'ESNW':
                if self.Maze.maze_map[currCell_start][direction]:
                    childCell = self.move_in_direction(currCell_start,direction)

                    if childCell not in visited_start:
                        visited_start[childCell] = currCell_start
                        frontier_start.append(childCell)
                        searchpath_start.add(childCell)

                        # Check if the cell is in the goal frontier
                        if childCell in visited_goal:
                            end_time = time.time()
                            efficiency = end_time - start_time
                            total_cells_searched = len(searchpath_start | searchpath_goal)
                            self.fwdPath = self.reconstructPath(visited_start, visited_goal, childCell)
                            self.cells_searched = len(searchpath_start | searchpath_goal)
                            self.efficiency = efficiency
                            return self.fwdPath, self.cells_searched, self.efficiency

            # Backward direction from goal
            currCell_goal = frontier_goal.pop(0)
            for direction in 'ESNW':
                if self.Maze.maze_map[currCell_goal][direction]:
                    childCell = self.move_in_direction(currCell_goal,direction)

                    if childCell not in visited_goal:
                        visited_goal[childCell] = currCell_goal
                        frontier_goal.append(childCell)
                        searchpath_goal.add(childCell)

                        # Check if the cell is in the start frontier
                        if childCell in visited_start:
                            end_time = time.time()
                            efficiency = end_time - start_time
                            total_cells_searched = len(searchpath_start | searchpath_goal)
                            self.fwdPath = self.reconstructPath(visited_start, visited_goal, childCell)
                            self.cells_searched = total_cells_searched
                            self.efficiency = efficiency
                            return self.fwdPath, self.cells_searched, self.efficiency

        # If no path is found
        end_time = time.time()
        efficiency = end_time - start_time
        total_cells_searched = len(searchpath_start | searchpath_goal)
        self.fwdPath = None
        self.cells_searched = len(searchpath_start | searchpath_goal)
        self.efficiency = efficiency
        return self.fwdPath, self.cells_searched,self.efficiency

    def reconstructPath(self, visited_start, visited_goal, meet_point):
        # Reconstruct path from start to meet point
        path_from_start = []
        curr = meet_point
        while curr is not None:
            path_from_start.append(curr)
            curr = visited_start[curr]
        path_from_start.reverse()
        
        # Reconstruct path from goal to meet point
        path_from_goal = []
        curr = meet_point
        while curr is not None:
            path_from_goal.append(curr)
            curr = visited_goal[curr]
        
        # Combine the paths
        return path_from_start[:-1] + path_from_goal
    def __repr__(self):
        return f'Bidirectional Search:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'
class GreedyBestFirstSearch(searcher):
    def __init__(self, Maze, Agent):
        super().__init__(Maze, Agent)
        """
        Performs Greedy Best-First Search to find a path in the maze.

        Returns:
            fwdPath (dict): The shortest path from the start to the goal.
            gbfsPath (dict): All explored paths during the Greedy Best-First Search.
        """
        self.fwdPath = None
        self.cells_searched = None
        self.efficiency = None
        self.perform_search()
    def perform_search(self):
        start_time = time.time()
        start = (self.Agent.x, self.Agent.y)  # Starting position of the agent
        gbfsPath = {}  # Dictionary to store the path
        open_set = PriorityQueue()  # Priority queue for open nodes
        open_set.put((0, start))  # Add the start node with priority 0
        explored = set()  # Set to keep track of explored nodes

        while not open_set.empty():
            currCell = open_set.get()[1]  # Get the cell with the lowest heuristic value

            # If the goal is reached, exit the loop
            if currCell == self.goal:
                break

            # Mark the current cell as explored
            explored.add(currCell)

            # Explore neighbors in all possible directions
            for d in 'ESNW':
                if self.Maze.maze_map[currCell][d] == True:  # If there is a valid path in this direction
                    childCell = self.move_in_direction(currCell, d)

                    # If the child cell has not been explored
                    if childCell not in explored:
                        heuristic = searcher.h(childCell, self.goal)  # Calculate the heuristic
                        open_set.put((heuristic, childCell))  # Add to the priority queue
                        gbfsPath[childCell] = currCell  # Record the path

        # Reconstruct the forward path
        self.fwdPath = self.fwdpather(gbfsPath)
        end_time = time.time()
        self.efficiency = end_time - start_time
        self.cells_searched = len(gbfsPath)
        return self.fwdPath,self.cells_searched,self.efficiency
    def __repr__(self):
        return f'Greedy Best First Search:\nTime Taken:\t{self.efficiency}\nCells searched:\t{self.cells_searched}\nPath length:\t{len(self.fwdPath)}'
if __name__ == '__main__':
    m = maze(10,10)
    m.CreateMaze()
    ags = agent(m,filled=True,footprints=True)
    path2 = BiDirectionalSearch(m,ags)
    path = path2.fwdPath
    searched_cell = path2.cells_searched
    efficiency = path2.efficiency
    print((searched_cell))
    m.tracePath({ags:path},delay =10)
    m.run()