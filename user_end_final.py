from joblib import load
import numpy as np
from pyamaze import *
from algorithims import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
# Example set of objects
available_algorithms = ['DFS', 'BFS', 'A*', 'Bi-directional', 'dijkstra', 'Greedy Best First Search']
codes = ['DFS-Effi', 'BFS-Effi', 'A*-Effi', 'Bi-Effi', 'dijkstra-Effi', 'GBest-Effi']
sizex, sizey, goalx, goaly, startx, starty, loop, pat = None, None, None, None, None, None, None, None
selected_algorithms = set()

# Load the saved model, scaler, and encoders
model = load('best_efficiency_classifier.joblib')
scaler = load('scaler.joblib')
pattern_encoder = load('pattern_encoder.joblib')
label_encoder = load('label_encoder.joblib')

# Creating Checkbox GUI
def create_checkbox_gui():
    global checkbox_vars
    global selected_algorithms
    global root

    root = tk.Tk()
    root.title("Select Algorithms")

    # Variables to track checkbox states
    checkbox_vars = [tk.BooleanVar() for _ in available_algorithms]
    selected_algorithms = set()

    # Create checkboxes
    for i, (alg, var) in enumerate(zip(available_algorithms, checkbox_vars)):
        tk.Checkbutton(root, text=alg, variable=var).pack(anchor='w')

    # Proceed button
    proceed_button = tk.Button(root, text="Proceed", command=proceed_with_selection)
    proceed_button.pack(pady=10)

    root.mainloop()

# Function to handle selection
def proceed_with_selection():
    global selected_algorithms
    for var, alg in zip(checkbox_vars, available_algorithms):
        if var.get():  # If the checkbox is checked
            selected_algorithms.add(codes[available_algorithms.index(alg)])
    root.destroy()  # Close the GUI after selection
    # You can pass the selected algorithms set to other functions for further processing

# Function to encode the pattern
def encode_pattern(pattern, encoder):
    # Ensure the encoder is fitted
    if encoder.classes_.size == 0:
        print("Encoder is not fitted.")
        return None
    return encoder.transform([pattern])[0]

# Function to predict maze efficiency
def predict_maze_efficiency(model, scaler, features, label_encoder, pattern_encoder):
    # Ensure the maze features are in the correct shape (2D array)
    maze_features = np.array(features).reshape(1, -1)

    # Scale the features without feature names
    scaler.fit(maze_features)
    scaled_features = scaler.transform(maze_features)

    # Get predictions
    prediction = model.predict(scaled_features)

    # Decode the prediction
    predicted_efficiency = decode_efficiency_class(prediction[0], label_encoder)
    return predicted_efficiency

# Function to decode efficiency classes
def decode_efficiency_class(encoded_efficiency, encoder):
    return encoder.inverse_transform([encoded_efficiency])[0]

def efficient_plotter(predicted_efficiency,mazes,agency):
    if predicted_efficiency == 'DFS-Effi':
        dfs = DFS(mazes,agency)
        path = dfs.fwdPath
        mazes.tracePath({agency: path}, delay=1)
        print(dfs)
    elif predicted_efficiency == 'BFS-Effi':
        bfs = BFS(mazes,agency)
        path= bfs.fwdPath
        mazes.tracePath({agency: path}, delay=10)
        print(bfs)

    elif predicted_efficiency == 'A*-Effi':
        astar = Astar(mazes,agency)
        path = astar.fwdPath
        mazes.tracePath({agency: path}, delay=10)
        print(astar)
    elif predicted_efficiency == 'Bi-Effi':
        bid = BiDirectionalSearch(mazes,agency)
        path = bid.fwdPath
        mazes.tracePath({agency: path}, delay=10)
        print(bid)

    elif predicted_efficiency == 'dijkstra-Effi':
        dijik = dijkstra(mazes,agency)
        path = dijik.fwdPath
        mazes.tracePath({agency: path}, delay=10)
        print(dijik)

    elif predicted_efficiency == 'GBest-Effi':
        gbes = GreedyBestFirstSearch(mazes,agency)
        path = gbes.fwdPath
        mazes.tracePath({agency: path}, delay=10)
        print(gbes)
def submit():
    global sizex, sizey, goalx, goaly, startx, starty, loop, pat
    try:
        if (not sizex1.get().strip() or not sizey1.get().strip() or not goalx1.get().strip() or 
            not startx1.get().strip() or not starty1.get().strip() or not goaly1.get().strip() or 
            not loop1.get().strip() or not pat1.get().strip()):
            raise ValueError('Please fill out all the parameters')
        # Get the value from the Entry widget
        sizex = int(sizex1.get())  # Call .get() on the Entry widget
        sizey = int(sizey1.get())
        goalx = int(goalx1.get())  # Call .get() on the Entry widget
        goaly = int(goaly1.get())
        startx = int(startx1.get())  # Call .get() on the Entry widget
        starty = int(starty1.get())
        loop = int(loop1.get())
        pat = str(pat1.get())
        # Validate the range
        if not (1 <= sizex <= 100):
            raise ValueError("Size (x-axis) must be in the range 1-100.")

        if not (1 <= sizey <= 100):
            raise ValueError("Size (y-axis) must be in the range 1-100.")
        
        if not (1 <= goalx<= sizex):
            raise ValueError(f"Goal (x-axis) must be in the range 1-{sizex}.")
        
        if not (1 <= goaly <= sizey):
            raise ValueError(f"Size (y-axis) must be in the range 1-{sizey}.")
        
        if not (1 <= startx<= sizex):
            raise ValueError(f"Goal (x-axis) must be in the range 1-{sizex}.")
        
        if not (1 <= starty<= sizey):
            raise ValueError(f"Size (y-axis) must be in the range 1-{sizey}.")

        if not (0 <= loop <= 100):
            raise ValueError("loop must be in the range 0-100.")

        if (pat.lower() not in 'hv'):
            raise ValueError("Enter the valid value for pattern (h,v).")
    except ValueError as e:
        # Show an error message for invalid input
        messagebox.showerror("Input Error", str(e))
def proceed():
    submit()
    root.destroy()
## Main Program Starts From here##

root = tk.Tk()
root.title('Maze Solver')

ttk.Label(root,text = 'Enter the size (x-axis) range(1-100)').grid(row=0, column=0, padx=10, pady=5, sticky="w")
sizex1 = ttk.Entry(root)
sizex1.grid(row=1, column=0, columnspan=2, pady=10)

ttk.Label(root,text = 'Enter the size (y-axis) range(1-100)').grid(row=0, column=20, padx=10, pady=5, sticky="e")
sizey1 = ttk.Entry(root)
sizey1.grid(row=1, column=20, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the endPoint (x-axis) range(1-size x)').grid(row=2, column=0, padx=10, pady=5, sticky="w")
goalx1 = ttk.Entry(root)
goalx1.grid(row=3, column=0, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the endpoint (y-axis) range(1- sizey)').grid(row=2, column=20, padx=10, pady=5, sticky="e")
goaly1 = ttk.Entry(root)
goaly1.grid(row=3, column=20, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the startPoint (x-axis) range(1-size x)').grid(row=4, column=0, padx=10, pady=5, sticky="w")
startx1 = ttk.Entry(root)
startx1.grid(row=6, column=0, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the startPoint (y-axis) range(1- sizey)').grid(row=4, column=20, padx=10, pady=5, sticky="e")
starty1 = ttk.Entry(root)
starty1.grid(row=6, column=20, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the loop percentage range(0- 100)').grid(row=7, column=0, padx=10, pady=5, sticky="w")
loop1 = ttk.Entry(root)
loop1.grid(row=8, column=0, columnspan=2, pady=10)

ttk.Label(root,text = f'Enter the Pattern (h for horizontal, v for vertical)').grid(row=7, column=20, padx=10, pady=5, sticky="e")
pat1 = ttk.Entry(root)
pat1.grid(row=8, column=20, columnspan=2, pady=10)

proceed_button = ttk.Button(root,text='Submit',command=submit)
proceed_button.grid(row=9,column=15,columnspan=2,pady=10)
proceed_button = ttk.Button(root,text='Proceed',command=proceed)
proceed_button.grid(row=9,column=10,columnspan=2,pady=10)
root.mainloop()
encoded_pat = encode_pattern(pat, pattern_encoder)


create_checkbox_gui()
# If encoding fails (None), exit
if encoded_pat is None:
    exit()

# Total size and encoded pattern
total_size = sizex * sizey

# Feature array
features = [sizex, sizey, total_size, goalx, goaly, startx, starty, loop, encoded_pat]

# Predict efficiency
predicted_efficiency = predict_maze_efficiency(model, scaler, features, label_encoder, pattern_encoder)
method = predicted_efficiency.split('-')

print(f'Estimated Method: {method[0]}')
m = maze(sizex, sizey)
m.CreateMaze(x=goalx, y=goaly, pattern=pat, loopPercent=loop)
effiag = agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.cyan)
dfsag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.blue)
bfsag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.green)
Astarag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.yellow)
biag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.red)
dijkstraag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.dark)
gbesag =  agent(m, x=startx, y=starty, footprints=True, filled=True,color = COLOR.light)

for alg in selected_algorithms:
    if alg == 'DFS-Effi' and predicted_efficiency!='DFS-Effi':
        efficient_plotter(alg, m, dfsag)
        dfs = textLabel(m,title='DFS Color is ',value='Blue')
    elif alg == 'BFS-Effi' and predicted_efficiency != 'BFS-Effi':
        efficient_plotter(alg, m, bfsag)
        bfs = textLabel(m,title='BFS Color is ',value='Green')
    elif alg == 'A*-Effi' and predicted_efficiency != 'A*-Effi':
        efficient_plotter(alg, m, Astarag)
        Astar = textLabel(m,title='A* Color is ',value='Yellow')
    elif alg == 'Bi-Effi' and predicted_efficiency!='Bi-Effi':
        efficient_plotter(alg, m, biag)
        bid = textLabel(m,title='Bidirectional Color is ',value='red')
    elif alg == 'dijkstra-Effi' and predicted_efficiency!='dijkstra-Effi':
        efficient_plotter(alg, m, dijkstraag)
        dij = textLabel(m,title='dijkstra Color is',value='white')

    elif alg == 'GBest-Effi' and predicted_efficiency!='GBest-Effi':
        efficient_plotter(alg, m, gbesag)
        gbe = textLabel(m,title='greedy best Color is ',value='Black & white')
efficient_plotter(predicted_efficiency,m,effiag)
best = textLabel(m,title=f'best algorithim is {method[0]} Color is ',value = 'Cyan')
m.run()