from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience
%matplotlib inline

# The following code block contains an 8x8 matrix that will be used as a maze object:
maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

# This helper function allows a visual representation of the maze object:
def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img
  
# The pirate agent can move in four directions: left, right, up, and down.

# While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. 
# This is called "exploration" and is defined by epsilon. This value is typically a lower value such as 0.1, which means for every ten attempts, the agent will 
# attempt to learn by experience nine times and will randomly explore a new path one time. You are encouraged to try various values for the exploration factor and 
# see how the algorithm performs.
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 0.1

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)

# This function simulates a full game based on the provided trained model. The other parameters include the TreasureMaze object and the starting position of the pirate.
def play_game(model, qmaze, pirate_cell):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False
          
# This function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. 
# In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check.
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True
  
 def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

# This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results 
# in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.
ef qtrain(model, maze, **opt):

    # exploration factor
    global epsilon 

    # number of epochs
    n_epoch = opt.get('n_epoch', 15000)

    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)

    # maximum data size for training
    data_size = opt.get('data_size', 50)

    # start time
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze (see above)
    qmaze = TreasureMaze(maze)

    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)
    
    win_history = []   # history of win/lose game
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    
    # Code that I wrote myself 
    
    # For loop which loops through the number of epochs
    for epoch in range(n_epoch):     
        print('Starting new epoch baby yeah!')
        # Creating two random values that we can use to randomly select a cell
        random_x_value = np.random.randint(0, 7)
        random_y_value = np.random.randint(0, 7)
        agent_cell = ([random_x_value, random_y_value]) # randomly selecting a cell
        qmaze.reset([0,0]) # Reset the maze
        
        # Initializing some variables we'll use later
        loss = 0
        envstate = qmaze.observe() # envstate = Environment.current_state      
        n_episodes = 0
        
        # While loop that runs as long as the game status is "not over"
        while qmaze.game_status() == 'not_over':     
            
            previous_envstate = envstate
            # Getting a set of valid actions
            available_actions = qmaze.valid_actions() 
            
            # Getting the next action 
            # np.random.rand() will give us a random number between 0 and 1, we compare that to epsilon
            if np.random.rand() < epsilon:
                action = np.random.randint(0, len(available_actions)) # Randomly choosing action
            else:
                action = np.argmax(experience.predict(envstate))
            
            envstate, reward, game_status = qmaze.act(action)
            episode = [previous_envstate, action, reward, envstate, game_status]
            n_episodes += 1 # Increase the number of episodes by one each time an action is performed
            # Store episode in Experience replay object
            experience.remember(episode)
            # Train neural network model and evaluate loss
            inputs, targets = experience.get_data()
            history = model.fit(inputs, targets)
            # Calling model.evaluate to determine loss
            loss = model.evaluate(inputs, targets) # Returns the loss value & metrics values for the model
            
            game_outcome = episode[4] # episode[4] = game_status
            
            if game_outcome == 'win':
                win_history.append(1) # Appending win_history array
                win_rate = sum(win_history) / len(win_history) # Calculating win to loss ratio
                break
            elif game_outcome == 'lose': 
                win_history.append(0) # Appending win_history array
                win_rate = sum(win_history) / len(win_history) # Calculating win to loss ratio
                break
            # Given neither of these outcomes, the game is still in play
            
        if win_rate > epsilon:
            if completion_check(model, qmaze) == True: 
                print('Completion check passed woot woot baby yeah!')
                
    # Code that I wrote myself^

    #Print the epoch, loss, episodes, win count, and win rate for each epoch
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # We simply check if training has exhausted all free cells and if in all
        # cases the agent won.
        if win_rate > 0.9 : epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break
    
    
    # Determine the total time for training
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
      
qmaze = TreasureMaze(maze)
show(qmaze)
model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)
completion_check(model, qmaze)
show(qmaze)
pirate_start = (0, 0)
play_game(model, qmaze, pirate_start)
show(qmaze)
