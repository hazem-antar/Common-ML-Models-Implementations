import pygame
import sys
import random

# Define the constants
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 10, 10
SQUARE_SIZE = WIDTH // COLS

# Define the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class MazeEnv:
    '''
    Class of a manually designed 10x10 maze escaping environment
    '''
    def __init__(self):

        # Initialize pygame
        pygame.init()

        # Set the window display mode
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Set caption for the game
        pygame.display.set_caption("Maze Environment")

        # Define the maze specs (walls, terminals)
        self.maze = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        ]

        # Define agent position
        self.agent_pos = [0, 0]

        # Define action space size
        self.action_space_n = 4

        # Define observation space size
        self.observation_space_n = ROWS * COLS

    def draw_grid(self):
        '''
        Method to re-draw the grid every timestep 
        '''
        # Iterate over rows and columns of the maze
        for y in range(ROWS):
            for x in range(COLS):

                # Define a rectangle
                rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                
                # Initialze color for empty cell
                color = WHITE

                # Set color of the walls
                if self.maze[y][x] == 1:  
                    color = BLACK
                # Set color of terminals
                elif self.maze[y][x] == 2: 
                    color = GREEN

                # Draw the rectangle with the chosen color
                pygame.draw.rect(self.screen, color, rect)

                # Draw red circle if agent exist here
                if [y, x] == self.agent_pos: 
                    pygame.draw.circle(self.screen, RED, (x * SQUARE_SIZE + SQUARE_SIZE // 2, y * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 4)

    def move_agent(self, dx, dy):
        '''
        Method to move the agent position given delta x and y
        '''
        # Update the position by adding the deltas
        new_y = self.agent_pos[0] + dy
        new_x  = self.agent_pos[1] + dx

        # If valid new position
        if (0 <= new_x < COLS) and (0 <= new_y < ROWS) and (self.maze[new_y][new_x] != 1):
            # Update agent's position
            self.agent_pos = [new_y, new_x]
            
            # If reached a terminal
            if self.maze[new_y][new_x] == 2:
                return 10
            
            # If still in maze
            return -0.1
        
        # Not a valid new position (hitted a wall)
        return -1  

    def reset(self):
        '''
        Method to reset the environment
        '''
        # Find all free cells in the environment
        free_cells = [(y, x) for y in range(ROWS) for x in range(COLS) if self.maze[y][x] == 0]

        # Update agent's position by randomly selecting from the free cells
        self.agent_pos = random.choice(free_cells)  
        
        # Convert state from 2D form to 1D index
        state = self.agent_pos[0] * COLS + self.agent_pos[1]

        return state

    def step(self, action):
        '''
        Method to take a step in the environment
        '''
        # Map the action to delta x and delta y
        map_actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        dx, dy = map_actions[action]

        # Move the agent and get the reward
        reward = self.move_agent(dx, dy)

        # Convert state from 2D form to 1D index
        state = self.agent_pos[0] * COLS + self.agent_pos[1]

        # check if the agent reached a terminal and set done flag
        done = self.maze[self.agent_pos[0]][self.agent_pos[1]] == 2

        # Reset environment when done
        if done:
            self.reset() 
            
        return state, reward, done

    def render(self):
        '''
        Method to render the environment
        '''
        # Put white background
        self.screen.fill(WHITE)

        # Draw the grid
        self.draw_grid()

        # Update the objects locations on the display
        pygame.display.update()

        # Check if game ended
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

class MazeVisualizer:
    '''
    Helper class that visualizes the state-value function and policy on the
    actual maze grid. 
    '''
    def __init__(self, V = None, policy = None):

        # Initialize pygame
        pygame.init()

        # Set the window display mode
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

         # Set caption for the game
        pygame.display.set_caption("Maze Environment Visualization")
        
        # Define the maze specs (walls, terminals)
        self.maze = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        ]

        # Define source of data (Q-table / direct V(s) and PI(s))
        self.V = V
        self.policy = policy

        # Compute the size of a cell 
        self.cell_size = WIDTH // len(self.maze[0])

        # Set font size
        self.font = pygame.font.Font(None, 24)

        # Load arrow images
        self.arrows = {
            0: pygame.image.load('up.jpg').convert_alpha(),    # Up arrow
            1: pygame.image.load('right.jpg').convert_alpha(),  # Right arrow
            2: pygame.image.load('down.jpg').convert_alpha(),  # Down arrow
            3: pygame.image.load('left.jpg').convert_alpha()   # Left arrow
        }

        # Scale the images to fit the cells
        for key in self.arrows:
                self.arrows[key] = pygame.transform.scale(self.arrows[key], (self.cell_size, self.cell_size))

    def draw_environment(self):
        '''
        Method to re-draw the grid every timestep without the agent
        '''
        # Clear screen
        self.screen.fill(WHITE)  

        # Iterate over rows and columns of the maze
        for y in range(ROWS):
            for x in range(COLS):
                # Define a rectangle
                rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                
                # Initialze color for empty cell
                color = WHITE

                # Set color of the walls
                if self.maze[y][x] == 1:  
                    color = BLACK
                # Set color of terminals
                elif self.maze[y][x] == 2: 
                    color = GREEN

                # Draw the rectangle with the chosen color
                pygame.draw.rect(self.screen, color, rect)

    def draw_values(self):
        '''
        Method to plot the text values on each cell of the grid
        '''
        # Draw the background
        self.draw_environment()

        # Iterate over each cell
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                
                # Check if cell is net a wall or terminal
                if self.maze[y][x] == 0:

                    # Compute the pixel location of cell
                    cell_x = x * self.cell_size
                    cell_y = y * self.cell_size

                    # Convert state from 2D form to 1D indexing
                    state = y * COLS + x

                    # Retrieve the value of the state
                    value = self.V[state]
                    
                    # Plot a text on this cell with that value
                    self.text_on_cell(cell_x, cell_y, value)

    def draw_policy(self):
        '''
        Method to plot the arrow images on each cell of the grid
        '''
        # Draw the background
        self.draw_environment()

        # Iterate over each cell
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):

                # Check if cell is not a wall or terminal
                if self.maze[y][x] == 0:

                    # Compute the pixel location of cell
                    cell_x = x * self.cell_size
                    cell_y = y * self.cell_size

                    # Convert state from 2D form to 1D indexing
                    state = y * COLS + x

                    # Retrieve the best action in the state
                    best_action = self.policy[state]

                    # Plot an arrow on this cell pointing to the best action
                    self.arrow_on_cell(cell_x, cell_y, best_action)

    def text_on_cell(self, x, y, value):
        '''
        Method to plot a text on the center of a cell 
        '''

        # Render text to get it's size
        text_surface = self.font.render(f"{value:.2f}", True, BLACK)

        # Get the text size
        text_width, text_height = text_surface.get_size()

        # Calculate the centered position by subtracting half the text size from the middle of the cell
        text_x = x + (self.cell_size - text_width) // 2
        text_y = y + (self.cell_size - text_height) // 2

        # Blit the text in the center
        self.screen.blit(text_surface, (text_x, text_y))

    def arrow_on_cell(self, x, y, action):
        '''
        Method to plot an arrow on a cell 
        '''
        # Choose the arrow image based on the action
        arrow_image = self.arrows[action]

        # Blit the arrow
        self.screen.blit(arrow_image, (x, y))

    def start(self, delay = 3000, title = ""):
        '''
        Method to start the visuallizer
        '''
        # Display state values
        self.draw_values()
        pygame.display.flip()

        # Save the screen as an image
        pygame.image.save(self.screen, title + "_state_values.png")
        
        # Display for (t) seconds
        pygame.time.wait(delay)  

        # Display policy arrows
        self.draw_policy()
        pygame.display.flip()

        # Save the screen as an image
        pygame.image.save(self.screen, title + "_policy.png")

        # Display for (t) seconds
        pygame.time.wait(delay)  

        pygame.quit()
