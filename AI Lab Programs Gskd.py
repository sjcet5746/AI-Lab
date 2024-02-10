#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Program-1
from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs_util(self, v, visited):
        visited.add(v)
        print(v, end=' ')

        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self.dfs_util(neighbor, visited)

    def dfs(self, start):
        visited = set()
        self.dfs_util(start, visited)

    def bfs(self, start):
        visited = set()
        queue = [start]

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                print(vertex, end=' ')
                visited.add(vertex)
                queue.extend([neighbor for neighbor in self.graph[vertex] if neighbor not in visited])


# Example Usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("DFS:")
g.dfs(2)
print("\nBFS:")
g.bfs(2)


# In[2]:


#Program-2
import sys

class TSP:
    def __init__(self, n):
        self.n = n
        self.graph = [[0 for _ in range(n)] for _ in range(n)]
    
    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight
        self.graph[v][u] = weight
    
    def nearest_neighbor(self, start):
        visited = [False] * self.n
        path = [start]
        visited[start] = True
        total_distance = 0
        
        for _ in range(self.n - 1):
            min_dist = sys.maxsize
            nearest_city = None
            current_city = path[-1]
            for next_city in range(self.n):
                if not visited[next_city] and self.graph[current_city][next_city] < min_dist:
                    min_dist = self.graph[current_city][next_city]
                    nearest_city = next_city
            path.append(nearest_city)
            visited[nearest_city] = True
            total_distance += min_dist
        
        total_distance += self.graph[path[-1]][start]  # Return to the starting city
        return path, total_distance

# Example Usage:
tsp = TSP(4)
tsp.add_edge(0, 1, 10)
tsp.add_edge(0, 2, 15)
tsp.add_edge(0, 3, 20)
tsp.add_edge(1, 2, 35)
tsp.add_edge(1, 3, 25)
tsp.add_edge(2, 3, 30)

start_city = 0
path, total_distance = tsp.nearest_neighbor(start_city)
print("Shortest Path:", path)
print("Total Distance:", total_distance)


# In[ ]:


import numpy as np
import random
import math


class SimulatedAnnealingTSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances
        self.num_cities = len(cities)
    
    def initial_solution(self):
        return random.sample(self.cities, len(self.cities))
    
    def total_distance(self, path):
        total_distance = 0
        for i in range(len(path)):
            total_distance += self.distances[path[i-1]][path[i]]
        return total_distance
    
    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temperature)
    
    def simulated_annealing(self, initial_temperature=1000, cooling_rate=0.003, stopping_temperature=1e-8):
        current_solution = self.initial_solution()
        current_cost = self.total_distance(current_solution)
        
        temperature = initial_temperature
        
        while temperature > stopping_temperature:
            new_solution = current_solution[:]
            i, j = sorted(random.sample(range(self.num_cities), 2))
            new_solution[i:j+1] = reversed(new_solution[i:j+1])
            new_cost = self.total_distance(new_solution)
            
            if self.acceptance_probability(current_cost, new_cost, temperature) > random.random():
                current_solution = new_solution
                current_cost = new_cost
            
            temperature *= 1 - cooling_rate
        
        return current_solution, current_cost


# Example Usage:
cities = ["A", "B", "C", "D"]
distances = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

tsp = SimulatedAnnealingTSP(cities, distances)
shortest_path, shortest_distance = tsp.simulated_annealing()
print("Shortest Path:", shortest_path)
print("Shortest Distance:", shortest_distance)


# In[4]:


#Program-4
import random

class WumpusWorld:
    def __init__(self, size=4, pit_prob=0.2, wumpus_prob=0.1):
        self.size = size
        self.grid = [[None] * size for _ in range(size)]
        self.agent_location = (0, 0)
        self.generate_world(pit_prob, wumpus_prob)

    def generate_world(self, pit_prob, wumpus_prob):
        # Place gold
        gold_x, gold_y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        self.grid[gold_x][gold_y] = 'G'

        # Place pits
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) != (gold_x, gold_y):
                    if random.random() < pit_prob:
                        self.grid[x][y] = 'P'

        # Place wumpus
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) != (gold_x, gold_y) and not self.has_pit(x, y):
                    if random.random() < wumpus_prob:
                        self.grid[x][y] = 'W'

    def has_pit(self, x, y):
        return self.grid[x][y] == 'P'

    def has_wumpus(self, x, y):
        return self.grid[x][y] == 'W'

    def is_gold(self, x, y):
        return self.grid[x][y] == 'G'

    def is_safe(self, x, y):
        return x >= 0 and y >= 0 and x < self.size and y < self.size and not self.has_pit(x, y) and not self.has_wumpus(x, y)

    def move(self, direction):
        x, y = self.agent_location
        if direction == 'up':
            x -= 1
        elif direction == 'down':
            x += 1
        elif direction == 'left':
            y -= 1
        elif direction == 'right':
            y += 1
        if self.is_safe(x, y):
            self.agent_location = (x, y)
            print("Moved to:", self.agent_location)
            if self.is_gold(x, y):
                print("Found gold!")
        else:
            print("Can't move there, it's unsafe!")

# Example Usage:
world = WumpusWorld()
print(world.grid)  # Print the initial world
world.move('right')  # Move to the right


# In[5]:


#Program-5
import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, move=None, depth=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.score = self.compute_score()

    def __lt__(self, other):
        return self.score < other.score

    def compute_score(self):
        return self.depth + self.manhattan_distance()

    def manhattan_distance(self):
        distance = 0
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # Goal state
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 0:
                    x, y = divmod(self.state[i][j] - 1, 3)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def generate_neighbors(self):
        neighbors = []
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Possible moves: right, left, down, up
        for dx, dy in moves:
            new_x, new_y = self.find_blank()
            new_x += dx
            new_y += dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_state = [row[:] for row in self.state]
                new_state[new_x][new_y], new_state[self.find_blank()] = new_state[self.find_blank()], new_state[new_x][new_y]
                neighbors.append(PuzzleNode(new_state, self, (new_x, new_y), self.depth + 1))
        return neighbors

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j


def a_star(start_state):
    start_node = PuzzleNode(start_state)
    if start_node.manhattan_distance() == 0:
        return start_node

    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, start_node)

    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        visited.add(tuple(map(tuple, current_node.state)))

        if current_node.state == [[1, 2, 3], [4, 5, 6], [7, 8, 0]]:
            return current_node

        for neighbor in current_node.generate_neighbors():
            if tuple(map(tuple, neighbor.state)) not in visited:
                heapq.heappush(priority_queue, neighbor)

# Example Usage:
initial_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

solution_node = a_star(initial_state)

# Trace back the solution path
solution_path = []
while solution_node:
    solution_path.append(solution_node.state)
    solution_node = solution_node.parent

# Print the solution path in reverse order
for state in reversed(solution_path):
    for row in state:
        print(row)
    print()


# In[6]:


#Program-6
def towers_of_hanoi(n, source, auxiliary, target):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    towers_of_hanoi(n - 1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    towers_of_hanoi(n - 1, auxiliary, source, target)

# Example Usage:
num_disks = 3
source_rod = "A"
auxiliary_rod = "B"
target_rod = "C"
towers_of_hanoi(num_disks, source_rod, auxiliary_rod, target_rod)


# In[8]:


#Program-7
import heapq

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_edge(self, u, v, weight):
        if u not in self.vertices:
            self.vertices[u] = []
        self.vertices[u].append((v, weight))

    def a_star(self, start, goal):
        frontier = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}

        while frontier:
            current_cost, current_node = heapq.heappop(frontier)

            if current_node == goal:
                break

            for neighbor, cost in self.vertices[current_node]:
                new_cost = cost_so_far[current_node] + cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current_node

        path = self.reconstruct_path(came_from, start, goal)
        return path

    def heuristic(self, current, goal):
        # Example heuristic: Manhattan distance
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

# Example Usage:
graph = Graph()
graph.add_edge((0, 0), (0, 1), 1)
graph.add_edge((0, 1), (1, 1), 1)
graph.add_edge((1, 1), (1, 0), 1)
graph.add_edge((1, 0), (0, 0), 1)
graph.add_edge((0, 0), (1, 1), 2)
graph.add_edge((0, 1), (1, 0), 2)

start = (0, 0)
goal = (1, 1)
path = graph.a_star(start, goal)
print("Shortest Path:", path)


# In[9]:


#Program-8
import random

# Objective function (fitness function)
def objective_function(solution):
    return sum(solution)

# Generate a random solution
def generate_solution(size):
    return [random.randint(0, 9) for _ in range(size)]

# Make a small change to the solution
def mutate_solution(solution):
    index = random.randint(0, len(solution) - 1)
    solution[index] = random.randint(0, 9)

# Hill Climbing algorithm
def hill_climbing(size, max_iter):
    current_solution = generate_solution(size)
    current_fitness = objective_function(current_solution)
    
    for _ in range(max_iter):
        new_solution = current_solution[:]
        mutate_solution(new_solution)
        new_fitness = objective_function(new_solution)
        
        if new_fitness > current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    
    return current_solution, current_fitness

# Example Usage:
solution_size = 5
max_iterations = 1000

best_solution, best_fitness = hill_climbing(solution_size, max_iterations)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)


# In[ ]:


#Program-9
import boto3

# Initialize AWS client
lex_client = boto3.client('lex-runtime', region_name='us-east-1')

# Function to interact with AWS Lex
def send_message(message):
    response = lex_client.post_text(
        botName='YourLexBotName',
        botAlias='YourLexBotAlias',
        userId='user',
        inputText=message
    )
    return response['message']

# Example Usage
while True:
    user_input = input("You: ")
    response = send_message(user_input)
    print("Bot:", response)


# In[ ]:


#Program-10
# Define a dictionary to store information about your college
college_info = {
    "programs": ["Computer Science", "Engineering", "Business", "Psychology"],
    "departments": ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Business Administration", "Psychology"],
    "faculty": {
        "Computer Science": ["Dr. Smith", "Dr. Johnson", "Dr. Lee"],
        "Engineering": ["Dr. Brown", "Dr. White", "Dr. Garcia"],
        "Business": ["Dr. Taylor", "Dr. Martinez", "Dr. Anderson"],
        "Psychology": ["Dr. Clark", "Dr. Wilson", "Dr. Lewis"]
    },
    "facilities": ["Library", "Gym", "Laboratories", "Student Center"],
    "events": ["Open House", "Career Fair", "Alumni Reunion"],
    # Add more information as needed
}

# Function to handle user queries
def get_college_info(intent):
    if intent == "programs":
        return "We offer programs in: " + ", ".join(college_info["programs"])
    elif intent == "departments":
        return "Our departments include: " + ", ".join(college_info["departments"])
    elif intent == "faculty":
        department = "Computer Science"  # Assume default department
        if "department" in slots:
            department = slots["department"]
        if department in college_info["faculty"]:
            return f"The faculty in {department} department are: " + ", ".join(college_info["faculty"][department])
        else:
            return "Sorry, I couldn't find information about that department."
    # Add more intents and responses as needed

# Example Usage
while True:
    user_input = input("You: ")
    # Perform intent detection (e.g., using NLP techniques or regular expressions) to determine user's query
    intent = "programs"  # Assume default intent for example
    response = get_college_info(intent)
    print("Bot:", response)


# In[ ]:


#Program-11
import wikipedia
import wolframalpha

# Define your Wolfram Alpha app ID
WOLFRAM_APP_ID = "YOUR_WOLFRAM_APP_ID"

# Function to search Wikipedia for a given query
def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:3]  # Limit to first 3 options
        return f"Multiple options found. Please specify one of the following: {', '.join(options)}"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find any information on that topic."

# Function to query Wolfram Alpha for a given query
def query_wolframalpha(query):
    client = wolframalpha.Client(WOLFRAM_APP_ID)
    res = client.query(query)
    answer = next(res.results, None)
    if answer:
        return answer.text
    else:
        return "Sorry, I couldn't find an answer to your query."

# Function to handle user queries
def handle_query(query):
    # First, search Wikipedia
    wikipedia_result = search_wikipedia(query)
    if "Sorry" not in wikipedia_result:
        return wikipedia_result
    # If not found on Wikipedia, query Wolfram Alpha
    return query_wolframalpha(query)

# Example Usage
while True:
    user_input = input("You: ")
    response = handle_query(user_input)
    print("Bot:", response)


# In[1]:


#Program-12
def countsubstring(s1, s2):
    def helper(substring, string):
        if len(string) < len(substring):
            return 0
        if string[:len(substring)] == substring:
            return 1 + helper(substring, string[1:])
        else:
            return helper(substring, string[1:])
    
    return helper(s1, s2)

# Example usage:
print(countsubstring('ab', 'cabalaba'))  # Output: 2


# In[2]:


#Program-13
def count(test_func, lst):
    count = 0
    for element in lst:
        if test_func(element):
            count += 1
    return count

# Example usage:
result = count(lambda x: x > 2, [1, 2, 3, 4, 5])
print(result)  # Output: 3


# In[ ]:


#Program-14
import random
import time

# Brute force solution for the Knapsack problem
def knapsack_brute_force(items, capacity):
    # Generate all possible combinations of items
    combinations = []
    for i in range(1, len(items) + 1):
        combinations.extend(itertools.combinations(items, i))
    
    # Find the combination with the maximum value within the capacity
    max_value = 0
    best_combination = []
    for combination in combinations:
        total_size = sum(item[1] for item in combination)
        total_value = sum(item[2] for item in combination)
        if total_size <= capacity and total_value > max_value:
            max_value = total_value
            best_combination = combination
    
    return best_combination, max_value

# Function to generate random problem instances for the Knapsack problem
def generate_random_problem_instance(N):
    items = [(f'Item{i}', random.randint(1, 5), random.randint(1, 10)) for i in range(1, N+1)]
    capacity = int(2.5 * N)
    return items, capacity

# Perform performance measurements
def measure_performance(problem_sizes):
    for N in problem_sizes:
        total_time = 0
        for _ in range(10):
            items, capacity = generate_random_problem_instance(N)
            start_time = time.time()
            knapsack_brute_force(items, capacity)
            end_time = time.time()
            total_time += end_time - start_time
        average_time = total_time / 10
        print(f"Average time for problem size {N}: {average_time:.6f} seconds")

# Perform performance measurements for different problem sizes
problem_sizes = [10, 12, 14, 16, 18, 20, 22]
measure_performance(problem_sizes)


# In[4]:


#Program-15
def layout(N, C, L):
    def backtrack(guests, table_assignments):
        if not guests:
            return table_assignments
        
        guest = guests[0]
        for table in range(C):
            valid_assignment = True
            for assigned_guest, assigned_table in table_assignments.items():
                if (assigned_guest, guest) in L or (guest, assigned_guest) in L:
                    if assigned_table == table:
                        valid_assignment = False
                        break
            if valid_assignment:
                table_assignments[guest] = table
                result = backtrack(guests[1:], table_assignments)
                if result:
                    return result
                del table_assignments[guest]
        return None
    
    table_assignments = {}
    result = backtrack(list(range(N)), table_assignments)
    if result:
        return result
    else:
        return False

# Example usage:
N = 5
C = 3
L = [(0, 1), (2, 3)]  # Guests 0 and 1 should not sit together, guests 2 and 3 should not sit together
print(layout(N, C, L))


# In[ ]:




