import sys
import csv
from random import random
from queue import Queue
from collections import deque
import heapq

class MovieEnvironment:
    def __init__(self):
        filepath =  r"disney-movies-data.csv"
        self.titles = []
        self.length = 0
        self.__tdict = {}
        self.__adj_list = {}

        self.__read_movie_data(filepath)
        self.__generate_graph()


    def __read_movie_data(self, filepath):
        file = open(filepath, "r")
        data = list(csv.reader(file, delimiter=","))
        self.titles = [row[0] for row in data]
        file.close()
        self.length = len(self.titles)

    def __generate_graph(self):
        i = 0
        while i < 500: # number of edges in the graph.
            r1 = int(random()*self.length)
            r2 = int(random()*self.length)
            while r2 == r1:
                r2 = int(random()*self.length)
            
            while (r1,r2) in self.__tdict.keys() or (r2,r1) in self.__tdict.keys():
                r2 = int(random()*self.length)

            self.__tdict[(r1,r2)] = 1
            self.__tdict[(r2,r1)] = 1

            weight = random()
            self.__adj_list.setdefault(self.titles[r1],{})[self.titles[r2]]=round(weight,2)*100
            self.__adj_list.setdefault(self.titles[r2],{})[self.titles[r1]]=round(weight,2)*100
            i+=1

    def get_neighbours(self, m1):
        """
        Returns the neighbours (similar movies) for a movie.

        :param str m1: The movie name whose neighbours to find.
        :return dict[str,float]: The dictionary of neighbour nodes and their link weights (0-100) as float which show similarity (lower value means more similar).
        """
        return self.__adj_list[m1]

    def display_graph(self):
        import networkx as nx
        g = nx.DiGraph(self.__adj_list)
        nx.draw(g, with_labels=True, font_weight='bold')
        import matplotlib.pyplot as plt
        plt.show()


""" Your code starts here   """

class MovieNode:

    def __init__(self,name):
     self.name = name
     self.parent = self
     self.weight = 0

    def __lt__(self, other):
        return self.getWeight() < other.getWeight() 
    
    def __eq__(self, other):
        return self.getName() == other.getName()
    
    def setParent(self, parent):
        self.parent = parent

    def setWeight(self, weight):
        self.weight = weight
    
    def getParent(self):
        return (self.parent)
    
    def getName(self):
        return (self.name)

    def getWeight(self):
        return (self.weight)
    
    def containedIn(self, list):
        for element in list:
            if (self == element):
                return element


def breadth_first_search(env, movie1, movie2):
    """
    Returns the shortest path from movie1 to movie2 (ignore the weights).
    """

    print("BFS:")

    queue = Queue(0)
    movieNode1 = MovieNode(movie1)
    queue.put(movieNode1)
    moves = 0
    deadEnds = 0

    explored = []

    # while queue is not empty 
    while (not queue.empty()):

        moves += 1

        # maximum attempts set at 1000
        if (moves == 1000):
            break

        # dequeuing
        movieNode = queue.get()

        # checking if the dequeued movie is the goal (movie 2)
        if (movieNode.getName() == movie2):
            print(f"success!\ntotal number of moves to reach the goal: {moves}\ntotal number of nodes in the search tree: {len(explored)}\nnumber of dead-ends found: {deadEnds}\nThe trace of path:")
            return (getPath(movieNode1, movieNode))

        neighbour = env.get_neighbours(movieNode.getName())

        # backtrack if dead-end reached
        if (not neighbour):
            deadEnds += 1
            continue

        # counter to find dead-ends
        i = 0
        
        # if not, enqueuing its neighbours
        for name, weight in neighbour.items():

            newMovieNode = MovieNode(name)
            newMovieNode.setParent(movieNode)

            if (not newMovieNode.containedIn(explored)):
                explored.append(newMovieNode)
                queue.put(newMovieNode)

            else:
                i += 1

        if (i == len(neighbour)):
            deadEnds += 1

    print(f"failure!\nno. of failed attempts: {moves}")


def depth_first_search(env, movie1, movie2):
    """
    Returns the path from movie1 to movie2
    """

    print("DFS:")

    stack = deque()
    movieNode1 = MovieNode(movie1)

    # pushing in stack
    stack.append(movieNode1)

    moves = 0
    deadEnds = 0

    explored = []
    explored.append(movieNode1.getName())

    # while stack is not empty 
    while (stack):

        moves += 1

        # maximum attempts set at 5000
        if (moves == 5000):
            break

        # popping stack
        movieNode = stack.pop()

        # checking if the popped movie is the goal (movie 2)
        if (movieNode.getName() == movie2):
            print(f"success!\ntotal number of moves to reach the goal: {moves}\ntotal number of nodes in the search tree: {len(explored)}\nnumber of dead-ends found: {deadEnds}\nThe trace of path:")
            return (getPath(movieNode1, movieNode))

        neighbour = env.get_neighbours(movieNode.getName())

        # backtrack if dead-end reached
        if (not neighbour):
            deadEnds += 1
            continue

        i = 0
    
        # if not, pushing its unvisited neighbours
        for name, weight in neighbour.items():

            if (name not in explored):
                
                explored.append(name)
                newMovieNode = MovieNode(name)
                newMovieNode.setParent(movieNode)
                
                # pushing in stack
                stack.append(newMovieNode)

            else:
                i += 1

        if (i == len(neighbour)):
            deadEnds += 1

    print(f"failure!\nno. of failed attempts: {moves}")


def uniform_cost_search(env, movie1, movie2):
    """
    Returns the path from movie1 to movie2 with the highest sum of weights.
    """

    print("UCS:")

    queue = []
    movieNode1 = MovieNode(movie1)
    heapq.heappush(queue, (0, movieNode1))
    i = 0
    deadEnds = 0

    explored = []
    explored.append(movieNode1)

    while (queue):

        i += 1
        
        # maximum attempts set at 1000
        if (i == 1000):
            break

        # dequeuing
        weight, movieNode = (heapq.heappop(queue))

        # checking if the dequeued movie is the goal (movie 2)
        if (movieNode.getName() == movie2):
            print(f"success!\ntotal number of moves to reach the goal: {i}\ntotal number of nodes in the search tree: {len(explored)}\nnumber of dead-ends found: {deadEnds}\ntotal cost of path: {movieNode.getWeight()}\nThe trace of path:")
            return getPath(movieNode1, movieNode)

        neighbour = env.get_neighbours(movieNode.getName())

        # backtrack if dead-end reached
        if (not neighbour):
            deadEnds += 1
            continue

        j = 0
        
        # if not, enqueuing its neighbours
        for name, weight in neighbour.items():

            newMovieNode = MovieNode(name)
            newMovieNode.setParent(movieNode)
            newWeight = weight + movieNode.getWeight()
            newMovieNode.setWeight(newWeight)

            # if movie is not already explored
            if (not newMovieNode.containedIn(explored)):
                explored.append(newMovieNode)
                heapq.heappush(queue, (newWeight, newMovieNode))

            else:
                # if child.STATE is in frontier with higher PATH-COST then replace that frontier node with child
                prevMovieNode = (MovieNode) (newMovieNode.containedIn(explored))
                if (prevMovieNode.getWeight() > newWeight):
                    queue.remove(prevMovieNode)
                    heapq.heappush(queue, (newWeight, newMovieNode))
                j += 1
            
        if (j == len(neighbour)):
            deadEnds += 1

    print(f"failure!\nno. of failed attempts: {i}")

def getPath(movieNode1, movieNode2):
    
    path = []
    tempNode = movieNode2

    while (True):
        path.insert(0, tempNode.getName())
        tempNode = tempNode.getParent()
        if (tempNode.getName() == movieNode1.getName()):
            path.insert(0, movieNode1.getName())
            break

    return path

""" Your code ends here     """


if __name__ == "__main__":
    env = MovieEnvironment()

    movie1 = input("enter movie1 name:")
    i=1
    while movie1 not in env.titles:
        print("name not in the list")
        movie1 = input("enter movie1 name:")
        i+=1
        if i>=3:
            sys.exit()

    movie2 = input("enter movie2 name:")
    i=1
    while movie2 not in env.titles:
        print("name not in the list")
        movie2 = input("enter movie1 name:")
        i+=1
        if i>=3:
            sys.exit()

    # movie1 = "Coco"
    # movie2 = "Tangled"
            
    print(breadth_first_search(env, movie1, movie2))
    print(depth_first_search(env, movie1, movie2))
    print(uniform_cost_search(env, movie1, movie2))

    env.display_graph()
