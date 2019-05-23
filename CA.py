import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from time import time

class CellGrid:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.grid = np.zeros((M, N))
        self.grid_next = np.zeros((M, N))
        self.grid_previous = np.zeros((M, N))

        self.neighbors = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        self.orthogonal = [(0,1), (0,-1), (1,0), (-1,0)]

        self.shift = 0

    def randomize(self):
        self.grid[:,:] = np.random.choice((0, 1), (self.M,self.N))

    def heat(self, gamma, q):
        apply_heat = np.random.choice((True, False), (self.M, self.N), p=(gamma, 1.-gamma))
        self.grid[apply_heat] += q

    def gt_number(self, number):
        self.grid_next[:,:] = 0

        self.grid_previous[:,:] = self.grid
        for row_index, row in enumerate(self.grid):
            for col_index, element in enumerate(row):
                sum = 0

                for x,y in self.neighbors:
                    if self.grid[(row_index + x)%self.M, (col_index + y)%self.N] == 1:
                        sum += 1

                if sum + element > number:
                    self.grid_next[row_index, col_index] = 1

        self.grid[:,:] = self.grid_next

    def rotate_each(self):
        self.grid_next[:,:] = 0

        for row_index, row in enumerate(self.grid):
            for col_index, element in enumerate(row):
                if ((row_index - self.shift)%3 == 0) and ((col_index + self.shift)%3 == 0):
                    for neighbor_index, (dx1, dy1) in enumerate(self.neighbors):
                        dx2, dy2 = self.neighbors[(neighbor_index + 1)%len(self.neighbors)]

                        self.grid_next[(row_index + dx1)%self.M, (col_index + dy1)%self.N] = self.grid[(row_index + dx2)%self.M, (col_index + dy2)%self.N]

        self.grid[:,:] = self.grid_next
        self.shift += 1

    def diffusion(self, c):
        self.grid_next[:,:] = 0.0
        self.grid_previous[:,:] = self.grid

        for row_index, row in enumerate(self.grid):
            for col_index, element in enumerate(row):

                sum_neighbors = 0
                for dx, dy in self.orthogonal:
                    sum_neighbors += self.grid[(row_index + dx)%self.M, (col_index + dy)%self.N]

                self.grid_next[row_index, col_index] = self.grid[row_index, col_index] + c*(sum_neighbors - 4.*element)

        self.grid[:,:] = self.grid_next

    def scattering(self, gamma):
        self.grid_next[:,:] = self.grid
        self.grid_previous[:,:] = self.grid

        for row_index, row in enumerate(self.grid):
            for col_index, element in enumerate(row):
                if np.random.uniform(0.0, 1.0) < gamma:
                    dx, dy = random.choice(self.neighbors)
                    self.grid_next[(row_index + dx)%self.M, (col_index + dy)%self.N] = self.grid[row_index, col_index]
                    self.grid_next[row_index, col_index] = self.grid[(row_index + dx)%self.M, (col_index + dy)%self.N]

        self.grid[:,:] = self.grid_next

    def toggling(self, gamma):
        self.grid_next[:,:] = self.grid

        for row_index, row in enumerate(self.grid):
            for col_index, element in enumerate(row):
                if np.random.uniform(0.0, 1.0) < gamma:
                    self.grid_next[row_index, col_index] = (self.grid_next[row_index, col_index] + 1)%2
        self.grid[:,:] = self.grid_next

def main():
    plt.figure(1)
    plt.ion()

    grid = CellGrid(400, 400)
    grid.randomize()

    timesteps = 10000
    for time_index in range(timesteps):

        plt.figure(1)
        plt.clf()
        plt.imshow(grid.grid)
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)

        start = time()
        grid.scattering(0.2)
        grid.gt_number(4)
        #print(time() - start)

        print(np.sum(grid.grid))

if __name__ == '__main__':
    main()
