#此程序使用gaft库进行计算函数最大值
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Find the global maximum for function: f(x) = -3*(x-30)**2*math.sin(x)
'''
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 15)], eps=0.001)
population = Population(indv_template=indv_template, size=100).init()

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return -3*(x-30)**2*sin(x)

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine and print every generation
    engine.run(ng=500)
    best_indv = engine.population.best_indv(engine.fitness)
    print('Max({0},{1})'.format(best_indv.solution[0],engine.fitness(best_indv)))
    x=np.linspace(0,15,10000)  
    y=[-3*(i-30)**2*math.sin(i) for i in x]  
    plt.plot(x,y)  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('function')
    plt.axis([-1, 16, -3000, 3000]) 
    plt.scatter(best_indv.solution[0],engine.fitness(best_indv),color='r')
    a = round(best_indv.solution[0],4)
    b = round(engine.fitness(best_indv),4)
    plt.annotate ('Max('+str(a)+','+str(b)+')', xy = (best_indv.solution[0],engine.fitness(best_indv)), xytext = (7, 2500),arrowprops = dict(facecolor = 'black', shrink = 0.1, width = 2))
    plt.show() 
