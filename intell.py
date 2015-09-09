#!/usr/bin/env python

"""intell.py: Experiments with Hawkins-esque memory-brain intelligence"""

__author__      = "Jeremy Gordon"
__copyright__   = "Copyright 2015"

import numpy as np
import math
from htm_brain import HTMBrain
from brain import Brain
from Tkinter import *
from constants import *

class Drawable2D(object):

    def setupGraphics(self, canvas):
        pass

    def draw(self, canvas):
        pass

class Base2DWorld(Drawable2D):

    def __init__(self, shape):
        self.shape = shape  # List of size of world, len = # of dimensions

    def dimensions(self):
        return len(self.shape)

    def center(self):
        return np.array(self.shape) / 2


class BaseGame(object):
    '''
    Base game definition with a target winning utility (sensor reading), and a timeout (units of time)
    '''

    def __init__(self, win_reward=100, timeout=100, animate=False):
        # Setup
        self.win_reward = win_reward
        self.timeout = timeout
        self.animate = animate

        # State
        self.time = 0  # At 100, game ends
        self.running = False
        self.agent = None
        self.world = None
        
        # 2D Graphics
        self.window = None
        self.canvas = None


    def run(self):
        self.running = True
        if self.animate:
            self.setupGraphics()
            self.window.after(0, self.step)  # Schedule next draw
            self.window.mainloop()
        else:
            while self.running:
                self.step()

    def step(self):
        print self.time
        self.time += 1
        self.agent.step(self.world)
        if self.time >= self.timeout or self.agent.getReward() >= self.win_reward:
            self.end()
        if self.animate:
            self.draw()
            if self.running:
                self.window.after(50, self.step)  # Schedule next draw

    def end(self):
        self.running = False
        print 'game over'

    # Setters

    def setAgent(self, agent):
        self.agent = agent

    def setWorld(self, world):
        self.world = world

    # Graphics

    def setupGraphics(self):
        world = self.world
        self.window = Tk()
        self.canvas = Canvas(self.window, width=world.shape[0] * CELL_RESOLUTION, height=world.shape[1] * CELL_RESOLUTION)
        self.canvas.pack()
        drawables = [self.agent, self.world]
        for d in drawables:
            d.setupGraphics(self.canvas)

    def draw(self):
        drawables = [self.agent, self.world]
        for d in drawables:
            d.draw(self.canvas)

class Behavior(object):
    '''
    Should behaviors have explicitly defined sensors?
    '''

    def __init__(self, nickname, world):
        self.nickname = nickname
        self.world = world

    def actuate(self, magnitude, state):
        '''
        Override
        '''
        print "%s (%s, state: %s)" % (self.nickname, magnitude, state)

class Sensor(object):
    '''
    Takes reading from physical world and converts into a float between 0 and 1 for present time
    If utility is positive, we want to maximize the reading on this
    If negative, we want to minimize (objective predefined values, e.g. pain, pleasure)
    '''

    def __init__(self, nickname, world, utility=0):
        self.nickname = nickname
        self.world = world
        self.utility = utility

    def observe(self, state):
        '''
        Override depending on type of sensor and definition of world
        '''
        return 0


class Agent(Drawable2D):

    def __init__(self, state, brain, sensors, behaviors, columns_per_region=None):
        # Properties
        self.sensors = sensors  # List of Sensor()
        self.behaviors = behaviors  # List of Behavior()

        # Present State
        self.state = state  # Physical state
        self.reward = 0.0  # Cumulative
        self.active_behaviors = np.zeros(len(self.behaviors))

        # Graphics
        self.drawable = None

        self.brain = brain
        self.brain.initialize(r1_inputs=len(sensors)+len(behaviors))

        print "Created: %s" % self

    def __str__(self):
        return "Agent (%d sensors, %d behaviors)" % (len(self.sensors), len(self.behaviors))

    def step(self, world):
        '''
        Observe, think and behave for current time step
        '''
        readings = self.readSensors()
        self.think(readings)  # Set active_behaviors as output motor commands from cortex (L5)
        self.actuateBehaviors()


    def readSensors(self):
        readings = []
        for s in self.sensors:
            observed = s.observe(self.state)
            if s.utility != 0 and observed != 0:
                self.gotReward(s.utility * observed)
            readings.append(observed)
        # Add readings from behavior (to learn motor commands -> sensed outcomes)
        readings.extend(self.active_behaviors)
        return np.array(readings)


    def think(self, readings):
        # Old brain receives motor commands from each region
        O = self.brain.process(readings)
        # TODO: Processing to squash each regions outputs into a single array of motor commands
        # self.active_behaviors = np.squeeze(np.asarray(O)) # Convert to array


    def actuateBehaviors(self):
        for b, mag in zip(self.behaviors, self.active_behaviors):
            b.actuate(mag, self.state)

    def getReward(self):
        return self.reward

    def gotReward(self, reward):
        self.reward += reward
        print "Rewarded! %s, Current -> %s" % (reward, self.reward)        

