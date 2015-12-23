#!/usr/bin/env python

import numpy as np
from intell import Agent, Behavior, BaseGame, Sensor, Base2DWorld, Drawable2D
import util
from constants import *

class SimpleHeatGame(BaseGame):

    def __init__(self, animate=False, timeout=50):
        super(SimpleHeatGame, self).__init__(animate=animate, timeout=timeout)

class SimpleHeatSource(Drawable2D):
    '''
    Position must be same length as world dimensionality
    Heat is in range [0,1] and is experienced fully within radius
    Outside of radius is a heat gradient that reduces with distance from center

    Heat currently remains indefinitely when agent is in region
    '''

    def __init__(self, radius, position, heat=1):
        self.radius = radius
        self.position = position
        self.heat = heat

    def setupGraphics(self, canvas):
        core_corners = util.toScreen(util.circleCorners(self.position, self.radius), CELL_RESOLUTION)
        outer_corners = util.toScreen(util.circleCorners(self.position, self.maxDistance()), CELL_RESOLUTION)
        canvas.create_oval(*outer_corners, fill="red")        
        canvas.create_oval(*core_corners, fill="black")

    def maxDistance(self):
        # Outside of which heat not felt
        return self.radius * 3

    def getHeat(self, distance=0):
        within = distance < self.radius
        max_distance = self.maxDistance()
        if within:
            return self.heat
        elif distance > max_distance:
            return 0
        else:
            # Within linear gradient
            return self.heat * ((max_distance - distance)/max_distance)

class SimpleHeatWorld(Base2DWorld):
    '''
    Basic world in 2 or 3 dimensions 
    Heat sources can be added with a single position, and pos or neg temperature
    The heat at any location in the world is defined by distance from each heat source
    '''

    MAX_VELOCITY = 2
    MAX_ROT_VELOCITY = np.pi/6.

    def __init__(self, shape):
        super(SimpleHeatWorld, self).__init__(shape)
        self.hsources = []

    def addSource(self, source):
        self.hsources.append(source)

    def heatAt(self, position):
        heat_from_sources = []
        for hs in self.hsources:
            dist = util.distance(position, hs.position)
            heat = hs.getHeat(dist)
            heat_from_sources.append(heat)
        return sum(heat_from_sources)

    def setupGraphics(self, canvas):
        for hs in self.hsources:
            hs.setupGraphics(canvas)


class HeatAntennaSensor(Sensor):
    '''
    Gets reward at defined offset from agent
    '''

    def __init__(self, nickname, world, offset, utility=0):
        super(HeatAntennaSensor, self).__init__(nickname, world, utility=utility)      
        self.offset = offset

    def observe(self, state):
        observe_loc = state.position + self.offset
        return self.world.heatAt(observe_loc)

class AgentState():

    def __init__(self, world):
        self.position = world.center()
        self.direction = np.array([0, 1]) # Vector of length 1.

    def __str__(self):
        dx, dy = self.direction
        bearing_deg = np.degrees(np.arctan(-1*dx/dy))
        return "At %s, Heading: %s" % (self.position, bearing_deg)

class RotateBehavior(Behavior):

    def __init__(self, nickname, world, cw=True):
        self.cw = cw
        super(RotateBehavior, self).__init__(nickname, world)

    def actuate(self, magnitude, state):
        super(RotateBehavior, self).actuate(magnitude, state)
        direction = 1 if self.cw else -1
        theta = magnitude * direction * (self.world.MAX_ROT_VELOCITY) # Rad, CCW
        z_axis = [0, 0, 1]
        rm = util.rotation_matrix(z_axis, theta)
        dir_3d = np.append(state.direction, [0])
        state.direction = np.dot(rm, dir_3d)[:2] # Back to 2D

class DriveBehavior(Behavior):

    def __init__(self, nickname, world):
        super(DriveBehavior, self).__init__(nickname, world)

    def actuate(self, magnitude, state):
        super(DriveBehavior, self).actuate(magnitude, state)
        move = magnitude * state.direction * self.world.MAX_VELOCITY  # Does this update the state object?
        new_position = state.position + move
        world_ok = util.inside(new_position, [0,0], self.world.shape)
        if world_ok:
            # Ok to move
            state.position = state.position + move



class SimpleHeatAgent(Agent):
    '''
    Agent with 6 directional heat antennas and 3 behaviors
    
    '''
    DRAW_SIZE = 1
    DRAW_ARC = 80

    def __init__(self, world, brain):
        sensors = [
            HeatAntennaSensor('nw', world, [-1,1]),
            HeatAntennaSensor('nnw', world, [-0.5,1]),
            HeatAntennaSensor('n', world, [0,1]),                     
            HeatAntennaSensor('nne', world, [0.5,1]),                     
            HeatAntennaSensor('ne', world, [1,1]),                                                
            HeatAntennaSensor('here', world, [0,0], utility=1)  # Redeemed reward
        ]
        behaviors = [
            RotateBehavior('turn right', world, cw=True),
            RotateBehavior('turn left', world, cw=False),            
            DriveBehavior('drive', world)
        ]
        state = AgentState(world)
        super(SimpleHeatAgent, self).__init__(state, brain, sensors, behaviors)

    # Graphics

    def getCoords(self):
        x, y = self.state.position
        x0 = x - self.DRAW_SIZE
        y0 = y - self.DRAW_SIZE
        x1 = x + self.DRAW_SIZE
        y1 = y + self.DRAW_SIZE
        return util.toScreen([x0, y0, x1, y1], CELL_RESOLUTION)

    def setupGraphics(self, canvas):
        x0, y0, x1, y1 = self.getCoords()
        self.drawable = canvas.create_arc(x0, y0, x1, y1, fill="red", start=320, extent=self.DRAW_ARC)

    def draw(self, canvas):
        dx, dy = self.state.direction
        bearing_deg = np.degrees(np.arctan(-1*dx/dy))
        coords = self.getCoords()
        canvas.coords(self.drawable, *coords)
        canvas.itemconfig(self.drawable, start=bearing_deg - (self.DRAW_ARC / 2))
