#!/usr/bin/env python

from Tkinter import *
import math
import util

REGION_BUFFER = 30
INHIBITION_RADIUS_BAR_HEIGHT = 3
INHIBITION_RADIUS_BAR_BUFFER = 10
CELL_PX = 10

class CHTMPrinter(object):


    def __init__(self, brain):
        self.canvas = None
        self.brain = brain
        self.window = None
        self.region_canvases = []

    def setup(self):
        self.window = Tk()
        max_height = 0
        top = REGION_BUFFER
        # Create RegionCanvases
        input_rc = RegionCanvas(REGION_BUFFER, top, self.brain.n_inputs) # Inputs
        self.region_canvases.append(input_rc) 
        x, y = input_rc.bottom_right()
        for r in self.brain.regions:
            side = r._cell_side_len()
            rc = RegionCanvas(x + REGION_BUFFER, top, r.n_cells, region=r)
            self.region_canvases.append(rc)
            x, y = rc.bottom_right()
            if y > max_height:
                max_height = y
        self.canvas = Canvas(self.window, width=x + REGION_BUFFER, height=max_height + REGION_BUFFER)
        self.canvas.pack()

        for rc in self.region_canvases:
            rc.canvas = self.canvas
            rc.render_bg()

    def render(self, inputs=None):
        for rc in self.region_canvases:
            if rc.region:
                activations = [cell.activation for cell in rc.region.cells]
                rc.render_cells(activations)
                rc.render_inhibition_radius()
            elif inputs is not None:
                # Inputs
                rc.render_cells(inputs)

class RegionCanvas(object):

    def __init__(self, x, y, n_cells, canvas=None, region=None):
        self.x = x # upper left
        self.y = y # upper left
        self.n_cells = n_cells
        self.side = math.sqrt(n_cells)
        self.canvas = canvas
        self.region = region

    def render_inhibition_radius(self):
        self.canvas.create_rectangle(self.x, self.y - INHIBITION_RADIUS_BAR_BUFFER, self.x + self.region.inhibition_radius * CELL_PX, self.y - INHIBITION_RADIUS_BAR_BUFFER + INHIBITION_RADIUS_BAR_HEIGHT, fill="#000000")

    def render_bg(self):
        color = "#CCCCCC" if self.region else "#EFEFEF"
        self.canvas.create_rectangle(self.x, self.y, self.x + self.side * CELL_PX, self.y + self.side*CELL_PX, fill=color)

    def render_cells(self, array):
        for index, activation in enumerate(array):
            self.render_cell(index, activation)

    def render_cell(self, index, activation=0.0):
        if self.region:
            r = int(activation * 255)
            g = b = 0
            if activation == 1.0:
                g = 180
        else:
            b = int(activation * 255)
            g = r = 0
        cell_color = '#%02x%02x%02x' % (r, g, b)
        x, y = util.coords_from_index(index, self.side)
        abs_x, abs_y = (x*CELL_PX + self.x, y*CELL_PX + self.y)
        self.canvas.create_rectangle(abs_x, abs_y, abs_x + CELL_PX, abs_y + CELL_PX, fill=cell_color)

    def bottom_right(self):
        return (self.x + self.side*CELL_PX, self.y + self.side*CELL_PX)
