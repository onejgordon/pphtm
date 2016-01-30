#!/usr/bin/env python

from Tkinter import *
import math
import util

REGION_BUFFER = 20
HIGHLIGHT_DELAY = 3000
INHIBITION_RADIUS_BAR_HEIGHT = 3
INHIBITION_RADIUS_BAR_BUFFER = 10
CELL_PX = 10

class VIEW():
    ACTIVATION = 1
    OVERLAP = 2
    BIAS = 3
    PRIOR_ACTIVATION = 4

    COLOR = {
        ACTIVATION: (255,0,0),
        OVERLAP: (0,255,0),
        BIAS: (255,255,0),
        PRIOR_ACTIVATION: (255,0,255)
    }

    DRAW = [ACTIVATION, OVERLAP, BIAS, PRIOR_ACTIVATION]

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
        x, _y = input_rc.bottom_right()
        for r in self.brain.regions:
            y = top
            side = r._cell_side_len()
            for view_index, view_type in enumerate(VIEW.DRAW):
                rc = RegionCanvas(x + REGION_BUFFER, y, r.n_cells, region=r, window=self.window, view_index=view_index, view_type=view_type)
                self.region_canvases.append(rc)
                _x, y = rc.bottom_right()
                if y > max_height:
                    max_height = y
                y += REGION_BUFFER
            x = _x + REGION_BUFFER
        self.canvas = Canvas(self.window, width=x + REGION_BUFFER, height=max_height + REGION_BUFFER)
        self.canvas.pack()

        for rc in self.region_canvases:
            rc.canvas = self.canvas
            rc.create_cells()
            # rc.create_inhibition_radius()

    def render(self, inputs=None):
        for rc in self.region_canvases:
            if rc.region:
                rc.render()
                # rc.update_inhibition_radius()
            elif inputs is not None:
                # Inputs
                rc._render_view(inputs, color=(0,0,255))

class RegionCanvas(object):
    '''
    Canvas to draw a view of one region and handle clicks
    '''

    def __init__(self, x, y, n_cells, canvas=None, window=None, region=None, view_index=0, view_type=None):
        self.x = x # upper left
        self.y = y # upper left
        self.n_cells = n_cells
        self.side = math.sqrt(n_cells)
        self.canvas = canvas
        self.window = window
        self.region = region
        self.view_type = view_type
        self.view_index = view_index
        self.inhibition_radius = None
        self.cells = []
        self.prior_activations = []
        self.frame = None
        self.temporary_widgets = []

    def __str__(self):
        return "Region %s View %d" % (self.region, self.view_index)

    def handle_click(self, event, type="left"):
        coordx = int((event.x - self.x) / CELL_PX)
        coordy = int((event.y - self.y) / CELL_PX)
        index = util.index_from_coords(coordx, coordy, self.side)
        cell = self.region.cells[index]
        if cell:
            if type == "left":
                for seg_index, seg in enumerate(cell.distal_segments):
                    connected_synapses = seg.connected_synapses()
                    for i in connected_synapses:
                        text = self.highlight_cell(seg.source(i), text=str(seg_index))
                        self.temporary_widgets.append(text)
                    print "Segment %d active: %s" % (seg_index, seg.active())
                self.window.after(HIGHLIGHT_DELAY, self.clear_temporary_widgets)
                value = self.value_at(index)
                if value is not None:
                    # This is reporting old values for prior_activation
                    print "\nValue at index %d: %s" % (index, value)

    def value_at(self, index):
        if self.view_type == VIEW.ACTIVATION:
            return self.region.cells[index].activation
        elif self.view_type == VIEW.OVERLAP:
            return self.region.overlap[index]
        elif self.view_type == VIEW.BIAS:
            return self.region.bias[index]
        elif self.view_type == VIEW.PRIOR_ACTIVATION:
            return self.prior_activations[index]
        return None

    def handle_left_click(self, event):
        self.handle_click(event, type="left")

    def handle_right_click(self, event):
        self.handle_click(event, type="right")        

    def clear_temporary_widgets(self):
        for w in self.temporary_widgets:
            self.canvas.delete(w)

    def highlight_cell(self, index, text="?"):
        x, y = self.cell_position(index)
        x += CELL_PX / 2
        y += CELL_PX / 2
        t = self.canvas.create_text((x, y), text=text, fill="#FFFFFF", font=("Purisa", 9))
        return t

    def cell_position(self, index):
        ''' Return (x,y)'''
        x, y = util.coords_from_index(index, self.side)
        abs_x, abs_y = (x*CELL_PX + self.x, y*CELL_PX + self.y)
        return (abs_x, abs_y)

    def create_cells(self):
        for index in range(self.n_cells):
            abs_x, abs_y = self.cell_position(index)
            cell = self.canvas.create_rectangle(abs_x, abs_y, abs_x + CELL_PX, abs_y + CELL_PX)
            self.canvas.tag_bind(cell, "<Button-1>", self.handle_left_click)
            self.canvas.tag_bind(cell, "<Button-3>", self.handle_right_click)
            self.cells.append(cell)

    # def create_inhibition_radius(self):
    #     if self.region:
    #         self.inhibition_radius = self.canvas.create_rectangle(self.x, self.y - INHIBITION_RADIUS_BAR_BUFFER, self.x + self.region.inhibition_radius * CELL_PX, self.y - INHIBITION_RADIUS_BAR_BUFFER + INHIBITION_RADIUS_BAR_HEIGHT, fill="#000000")

    # def update_inhibition_radius(self):
    #     if self.region:
    #         self.canvas.itemconfig(self.inhibition_radius, width=self.region.inhibition_radius*CELL_PX)

    def _update_cell(self, index, brightness=0.0, color=(255,0,0), saturate=None):
        r,g,b = [int(brightness*c) for c in color]
        if saturate and brightness == 1.0:
            r,g,b = saturate
        cell_color = '#%02x%02x%02x' % (r, g, b)
        self.canvas.itemconfig(self.cells[index], fill=cell_color)

    def _render_view(self, values, color=(255,0,0), saturate=None):
        for index, value in enumerate(values):
            self._update_cell(index, value, color=color, saturate=saturate)

    def render(self):
        color = VIEW.COLOR[self.view_type]
        if self.view_type == VIEW.ACTIVATION:
            values = [cell.activation for cell in self.region.cells]
        elif self.view_type == VIEW.OVERLAP:
            values = util.normalize(self.region.overlap)
        elif self.view_type == VIEW.BIAS:
            values = util.normalize(self.region.bias)
        elif self.view_type == VIEW.PRIOR_ACTIVATION:
            values = self.prior_activations
        saturate = (255,150,0) if self.view_type == VIEW.ACTIVATION else None
        self._render_view(values, color=color, saturate=saturate)

        # Store prior activations
        self.prior_activations = [cell.activation for cell in self.region.cells]


    def bottom_right(self):
        return (self.x + self.side*CELL_PX, self.y + self.side*CELL_PX)
