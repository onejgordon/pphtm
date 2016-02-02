#!/usr/bin/env python

from Tkinter import *
import math
import util
from functools import partial

REGION_BUFFER = 20
HIGHLIGHT_DELAY = 3000
INHIBITION_RADIUS_BAR_HEIGHT = 3
INHIBITION_RADIUS_BAR_BUFFER = 10
CELL_PX = 15
OUTLINE_COLOR = "#555555"
VIEWABLE_SEGMENTS = 5

class VIEW():
    INPUTS = 1
    ACTIVATION = 2
    OVERLAP = 3
    BIAS = 4
    SEGMENT_DETAIL = 5

    COLOR = {
        INPUTS: (0,0,255),
        ACTIVATION: (255,0,0),
        OVERLAP: (0,255,0),
        BIAS: (255,255,0),
        SEGMENT_DETAIL: (0,255,255)
    }

    @staticmethod
    def input_value(brain, index):
        return brain.inputs[index]

    @staticmethod
    def activation_value(region, index):
        '''
        Activation from this time step (before learning)
        '''
        return region.last_activation[index]

    @staticmethod
    def overlap_value(region, index):
        return region.overlap[index]

    @staticmethod
    def bias_value(region, index):
        return region.bias[index]

    @staticmethod
    def cell_connections(cell, index, segment_index=0):
        if segment_index < len(cell.distal_segments):
            seg = cell.distal_segments[segment_index]
            if index in seg.syn_sources:
                syn_index = seg.syn_sources.index(index)
                perm = seg.syn_permanences[syn_index] / 0.2
                return perm
        return 0

    # Label functions

    @staticmethod
    def activating_now(region, index):
        '''
        Cells activated by current time step
        '''
        activating = region.cells[index].activation == 1.0
        return "A" if activating else ""

    @staticmethod
    def synapse_change(cell, index, segment_index=0):
        if segment_index < len(cell.distal_segments):
            seg = cell.distal_segments[segment_index]
            if index in seg.syn_sources:
                syn_index = seg.syn_sources.index(index)
                change = seg.syn_last_change[syn_index]
                if change == 0:
                    label = ""
                elif change > 0:
                    label = "+"
                elif change < 0:
                    label = "-"
                return label
        return None

    @staticmethod
    def label(region, index):
        return ""

    DRAW = [ACTIVATION, OVERLAP, BIAS]

VALUE_FN = {
    VIEW.INPUTS: VIEW.input_value,
    VIEW.ACTIVATION: VIEW.activation_value,
    VIEW.OVERLAP: VIEW.overlap_value,
    VIEW.BIAS: VIEW.bias_value,
}

LABEL_FN = {
    VIEW.ACTIVATION: VIEW.activating_now,
    VIEW.OVERLAP: VIEW.label,
    VIEW.BIAS: VIEW.label,
}

class CHTMPrinter(object):


    def __init__(self, brain):
        self.brain = brain
        self.window = None
        self.canvas = None
        self.cell_window = None
        self.cell_canvas = None
        self.main_grids = []
        self.focus_grids = []
        self.focus_cell_index = None

    def setup(self):
        self.window = Tk()
        max_height = max_side = max_cells = 0
        top = REGION_BUFFER

        # Main canvas
        self.canvas = Canvas(self.window)

        # Create input grid
        value_fn = partial(VIEW.input_value, self.brain)
        input_rc = CellGrid(REGION_BUFFER, top, self.brain.n_inputs, canvas=self.canvas, cell_value_fn=value_fn, view_type=VIEW.INPUTS) # Inputs
        self.main_grids.append(input_rc)
        x, _y = input_rc.bottom_right()

        # Create main region grids
        for r in self.brain.regions:
            y = top
            side = r._cell_side_len()
            for view_index, view_type in enumerate(VIEW.DRAW):
                value_fn = partial(VALUE_FN.get(view_type), r)
                label_fn = partial(LABEL_FN.get(view_type), r)
                on_cell_click = partial(self.focus_cell, r)
                cg = CellGrid(x + REGION_BUFFER, y, r.n_cells, canvas=self.canvas, window=self.window, view_type=view_type, cell_value_fn=value_fn, cell_label_fn=label_fn, on_cell_click=on_cell_click)
                if cg.side > max_side:
                    max_side = cg.side * CELL_PX
                if cg.n_cells > max_cells:
                    max_cells = cg.n_cells
                self.main_grids.append(cg)
                _x, y = cg.bottom_right()
                if y > max_height:
                    max_height = y
                y += REGION_BUFFER
            x = _x + REGION_BUFFER

        self.canvas.config(width=x+REGION_BUFFER, height=max_height+REGION_BUFFER)
        self.canvas.pack()

        # Cell window & canvas
        self.cell_window = Toplevel()
        self.cell_canvas = Canvas(self.cell_window, width=max_side + 2*REGION_BUFFER, height=VIEWABLE_SEGMENTS * (max_side + REGION_BUFFER) + REGION_BUFFER)
        self.cell_canvas.pack()

        for i in range(VIEWABLE_SEGMENTS):
            value_fn = None
            label_fn = None
            cg = CellGrid(REGION_BUFFER, (i * (max_side + REGION_BUFFER)) + REGION_BUFFER, r.n_cells, canvas=self.cell_canvas, window=self.window, view_type=VIEW.SEGMENT_DETAIL, cell_value_fn=value_fn, cell_label_fn=label_fn)
            self.focus_grids.append(cg)

        # Initialize all cell grids
        for cg in self.main_grids:
            cg.initialize()
        for cg in self.focus_grids:
            cg.initialize()

    def render(self):
        for cg in (self.main_grids + self.focus_grids):
            cg.render(self.focus_cell_index)

    def focus_cell(self, region, index):
        self.focus_cell_index = index
        cell = region.cells[index]
        self.cell_window.wm_title("Cell %d" % index)
        print "\nFocusing on %s" % (cell)
        for si, cg in enumerate(self.focus_grids):
            cg.cell_value_fn = partial(VIEW.cell_connections, cell, segment_index=si)
            cg.cell_label_fn = partial(VIEW.synapse_change, cell, segment_index=si)
            cg.render()

class CellGrid(object):
    '''
    Canvas to draw a view of one region and handle clicks
    '''

    def __init__(self, x, y, n_cells, canvas=None, window=None, view_type=None, cell_value_fn=None, cell_label_fn=None, on_cell_click=None):
        self.x, self.y = x, y # upper left
        self.n_cells = n_cells
        self.side = math.sqrt(n_cells)
        self.canvas = canvas
        self.window = window
        self.view_type = view_type
        self.cell_value_fn = cell_value_fn
        self.cell_label_fn = cell_label_fn
        self.on_cell_click = on_cell_click # fn
        self.cell_rects = []
        self.prior_activations = []
        self.temporary_widgets = []

    def __str__(self):
        return "View (type %d)" % (self.view_type)

    def initialize(self):
        for index in range(self.n_cells):
            abs_x, abs_y = self._cell_position(index)
            cell_rect = self.canvas.create_rectangle(abs_x, abs_y, abs_x + CELL_PX, abs_y + CELL_PX, outline=OUTLINE_COLOR)
            self.canvas.tag_bind(cell_rect, "<Button-1>", self.handle_click)
            self.cell_rects.append(cell_rect)

    def _cell_position(self, index):
        ''' Return (x,y)'''
        x, y = util.coords_from_index(index, self.side)
        abs_x, abs_y = (x*CELL_PX + self.x, y*CELL_PX + self.y)
        return (abs_x, abs_y)

    def _update_cell(self, index, brightness=0.0, color=(255,0,0), saturate=(255,255,255)):
        r,g,b = [int(brightness*c) for c in color]
        if brightness >= 1.0:
            r,g,b = saturate
        cell_color = '#%02x%02x%02x' % (r, g, b)
        self.canvas.itemconfig(self.cell_rects[index], fill=cell_color)

    def handle_click(self, event):
        coordx = int((event.x - self.x) / CELL_PX)
        coordy = int((event.y - self.y) / CELL_PX)
        index = util.index_from_coords(coordx, coordy, self.side)
        if self.on_cell_click:
            self.on_cell_click(index)

    def value_at(self, index):
        if self.cell_value_fn:
            return self.cell_value_fn(index)

    def label_at(self, index):
        if self.cell_label_fn:
            return self.cell_label_fn(index)
        return None

    def clear_temporary_widgets(self):
        for w in self.temporary_widgets:
            self.canvas.delete(w)

    def add_cell_label(self, index, text="?"):
        x, y = self._cell_position(index)
        x += CELL_PX / 2
        y += CELL_PX / 2
        t = self.canvas.create_text((x, y), text=text, fill="#FFFFFF", font=("Purisa", 7))
        self.temporary_widgets.append(t)
        return t

    def render(self, focus_index=None):
        if self.cell_value_fn:
            self.clear_temporary_widgets()
            color = VIEW.COLOR[self.view_type]
            saturate = (255,150,0) if self.view_type == VIEW.ACTIVATION else color
            for index in range(self.n_cells):
                value = self.value_at(index)
                label = self.label_at(index)
                self._update_cell(index, value, color=color, saturate=saturate)
                if label:
                    self.add_cell_label(index, label)
            # Add focused cell
            if focus_index is not None:
                self.add_cell_label(focus_index, "*")

    def bottom_right(self):
        return (self.x + self.side*CELL_PX, self.y + self.side*CELL_PX)
