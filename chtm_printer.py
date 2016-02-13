#!/usr/bin/env python

from Tkinter import *
import math
import util
from functools import partial
import numpy as np

REGION_BUFFER = 20
CELL_PX = 12
GRID_OUTLINE = "#555555"
VIEWABLE_SEGMENTS = 3
INDICATOR_DIAMETER = CELL_PX / 3

################
# TODO
# Print segment activation (seems that some segments are not activating when they should)
# Class for region printer to more gracefully handle multiple regions
################

class VIEW():
    INPUTS = 1
    ACTIVATION = 2
    OVERLAP = 3
    BIAS = 4
    PREACTIVATION = 5
    SEGMENT_DETAIL = 6

    COLOR = {
        INPUTS: (0,0,255),
        ACTIVATION: (255,0,0),
        OVERLAP: (0,255,0),
        BIAS: (255,255,0),
        PREACTIVATION: (255,180,0),
        SEGMENT_DETAIL: (0,255,255)
    }

    LABEL = {
        ACTIVATION: "Activation",
        OVERLAP: "Overlap",
        BIAS: "Bias",
        PREACTIVATION: "Preactivation"
    }

    MAX_VALUE = {
        BIAS: 3,
        OVERLAP: 2
    }

    # DRAW FUNCTIONS
    # These methods return a tuple (fill value, indicator color, outline color)

    @staticmethod
    def draw_input(printer, index):
        return (printer.brain.inputs[index], None, None)

    @staticmethod
    def draw_activation(printer, region, index):
        '''
        Activation from this time step (before learning)
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        value = region.last_activation[index]
        activating = region.cells[index].activation == 1.0
        hl = "#fff" if activating else None
        return (value, hl, ol)

    @staticmethod
    def draw_preactivation(printer, region, index):
        '''
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        return (region.pre_activation[index], None, ol)

    @staticmethod
    def draw_overlap(printer, region, index):
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        return (region.overlap[index], None, ol)

    @staticmethod
    def draw_bias(printer, region, index):
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        overlap_match = region.bias[index] > 0 and region.overlap[index] > 0
        indicator = "#fff" if overlap_match else ""
        if printer.view_active("next_bias"):
            cell = region.cells[index]
            for seg in cell.distal_segments:
                active_after_current_activation = seg.active()
                if active_after_current_activation:
                    ol = "#FFFF00"
                    continue
        return (region.bias[index], indicator, ol)

    @staticmethod
    def draw_cell_synapses(printer, cell, index, segment_index=0):
        perm = 0
        indicator = None
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        if segment_index < len(cell.distal_segments):
            seg = cell.distal_segments[segment_index]
            if index in seg.syn_sources:
                # There is a synapse here
                syn_index = seg.syn_sources.index(index)
                perm = seg.syn_permanences[syn_index]
                change = seg.syn_last_change[syn_index]
                contribution = seg.syn_last_contribution[syn_index]
                ol = "#F36B1D" if contribution > 0.5 else ""
                if change == 0:
                    indicator = None
                elif change > 0:
                    indicator = "#fff"
                elif change < 0:
                    indicator = "#000"
        return (perm, indicator, ol)

    # GRID OUTLINE METHODS
    # These methods return a color for the full grid's outline

    @staticmethod
    def segment_learning(cell, segment_index=0):
        if segment_index < len(cell.distal_segments):
            seg = cell.distal_segments[segment_index]
            learning = any(seg.syn_last_change)
            active = seg.active_before_learning
            if learning and active:
                return "#FFFF55"
            elif learning:
                return "#00CCFF"
            elif active:
                return "#FFFF00"
        return ""

    DRAW = [ACTIVATION, PREACTIVATION, OVERLAP, BIAS]

DRAW_FN = {
    VIEW.INPUTS: VIEW.draw_input,
    VIEW.ACTIVATION: VIEW.draw_activation,
    VIEW.PREACTIVATION: VIEW.draw_preactivation,
    VIEW.OVERLAP: VIEW.draw_overlap,
    VIEW.BIAS: VIEW.draw_bias,
}

class CHTMPrinter(object):

    def __init__(self, brain):
        self.brain = brain
        self.window = None
        self.canvas = None
        self.cell_window = None
        self.cell_canvas = None
        self.focus_cell_index = None
        self.rolling_bias_match = []
        self.menubar = None

        # State
        self.view_toggles = ["next_bias"]

        # Canvas items
        self.rolling_match_text = None
        self.main_grids = []
        self.focus_grids = []

    def setup(self):
        self.window = Tk()

        max_height = max_side = max_cells = 0
        top = REGION_BUFFER

        # Main canvas
        self.canvas = Canvas(self.window)

        # Create input grid
        draw_fn = partial(VIEW.draw_input, self)
        input_rc = CellGrid(REGION_BUFFER, top, self.brain.n_inputs, canvas=self.canvas, cell_draw_fn=draw_fn, view_type=VIEW.INPUTS) # Inputs
        self.main_grids.append(input_rc)
        x, _y = input_rc.bottom_right()

        # Create main region grids
        for r in self.brain.regions:
            y = top
            side = r._cell_side_len()
            for view_index, view_type in enumerate(VIEW.DRAW):
                draw_fn = partial(DRAW_FN.get(view_type), self, r)
                on_cell_click = partial(self.focus_cell, r)
                max_value = VIEW.MAX_VALUE.get(view_type)
                grid_title = VIEW.LABEL.get(view_type)
                if max_value is None:
                    max_value = 1.0
                cg = CellGrid(x + REGION_BUFFER, y, r.n_cells, canvas=self.canvas, window=self.window, view_type=view_type, cell_draw_fn=draw_fn, on_cell_click=on_cell_click, max_value=max_value, title=grid_title)
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

        main_width = x+REGION_BUFFER
        main_height = max_height+REGION_BUFFER
        self.canvas.config(width=main_width, height=main_height)
        self.canvas.pack()

        self.rolling_match_text = self.canvas.create_text(main_width / 6, main_height - 100, fill="#000", font=("Purisa", 25), anchor="w")

        # Cell detail window & canvas
        width = max_side + 2*REGION_BUFFER
        height = VIEWABLE_SEGMENTS * (max_side + REGION_BUFFER) + REGION_BUFFER
        self.cell_window = Toplevel()
        self.cell_canvas = Canvas(self.cell_window, width=width, height=height)
        self.cell_canvas.pack()

        for i in range(VIEWABLE_SEGMENTS):
            draw_fn = None
            grid_outline_fn = None
            on_cell_click = partial(self.focus_synapse, r, i)
            cg = CellGrid(REGION_BUFFER, (i * (max_side + REGION_BUFFER)) + REGION_BUFFER, r.n_cells, canvas=self.cell_canvas, window=self.window, view_type=VIEW.SEGMENT_DETAIL, cell_draw_fn=draw_fn, grid_outline_fn=grid_outline_fn, saturate_value=0.2, max_value=0.5, on_cell_click=on_cell_click)
            self.focus_grids.append(cg)

        # Initialize all cell grids
        for cg in self.main_grids:
            cg.initialize()
        for cg in self.focus_grids:
            cg.initialize()

        # Create menu
        self.menubar = Menu(self.window)
        action_menu = Menu(self.menubar, tearoff=0)
        action_menu.add_command(label="Toggle Next Bias Showing", command=partial(self.toggle_view, 'next_bias'))
        action_menu.add_command(label="Segment Activation Showing", command=partial(self.toggle_view, 'segment_activation'))
        self.menubar.add_cascade(label="Action", menu=action_menu)
        self.window.config(menu=self.menubar)

        # Move main window to right of cell
        self.window.geometry('%dx%d+%d+%d' % (main_width, main_height, width, 0))

    def toggle_view(self, toggle):
        if toggle in self.view_toggles:
            self.view_toggles.remove(toggle)
        else:
            self.view_toggles.append(toggle)

    def view_active(self, view):
        return view in self.view_toggles

    def render(self):
        for cg in self.main_grids + self.focus_grids:
            cg.render(self.focus_cell_index)

        r = self.brain.regions[0]
        bias_count = float(sum(r.bias > 0))
        if bias_count > 0:
            bias_match = sum(np.logical_and(r.bias > 0, r.overlap > 0))
            ratio = bias_match / bias_count
        else:
            ratio = 0
        if ratio:
            util.rolling_list(self.rolling_bias_match, length=5, new_value=ratio)
        rolling_average = util.average(self.rolling_bias_match)
        self.canvas.itemconfig(self.rolling_match_text, text="%.2f" % rolling_average)

    def focus_cell(self, region, index):
        if index < len(region.cells):
            self.focus_cell_index = index
            cell = region.cells[index]
            self.cell_window.wm_title("Cell %d" % index)
            print "Focusing on %s" % (cell)
            for si, cg in enumerate(self.focus_grids):
                cg.cell_draw_fn = partial(VIEW.draw_cell_synapses, self, cell, segment_index=si)
                cg.grid_outline_fn = partial(VIEW.segment_learning, cell, segment_index=si)
                cg.render()

    def focus_synapse(self, region, segment_index, index):
        if self.focus_cell_index:
            cell = region.cells[self.focus_cell_index]
            seg = cell.distal_segments[segment_index]
            perm, contr, last_change = seg.synapse_state(index)
            print "Synapse focus_cell=%s source_index=%s permanence=%s contribution=%s last_change=%s" % (self.focus_cell_index, index, perm, contr, last_change)


class CellGrid(object):
    '''
    Canvas to draw a view of one region and handle clicks
    '''

    def __init__(self, x, y, n_cells, canvas=None, window=None, view_type=None, cell_draw_fn=None, grid_outline_fn=None, on_cell_click=None, max_value=1.0, saturate_value=1.0, title=None):
        self.x, self.y = x, y # upper left
        self.n_cells = n_cells
        self.max_value = max_value
        self.saturate_value = saturate_value
        self.saturate_color = None
        self.side = math.sqrt(n_cells)
        self.canvas = canvas
        self.window = window
        self.view_type = view_type
        self.cell_draw_fn = cell_draw_fn
        self.grid_outline_fn = grid_outline_fn
        self.on_cell_click = on_cell_click # fn
        self.cell_rects = []
        self.hl_indicators = []
        self.prior_activations = []
        self.temporary_widgets = []
        self.title = title
        self.hl = None

    def __str__(self):
        return "View (type %d)" % (self.view_type)

    def initialize(self):
        # Create frame background
        self.hl = self.canvas.create_rectangle(self.x, self.y, self.x + CELL_PX * self.side, self.y + CELL_PX * self.side, outline="#000")

        # Create cell rects & highlight items for rendering
        for index in range(self.n_cells):
            abs_x, abs_y = self._cell_position(index)
            cell_rect = self.canvas.create_rectangle(abs_x, abs_y, abs_x + CELL_PX, abs_y + CELL_PX, outline=GRID_OUTLINE)
            self.canvas.tag_bind(cell_rect, "<Button-1>", self.handle_click)
            self.cell_rects.append(cell_rect)
            # Create indicator
            abs_x += CELL_PX / 4
            abs_y += CELL_PX / 4
            x_br = abs_x + INDICATOR_DIAMETER
            y_br = abs_y + INDICATOR_DIAMETER
            hl_indicator = self.canvas.create_oval(abs_x, abs_y, x_br, y_br, fill="", outline="")
            self.hl_indicators.append(hl_indicator)
        if self.title:
            self.canvas.create_text((self.x, self.y - 9), text=self.title, fill="#000", font=("Purisa", 11), anchor="w")

    def _cell_position(self, index):
        ''' Return (x,y)'''
        x, y = util.coords_from_index(index, self.side)
        abs_x, abs_y = (x*CELL_PX + self.x, y*CELL_PX + self.y)
        return (abs_x, abs_y)

    def _update_cell(self, index, brightness=0.0, color=(255,0,0), saturate=(255,255,255), indicator_color=None, outline_color=None):
        r,g,b = [int(brightness*c) for c in color]
        if brightness >= self.saturate_value:
            r,g,b = saturate
        cell_color = '#%02x%02x%02x' % (r, g, b)
        if not outline_color:
            outline_color = GRID_OUTLINE
        self.canvas.itemconfig(self.cell_rects[index], fill=cell_color, outline=outline_color)
        hl_indicator = self.hl_indicators[index]
        indicator_outline = ""
        if indicator_color:
            indicator_outline = "#fff" if indicator_color == "#000" else "#000"
            state = "normal"
        else:
            state = "hidden"
        self.canvas.itemconfig(self.hl_indicators[index], fill=indicator_color, outline=indicator_outline, state=state)

    def handle_click(self, event):
        coordx = int((event.x - self.x) / CELL_PX)
        coordy = int((event.y - self.y) / CELL_PX)
        index = util.index_from_coords(coordx, coordy, self.side)
        if self.on_cell_click:
            self.on_cell_click(index)

    def draw_params_at(self, index):
        if self.cell_draw_fn:
            value, hl_color, ol_color = self.cell_draw_fn(index)
            if self.max_value != 1.0:
                value = value / self.max_value
            return value, hl_color, ol_color
        return (None, None, None)

    def clear_temporary_widgets(self):
        for w in self.temporary_widgets:
            self.canvas.delete(w)

    def outline_cell(self, index, color="#FFFFFF"):
        x, y = self._cell_position(index)
        t = self.canvas.create_rectangle(x, y, x + CELL_PX, y + CELL_PX, outline=color, fill="")
        self.temporary_widgets.append(t)

    def add_cell_label(self, index, text="?"):
        x, y = self._cell_position(index)
        x += CELL_PX / 2
        y += CELL_PX / 2
        t = self.canvas.create_text((x, y), text=text, fill="#fff", font=("Purisa", 7))
        self.temporary_widgets.append(t)
        return t

    def highlight_grid(self, color="#fff"):
        self.canvas.itemconfig(self.hl, outline=color, width=10.0)

    def render(self, focus_index=None):
        grid_outline = ""
        if self.grid_outline_fn:
            grid_outline = self.grid_outline_fn()
        self.highlight_grid(grid_outline)
        if self.cell_draw_fn:
            self.clear_temporary_widgets()
            color = VIEW.COLOR[self.view_type]
            saturate = self.saturate_color if self.saturate_color else color
            for index in range(self.n_cells):
                value, indicator_color, outline_color = self.draw_params_at(index)
                self._update_cell(index, value, color=color, saturate=saturate, indicator_color=indicator_color, outline_color=outline_color)

    def bottom_right(self):
        return (self.x + self.side*CELL_PX, self.y + self.side*CELL_PX)
