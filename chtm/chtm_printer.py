#!/usr/bin/env python

from __future__ import print_function

from Tkinter import *
import math
import util
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

REGION_BUFFER = 20
CELL_PX = 8
GRID_OUTLINE = "#555555"
VIEWABLE_SEGMENTS = 3
INDICATOR_DIAMETER = CELL_PX / 2
LABEL_GAP = 30
LABEL_SIZE = 10
VALUE_SIZE = 25
ROLLING_PREDICTION_COUNT = 20

################
# Chart Key
# ---------

# MAIN GRIDS
# ----------
# Blue - Inputs
# Red - Activation
#   * Colors are last step activation
#   * White indicator: activating resultant from this step
# Orange - Pre-activation (integration of bias and overlap, before winners picked)
# Green - Overlap (activation from inputs via proximal segment)
# Yellow - Bias & Post-Bias (distal/topdown influence)
#   * White indicator (bias only): overlap matches bias (predicted input?)

# SYNAPSE GRIDS
# -------------
# These show a detail for a single selected cell
# Each grid shows connections from a particular distal or proximal segment
# - Brightness of cell is permanence
# - Orange outlines indicate synapses that are contributing
# - Cell indicator indicates permanence increase (white) or decrease (black)
# - Grid outlines:
#   * Yellow: Active
#   * Blue: Learning
#   * Green: Learning AND Active

# PREDICTION GRIDS
# ----------------


class VIEW():
    INPUTS = 1
    ACTIVATION = 2
    OVERLAP = 3
    BIAS = 4
    PREACTIVATION = 5
    SEGMENT_DETAIL = 6
    POSTBIAS = 7
    PREDICTOR = 8

    COLOR = {
        INPUTS: (0,0,255),
        ACTIVATION: (255,0,0),
        OVERLAP: (0,255,0),
        BIAS: (255,255,0),
        PREACTIVATION: (255,180,0),
        SEGMENT_DETAIL: (0,255,255),
        POSTBIAS: (255,255,0),
        PREDICTOR: (0,255,0)
    }

    LABEL = {
        ACTIVATION: "Activation",
        OVERLAP: "Overlap",
        BIAS: "Bias",
        PREACTIVATION: "Preactivation",
        POSTBIAS: "Post-Bias",
        PREDICTOR: "Predictor"
    }

    MAX_VALUE = {
        BIAS: 3,
        POSTBIAS: 3,
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

        Returns:
            tuple (3): value, indicator_color, outline_color
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        value = region.last_activation[index]
        activating = region.cells[index].activation == 1.0
        indicator = "#fff" if activating else None
        return (value, indicator, ol)

    @staticmethod
    def draw_preactivation(printer, region, index):
        '''
        Returns:
            tuple (3): value, indicator_color, outline_color
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        return (region.pre_activation[index], None, ol)

    @staticmethod
    def draw_overlap(printer, region, index):
        '''
        Returns:
            tuple (3): value, indicator_color, outline_color
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        return (region.overlap[index], None, ol)

    @staticmethod
    def draw_bias(printer, region, index):
        '''
        Returns:
            tuple (3): value, indicator_color, outline_color
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        overlap_match = region.last_bias[index] > 0 and region.overlap[index] > 0
        indicator = "#fff" if overlap_match else ""
        return (region.last_bias[index], indicator, ol)

    @staticmethod
    def draw_post_bias(printer, region, index):
        '''
        Returns:
            tuple (3): value, indicator_color, outline_color
        '''
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        return (region.bias[index], None, ol)

    @staticmethod
    def draw_cell_synapses(printer, cell, index, segment_index=0, post_step=False, segment_type='distal'):
        '''
        Draw a single grid showing synapses for one distal segment

        Strength of blue color indicates permanence of synapse
        Orange outlines indicate synapses that are contributing
        Indicator indicates permanence increase (white) or decrease (black)
        '''
        perm = 0
        indicator = None
        ol = None
        if index == printer.focus_cell_index:
            ol = "#fff"
        segs = []
        if segment_type == 'distal':
            segs = cell.distal_segments
        elif segment_type == 'proximal':
            segs = cell.proximal_segments
        if segment_index < len(segs):
            seg = segs[segment_index]
            if index in seg.syn_sources:
                # There is a synapse here
                syn_index = seg.syn_sources.index(index)
                perm = seg.syn_permanences[syn_index]
                change = 0
                if post_step:
                    change = seg.syn_change[syn_index]
                    contribution = seg.syn_contribution[syn_index]
                else:
                    contribution = seg.syn_prestep_contribution[syn_index]
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
    def segment_learning(cell, segment_index=0, post_step=False):
        if segment_index < len(cell.distal_segments):
            seg = cell.distal_segments[segment_index]
            learning = any(seg.syn_change)
            active = seg.active_before_learning
            if learning and active:
                return "#5aff68"
            elif learning:
                return "#00CCFF"
            elif active:
                return "#FFFF00"
        return ""

    DRAW = [BIAS, OVERLAP, PREACTIVATION, ACTIVATION, POSTBIAS]

DRAW_FN = {
    VIEW.INPUTS: VIEW.draw_input,
    VIEW.ACTIVATION: VIEW.draw_activation,
    VIEW.PREACTIVATION: VIEW.draw_preactivation,
    VIEW.OVERLAP: VIEW.draw_overlap,
    VIEW.BIAS: VIEW.draw_bias,
    VIEW.POSTBIAS: VIEW.draw_post_bias
}

class CHTMPrinter(Tk):

    def __init__(self, brain, predictor=None, handle_run_batch=None, handle_quit=None, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.brain = brain
        self.predictor = predictor
        self.max_side = math.sqrt(brain.CELLS_PER_REGION) * CELL_PX
        self.handle_run_batch = handle_run_batch
        self.handle_quit = handle_quit
        self.focus_cell_index = None
        self.rolling_prediction_correct = []
        self.menubar = None

        # Windows
        self.main_window = None
        self.cell_window = None
        self.predictor_window = None

        # State
        self.view_toggles = []
        self.last_raw_input = None
        self.raw_input = None
        self.prediction = None
        self.prediction_correct = False

        # History
        self.prediction_correct_history = []
        self.rolling_match_history = []

    def setup(self):
        self.main_window = MainWindow(self, self.brain, on_focus_cell=self.focus_cell, max_side=self.max_side)
        self.cell_window = CellWindow(self, max_side=self.max_side, cpr=self.brain.CELLS_PER_REGION)

        self.main_window.move(x=self.cell_window.width, y=0)
        self.geometry('%dx%d+%d+%d' % (200, 150, 0, 400))

        # Predictor window & canvas
        if self.predictor:
            self.predictor_window = PredictorWindow(self, predictor=self.predictor, max_side=self.max_side, cpr=self.brain.CELLS_PER_REGION)
            self.predictor_window.move(self.main_window.width + self.cell_window.width)

        # Create menu
        self.menubar = Menu(self)
        action_menu = Menu(self.menubar, tearoff=0)
        file_menu = Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.handle_quit)
        action_menu.add_command(label="Segment Activation Showing", command=partial(self.toggle_view, 'segment_activation'))
        action_menu.add_command(label="Segment Post Step", command=partial(self.toggle_view, 'segment_post_step'))
        action_menu.add_command(label="Show Predicted Associations", command=partial(self.toggle_view, 'predicted_associations'))
        action_menu.add_command(label="Show Predicted Associations", command=partial(self.toggle_view, 'predicted_associations'))
        self.menubar.add_cascade(label="File", menu=file_menu)
        self.menubar.add_cascade(label="Action", menu=action_menu)
        self.config(menu=self.menubar)

        # Controls
        Label(self, text="Batch Size").grid(row=0)
        self.batch_entry = Entry(self)
        self.batch_entry.grid(row=1)
        b = Button(self, text="Run", command=self.parse_batch_size)
        b.grid(row=2)
        b = Button(self, text="Quit", command=self.handle_quit)
        b.grid(row=3)

    def parse_batch_size(self):
        _steps = self.batch_entry.get()
        if not _steps:
            steps = 1
        if _steps.isdigit():
            steps = int(_steps)
        self.handle_run_batch(steps)


    def startloop(self, delay, fn):
        self.after(delay, fn)
        print("Startig main loop...")
        self.mainloop()


    def toggle_view(self, toggle):
        if toggle in self.view_toggles:
            self.view_toggles.remove(toggle)
        else:
            self.view_toggles.append(toggle)


    def view_active(self, view):
        return view in self.view_toggles

    def set_raw_input(self, ri):
        if self.raw_input:
            self.last_raw_input = self.raw_input
        self.raw_input = ri
        self.prediction_correct = self.prediction and self.prediction == self.raw_input
        util.rolling_list(self.rolling_prediction_correct, length=ROLLING_PREDICTION_COUNT, new_value=1 if self.prediction_correct else 0)

    def set_prediction(self, prediction):
        self.prediction = prediction

    def focus_cell(self, region, index):
        if index < len(region.cells):
            self.focus_cell_index = index
            cell = region.cells[index]
            self.cell_window.wm_title("Cell %d" % index)
            print("Focusing on %s" % (cell))
            for si, cg in enumerate(self.cell_window.grids):
                post_step = self.view_active('segment_post_step')
                if cg.tag == 'distal':
                    cg.cell_draw_fn = partial(VIEW.draw_cell_synapses, self, cell, segment_index=si, post_step=post_step, segment_type=cg.tag)
                    cg.grid_outline_fn = partial(VIEW.segment_learning, cell, segment_index=si, post_step=post_step)
                elif cg.tag == 'proximal':
                    cg.cell_draw_fn = partial(VIEW.draw_cell_synapses, self, cell, segment_index=si, post_step=post_step, segment_type=cg.tag)
                cg.render()

            if self.predictor_window:
                for si, cg in enumerate(self.predictor_window.grids):
                    cg.render()

    def render(self):
        for window in [self.main_window, self.cell_window, self.predictor_window]:
            if window:
                window.render(self.focus_cell_index)

        rolling_average = util.average(self.rolling_prediction_correct)
        self.rolling_match_history.append(rolling_average)
        self.prediction_correct_history.append(self.prediction_correct)
        self.main_window.update_values(last_input=self.last_raw_input, raw_input=self.raw_input,
            prediction=self.prediction, prediction_correct=self.prediction_correct,
            rolling_average=rolling_average, time=str(self.brain.t))

    def show_run_summary(self):
        bar_vals = self.rolling_match_history
        bar_colors = self.prediction_correct_history

        ind = np.arange(len(bar_vals))  # the x locations for the groups
        width = 1.0       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, bar_vals, width, color=['g' if match else 'r' for match in bar_colors])

        ax.set_title("Prediction improvement over time")
        ax.set_ylabel('Rolling Prediction Match (%)')
        ax.set_xlabel('Time Step')
        plt.show(block=False)

class CanvasWindow(Toplevel):

    def __init__(self, master, width=100, height=100):
        Toplevel.__init__(self, master)
        self.width = width
        self.height = height
        self.master = self.printer = master
        self.grids = []
        self.labeled_values = {} # string key -> canvas item
        self.canvas = Canvas(self, width=width, height=height)
        self.canvas.pack()

    def set_size(self, width=100, height=100):
        self.width = width
        self.height = height
        self.canvas.config(width=width, height=height)

    def move(self, x=0, y=0):
        self.geometry('%dx%d+%d+%d' % (self.width, self.height, x, y))

    def add_grid(self, cg):
        self.grids.append(cg)

    def initialize_grids(self):
        for cg in self.grids:
            cg.initialize()

    def render(self, focus_cell_index=None):
        for cg in self.grids:
            cg.render(focus_index=focus_cell_index)

    def _setup_labeled_value(self, key, label="Label", value="N/A", x=0, y=0, color="#000"):
        # Setup label
        self.canvas.create_text(x, y - LABEL_GAP, fill=color, font=("Purisa", LABEL_SIZE), anchor="s", text=label)
        # Setup value
        self.labeled_values[key] = self.canvas.create_text(x, y, fill=color, font=("Purisa", VALUE_SIZE), anchor="s")

    def _update_labeled_value(self, key, value="N/A", color=None):
        lv = self.labeled_values.get(key)
        if lv:
            kwargs = {}
            if color:
                kwargs['fill'] = color
            self.canvas.itemconfig(lv, text=value, **kwargs)


class MainWindow(CanvasWindow):

    def __init__(self, master, brain, on_focus_cell=None, max_side=100, *args, **kwargs):
        CanvasWindow.__init__(self, master, *args, **kwargs)
        self.brain = brain
        self.printer = master
        top = REGION_BUFFER

        max_height = max_cells = 0

        # Create input grid
        draw_fn = partial(VIEW.draw_input, self)
        input_rc = CellGrid(REGION_BUFFER, top, self.brain.n_inputs, canvas=self.canvas, cell_draw_fn=draw_fn, view_type=VIEW.INPUTS, title="Inputs") # Inputs
        self.add_grid(input_rc)
        x, _y = input_rc.bottom_right()
        # Create region grids
        for r in self.brain.regions:
            y = top
            side = r._cell_side_len()
            for view_index, view_type in enumerate(VIEW.DRAW):
                draw_fn = partial(DRAW_FN.get(view_type), self.printer, r)
                on_cell_click = partial(on_focus_cell, r)
                max_value = VIEW.MAX_VALUE.get(view_type)
                grid_title = VIEW.LABEL.get(view_type)
                if max_value is None:
                    max_value = 1.0
                cg = CellGrid(x + REGION_BUFFER, y, r.n_cells, canvas=self.canvas, view_type=view_type, cell_draw_fn=draw_fn, on_cell_click=on_cell_click, max_value=max_value, title=grid_title)
                if cg.side > max_side:
                    max_side = cg.side * CELL_PX
                if cg.n_cells > max_cells:
                    max_cells = cg.n_cells
                self.add_grid(cg)
                _x, y = cg.bottom_right()
                if y > max_height:
                    max_height = y
                y += REGION_BUFFER

            x = _x + REGION_BUFFER

        main_width = x+REGION_BUFFER
        main_height = max_height+REGION_BUFFER

        self._setup_labeled_value("last_input", label="Last Input", x=input_rc.x_middle(), y=main_height - 220, color="#CCC")
        self._setup_labeled_value("raw_input", label="Raw Input", x=input_rc.x_middle(), y=main_height - 180, color="#000")
        self._setup_labeled_value("prediction", label="Prediction", x=input_rc.x_middle(), y=main_height - 140, color="#ccc12a")
        self._setup_labeled_value("rolling_prediction_correct", label="Pred. Matches", x=input_rc.x_middle(), y=main_height - 100, color="#000")
        self._setup_labeled_value("time", label="Time Step", x=input_rc.x_middle(), y=main_height - 60, color="#000")

        self.set_size(width=main_width, height=main_height)
        self.initialize_grids()

    def update_values(self, last_input=None, raw_input=None, prediction=None,
            prediction_correct=False, rolling_average=None, time=0):
        if last_input:
            self._update_labeled_value('last_input', value=last_input)
        if raw_input:
            color = '#14FF49' if prediction_correct else '#000000'
            self._update_labeled_value('raw_input', value=raw_input, color=color)
        if prediction:
            self._update_labeled_value('prediction', value=prediction)
        if rolling_average:
            self._update_labeled_value('rolling_prediction_correct', value="%.2f" % rolling_average)
        self._update_labeled_value('time', value=time)


class CellWindow(CanvasWindow):

    def __init__(self, master, max_side=100, cpr=100):
        CanvasWindow.__init__(self, master)
        self.region = None
        width = 2*max_side + 3*REGION_BUFFER
        height = VIEWABLE_SEGMENTS * (max_side + REGION_BUFFER) + REGION_BUFFER
        self.set_size(width, height)

        for i in range(VIEWABLE_SEGMENTS):
            draw_fn = None
            grid_outline_fn = None
            on_cell_click = partial(self.focus_synapse, self.region, i)
            # Proximal seg
            self.add_grid(CellGrid(REGION_BUFFER, (i * (max_side + REGION_BUFFER)) + REGION_BUFFER, cpr, canvas=self.canvas, view_type=VIEW.SEGMENT_DETAIL, cell_draw_fn=draw_fn, grid_outline_fn=grid_outline_fn, saturate_value=0.2, max_value=0.5, on_cell_click=on_cell_click, title="Proximal Seg %d" % (i+1), tag="proximal"))
            # Distal seg
            self.add_grid(CellGrid(2*REGION_BUFFER + max_side, (i * (max_side + REGION_BUFFER)) + REGION_BUFFER, cpr, canvas=self.canvas, view_type=VIEW.SEGMENT_DETAIL, cell_draw_fn=draw_fn, grid_outline_fn=grid_outline_fn, saturate_value=0.2, max_value=0.5, on_cell_click=on_cell_click, title="Distal Seg %d" % (i+1), tag="distal"))
        self.initialize_grids()

    def focus_synapse(self, region, segment_index, index):
        if self.printer.focus_cell_index:
            cell = region.cells[self.printer.focus_cell_index]
            seg = cell.distal_segments[segment_index]
            perm, contr, last_change = seg.synapse_state(index)
            print("Synapse focus_cell=%s source_index=%s permanence=%s contribution=%s last_change=%s" % (self.focus_cell_index, index, perm, contr, last_change))


class PredictorWindow(CanvasWindow):

    def __init__(self, master, predictor=None, max_side=100, cpr=100):
        CanvasWindow.__init__(self, master)
        self.printer = master
        self.predictor = predictor
        raw_inputs = self.predictor.categories
        pwidth = len(raw_inputs) * (max_side + REGION_BUFFER) + REGION_BUFFER
        pheight = (len(raw_inputs)+1) * (max_side + REGION_BUFFER) + REGION_BUFFER
        self.set_size(pwidth, pheight)

        x = REGION_BUFFER
        for ri in raw_inputs:
            draw_fn = partial(self.overlap_draw_fn, ri)
            input_overlap_cg = CellGrid(x, REGION_BUFFER, cpr, canvas=self.canvas, view_type=VIEW.PREDICTOR, cell_draw_fn=draw_fn, saturate_value=0.2, max_value=0.5, title=ri)
            self.add_grid(input_overlap_cg)
            # Create an NxN grid of grids showing activations of input y followed by input x
            y = 2*REGION_BUFFER + max_side
            for ri_prior in raw_inputs:
                draw_fn = partial(self.sequence_draw_fn, ri_prior, ri)
                title = "(%s)%s" % (ri_prior, ri)
                self.add_grid(CellGrid(x, y, cpr, canvas=self.canvas, view_type=VIEW.ACTIVATION, cell_draw_fn=draw_fn, saturate_value=0.2, max_value=0.5, title=title))
                y += max_side + REGION_BUFFER
            x += max_side + REGION_BUFFER
        self.initialize_grids()

    def overlap_draw_fn(self, raw_input, index):
        region = self.predictor.region
        last_overlap_for_input = self.predictor.overlap_lookup.get(raw_input)
        value = 0
        if last_overlap_for_input is not None:
            value = last_overlap_for_input[index]
        ind_color = ol_color = None
        bias = region.bias[index]
        cell = region.cells[index]
        predicted = value and bias
        associated = False
        if value and self.printer.view_active("predicted_associations"):
            # Highlight cells with distal links to any predicted cells
            indexes = np.nonzero(region.bias)
            for idx in indexes[0]:
                distal_connections_to_focus = cell.count_distal_connections(idx)
                if distal_connections_to_focus > 0:
                    associated = True
                    break
        if associated and predicted:
            ind_color = "#FF8C0C"
        elif associated:
            ind_color = "#FFF10C"
        elif predicted:
            ind_color = "#FF1A0E"
        return (value, ind_color, ol_color)

    def sequence_draw_fn(self, raw_input, raw_input2, index):
        region = self.predictor.region
        key = raw_input + raw_input2
        last_activation_for_input_seq = self.predictor.activation_lookup.get(key)
        value = 0
        ind_color = ol_color = None
        if last_activation_for_input_seq is not None:
            value = last_activation_for_input_seq[index]
        return (value, ind_color, ol_color)

class CellGrid(object):
    '''
    Canvas to draw a view of one region and handle clicks
    '''

    def __init__(self, x, y, n_cells, canvas=None, view_type=None, cell_draw_fn=None, grid_outline_fn=None, on_cell_click=None, max_value=1.0, saturate_value=1.0, title=None, tag=None):
        self.x, self.y = x, y # upper left
        self.n_cells = n_cells
        self.max_value = max_value
        self.saturate_value = saturate_value
        self.saturate_color = None
        self.side = math.sqrt(n_cells)
        self.canvas = canvas
        self.view_type = view_type
        self.cell_draw_fn = cell_draw_fn
        self.grid_outline_fn = grid_outline_fn
        self.on_cell_click = on_cell_click # fn
        self.cell_rects = []
        self.hl_indicators = []
        self.temporary_widgets = []
        self.title = title
        self.hl = None
        self.tag = tag

    def __str__(self):
        return "View (type %d, xy: %s, %s)" % (self.view_type, self.x, self.y)

    def initialize(self):
        # Create frame background
        self.hl = self.canvas.create_rectangle(self.x, self.y, self.x + CELL_PX * self.side, self.y + CELL_PX * self.side, outline="#000")

        # Create cell rects & highlight items for rendering
        for index in range(self.n_cells):
            abs_x, abs_y = self._cell_position(index)
            cell_rect = self.canvas.create_rectangle(abs_x, abs_y, abs_x + CELL_PX, abs_y + CELL_PX, outline=GRID_OUTLINE, tags="cell_clickable")
            self.cell_rects.append(cell_rect)
            # Create indicator
            abs_x += CELL_PX / 4
            abs_y += CELL_PX / 4
            x_br = abs_x + INDICATOR_DIAMETER
            y_br = abs_y + INDICATOR_DIAMETER
            hl_indicator = self.canvas.create_oval(abs_x, abs_y, x_br, y_br, fill="", outline="", tags="cell_clickable")
            self.hl_indicators.append(hl_indicator)
            self.canvas.tag_bind(cell_rect, "<Button-1>", self.handle_click)
            self.canvas.tag_bind(hl_indicator, "<Button-1>", self.handle_click)
        if self.title:
            self.canvas.create_text((self.x, self.y - 9), text=self.title, fill="#000", font=("Purisa", 11), anchor="w")
        # print("Initialized %s" % self)

    def _cell_position(self, index):
        '''Return (x,y)'''
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

    def x_middle(self):
        return self.x + (self.side / 2)*CELL_PX

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
