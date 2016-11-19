#!/usr/bin/env python

import numpy as np
import random
import math
from pphtm import util
from pphtm.util import printarray

# Settings (global vars, other vars set in brain.__init__)

VERBOSITY = 0
DEF_MIN_OVERLAP = 2
CONNECTED_PERM = 0.2  # If the permanence value for a synapse is greater than this value, it is said to be connected.
DUTY_HISTORY = 40

INIT_PERMANENCE = 0.2  # New learned synapses
INIT_PERMANENCE_JITTER = 0.05  # Max offset from CONNECTED_PERM when initializing synapses
T_START_PROXIMAL_BOOSTING = 0
T_START_DISTAL_BOOSTING = -1
BIAS_DUTY_CUTOFF = 1.0
PROXIMITY_WEIGHTING = 1 # (bool) For proximal connection init.
LIMIT_BIAS_DUTY_CYCLE = 0.6 # If biased > 60% of recent history

def log(message, level=1):
    if VERBOSITY >= level:
        print message

class Segment(object):
    '''
    Dendrite segment of cell (proximal, distal, or top-down prediction)
    Store all synapses activations / connectedness in arrays

    Exploring a theory where timing of inputs and delay in predicted outputs
    is modulated by speed of action potential through axon and distance
    travelled, as well as distance from source of ap on postsynaptic dendrite
    segment to cell body. Latter distance may be modulated by synapses moving
    toward or away from cell body, which also enables different coincidence
    detectors to form.

    Questions: how to move synapse distance, learning rule? Based on delay, or
    current activation when segment activates?
    '''
    PROXIMAL = 1
    DISTAL = 2
    TOPDOWN = 3  # prediction

    def __init__(self, cell, index, region, type=None):
        self.index = index  # Segment index
        self.region = region
        self.cell = cell
        self.type = type if type else self.PROXIMAL

        # Synapses (all lists below len == # of synapses)
        self.syn_sources = []  # Index of source (either input or cell in region, or above)
        self.syn_permanences = []  # (0,1)
        self.syn_distances = []  # (0,1) Distance from cell body
        self.syn_change = []  # -1, 0 1 (after step)
        self.syn_prestep_contribution = []  # (after step)
        self.syn_contribution = []  # (after step)

        # State
        self.active_before_learning = False

    def __repr__(self):
        t = self.region.brain.t
        return "<Segment type=%s index=%d potential=%d connected=%d>" % (self.print_type(), self.index, self.n_synapses(), len(self.connected_synapses()))

    def initialize(self):
        if self.proximal():
            # Setup initial potential synapses for proximal segments
            n_inputs = self.region.n_inputs
            cell_x, cell_y = util.coords_from_index(self.cell.index, self.region._cell_side_len())
            for source in range(n_inputs):
                # Loop through all inputs and randomly choose to create synapse or not
                input_x, input_y = util.coords_from_index(source, self.region._input_side_len())
                dist = util.distance((cell_x, cell_y), (input_x, input_y))
                max_distance = self.region.diagonal
                chance_of_synapse = ((self.region.brain.config("MAX_PROXIMAL_INIT_SYNAPSE_CHANCE") - self.region.brain.config("MIN_PROXIMAL_INIT_SYNAPSE_CHANCE")) * (1 - float(dist)/max_distance)) + self.region.brain.config("MIN_PROXIMAL_INIT_SYNAPSE_CHANCE")
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(source)
        elif self.distal():
            cell_index = self.cell.index
            for index in range(self.region.n_cells):
                if index == cell_index:
                    # Avoid creating synapse with self
                    continue
                chance_of_synapse = self.region.brain.config("DISTAL_SYNAPSE_CHANCE")
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index)
        else:
            # Top down connections
            for index in range(self.region.n_cells_above):
                chance_of_synapse = self.region.brain.config("TOPDOWN_SYNAPSE_CHANCE")
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index)
        log_message = "Initialized %s" % self
        log(log_message)

    def proximal(self):
        return self.type == self.PROXIMAL

    def distal(self):
        return self.type == self.DISTAL

    def topdown(self):
        return self.type == self.TOPDOWN

    def add_synapse(self, source_index=0, permanence=None):
        if permanence is None:
            permanence = CONNECTED_PERM + INIT_PERMANENCE_JITTER*(random.random()-0.5)
        self.syn_sources.append(source_index)
        self.syn_permanences.append(permanence)
        self.syn_change.append(0)
        self.syn_prestep_contribution.append(0)
        self.syn_contribution.append(0)

    def synapse_state(self, index=0):
        '''
        index is index in region
        '''
        last_change = permanence = contribution = None
        if index in self.syn_sources:
            syn_index = self.syn_sources.index(index)
            last_change = self.syn_change[syn_index]
            permanence = self.syn_permanences[syn_index]
            contribution = self.syn_contribution[syn_index]
        return (permanence, contribution, last_change)

    def source_cell(self, synapse_index=0):
        '''
        Args:
            synapse_index (int)

        Returns:
            Cell() object in own region or region above (if top-down segment)
        '''
        source = self.source(synapse_index)
        if self.proximal():
            region_below = self.region.region_below()
            if region_below:
                return region_below.cells[source]
        elif self.distal():
            # Cell in region
            return self.region.cells[source]
        else:
            # Top-down
            region_above = self.region.region_above()
            if region_above:
                return region_above.cells[source]
        return None

    def decay_permanences(self):
        '''Reduce connected permanences by a small decay factor.
        '''
        prox_decay = self.region.brain.config("SYNAPSE_DECAY_PROX")
        distal_decay = self.region.brain.config("SYNAPSE_DECAY_DIST")
        decay = prox_decay if self.proximal() else distal_decay
        for syn in self.connected_synapses():
            self.syn_permanences[syn] -= decay

    def distance_from(self, coords_xy, index=0):
        source_xy = util.coords_from_index(self.syn_sources[index], self.region._input_side_len())
        return util.distance(source_xy, coords_xy)

    def connected(self, index=0, connectionPermanence=CONNECTED_PERM):
        return self.syn_permanences[index] > connectionPermanence

    def print_type(self):
        return {
            self.PROXIMAL: "Proximal",
            self.DISTAL: "Distal",
            self.TOPDOWN: "Top-Down"
        }.get(self.type)

    def contribution(self, syn_index=0, absolute=False):
        '''Returns would-be contribution from a synapse. Permanence is not considered.

        Args:
            syn_index is a synapse index (within segment, not full region)

        Returns:
            double: a contribution (activation) [-1.0,1.0] of synapse at index
        '''
        if self.proximal():
            activation = self.region._input_active(self.syn_sources[syn_index])
            mult = 1
        else:
            # distal or top-down
            cell = self.source_cell(syn_index)
            activation = cell.activation
            mult = 1 if cell.excitatory else -1
        if not absolute:
            activation *= mult
        return activation

    def total_activation(self):
        '''Returns sum of contributions from all connected synapses.

        Return: double (not bounded)
        '''
        connected_syn_sources = np.take(np.asarray(self.syn_sources, dtype='int32'), self.connected_synapses())
        if self.proximal():
            region_below = self.region.region_below()
            excitatory_activation_mult = region_below.excitatory_activation_mult if region_below else np.ones(self.region.n_inputs)
            return sum(np.take(self.region.input * excitatory_activation_mult, connected_syn_sources))
        elif self.distal():
            DISTAL_FLOOR = 0.8
            filtered_activation = self.region.activation > DISTAL_FLOOR
            return sum(np.take(filtered_activation * self.region.excitatory_activation_mult, connected_syn_sources))
        else:
            # top down
            region_above = self.region.region_above()
            if region_above:
                connected_syn_sources = np.take(self.syn_sources, self.connected_synapses())
                return sum(np.take(region_above.activation * region_above.excitatory_activation_mult, connected_syn_sources))
            else:
                return 0.0

    def threshold(self):
        threshold = self.region.brain.config("PROXIMAL_ACTIVATION_THRESHHOLD") if self.proximal() else self.region.brain.config("DISTAL_ACTIVATION_THRESHOLD")
        return threshold

    def active(self):
        '''
        TODO: Cache this?
        '''
        return self.total_activation() >= self.threshold()

    def n_synapses(self):
        return len(self.syn_sources)

    def source(self, index):
        return self.syn_sources[index]

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        '''
        Return array of cell indexes (in segment)
        '''
        connected_indexes = [i for i,p in enumerate(self.syn_permanences) if p >= connectionPermanence]
        return connected_indexes


class Cell(object):
    '''
    An HTM abstraction of one or more biological neurons
    Has multiple dendrite segments connected to inputs
    '''

    def __init__(self, region, index):
        self.index = index
        self.region = region
        self.n_proximal_segments = self.region.brain.config("PROX_SEGMENTS")
        self.n_distal_segments = self.region.brain.config("DISTAL_SEGMENTS")
        self.n_topdown_segments = self.region.brain.config("TOPDOWN_SEGMENTS") if not self.region.is_top() else 0
        self.distal_segments = []
        self.proximal_segments = []
        self.topdown_segments = []
        self.activation = 0.0 # [0.0, 1.0]
        self.coords = util.coords_from_index(index, self.region._cell_side_len())
        self.fade_rate = self.region.brain.config("FADE_RATE")
        self.excitatory = random.random() > self.region.brain.config("CHANCE_OF_INHIBITORY")

        # History
        self.recent_active_duty = []  # After inhibition, list of bool
        self.recent_overlap_duty = []  # Overlap > min_overlap, list of bool
        self.recent_bias_duty = []  # Bias > BIAS_DUTY_CUTOFF, list of bool

    def __repr__(self):
        return "<Cell index=%d activation=%.1f bias=%s overlap=%s />" % (self.index, self.activation, self.region.bias[self.index], self.region.overlap[self.index])

    def initialize(self):
        for i in range(self.n_proximal_segments):
            proximal = Segment(self, i, self.region, type=Segment.PROXIMAL)
            proximal.initialize() # Creates synapses
            self.proximal_segments.append(proximal)
        for i in range(self.n_distal_segments):
            distal = Segment(self, i, self.region, type=Segment.DISTAL)
            distal.initialize() # Creates synapses
            self.distal_segments.append(distal)
        for i in range(self.n_topdown_segments):
            topdown = Segment(self, i, self.region, type=Segment.TOPDOWN)
            topdown.initialize() # Creates synapses
            self.topdown_segments.append(topdown)
        log("Initialized %s" % self)

    def all_segments(self):
        return self.proximal_segments + self.distal_segments + self.topdown_segments

    def active_segments(self, type=Segment.PROXIMAL):
        segs = {
            Segment.PROXIMAL: self.proximal_segments,
            Segment.DISTAL: self.distal_segments,
            Segment.TOPDOWN: self.topdown_segments
        }.get(type)
        return filter(lambda seg : seg.active(), segs)

    def update_duty_cycles(self, active=False, overlap=False, bias=False):
        '''
        Add current active state & overlap state to history and recalculate duty cycles
        '''
        self.recent_active_duty.insert(0, 1 if active else 0)
        self.recent_overlap_duty.insert(0, 1 if overlap else 0)
        self.recent_bias_duty.insert(0, 1 if bias else 0)
        # Truncate
        if len(self.recent_active_duty) > DUTY_HISTORY:
            self.recent_active_duty = self.recent_active_duty[:DUTY_HISTORY]
        if len(self.recent_overlap_duty) > DUTY_HISTORY:
            self.recent_overlap_duty = self.recent_overlap_duty[:DUTY_HISTORY]
        if len(self.recent_bias_duty) > DUTY_HISTORY:
            self.recent_bias_duty = self.recent_bias_duty[:DUTY_HISTORY]
        cell_active_duty = sum(self.recent_active_duty) / float(len(self.recent_active_duty))
        cell_overlap_duty = sum(self.recent_overlap_duty) / float(len(self.recent_overlap_duty))
        cell_bias_duty = sum(self.recent_bias_duty) / float(len(self.recent_bias_duty))
        self.region.active_duty_cycle[self.index] = cell_active_duty
        self.region.overlap_duty_cycle[self.index] = cell_overlap_duty
        self.region.bias_duty_cycle[self.index] = cell_bias_duty

    def connected_receptive_field_size(self):
        '''
        Returns max distance (radius) among currently connected proximal synapses
        '''
        connected_indexes = self.connected_synapses(type=Segment.PROXIMAL)
        side_len = self.region._input_side_len()
        distances = []
        for i in connected_indexes:
            distance = util.distance(util.coords_from_index(i, side_len), self.coords)
            distances.append(distance)
        if distances:
            return max(distances)
        else:
            return 0

    def connected_synapses(self, type=Segment.PROXIMAL):
        synapses = []
        segs = self.proximal_segments if type == Segment.PROXIMAL else self.distal_segments
        for seg in segs:
            synapses.extend(seg.connected_synapses())
        return synapses

    def count_distal_connections(self, cell_index):
        synapses = self.connected_synapses(type=Segment.DISTAL)
        return synapses.count(cell_index)

    def most_active_segment(self, type="distal"):
        if type == "distal":
            return sorted(self.distal_segments, key=lambda seg : seg.total_activation(), reverse=True)[0]
        if type == "topdown":
            return sorted(self.topdown_segments, key=lambda seg : seg.total_activation(), reverse=True)[0]
        else:
            return None

class Region(object):
    '''
    Made up of many columns
    '''

    def __init__(self, brain, index, n_cells=10, n_inputs=20, n_cells_above=0):
        self.index = index
        self.brain = brain

        # Region constants (spatial)
        self.inhibition_radius = 0

        # Hierarchichal setup
        self.n_cells = n_cells
        self.n_cells_above = n_cells_above
        self.n_inputs = n_inputs
        self.cells = []

        # State (historical)
        self.input = None  # Inputs at time t - input[t, j] is double [0.0, 1.0]

        # State
        self.overlap = np.zeros(self.n_cells)  # Overlap for each cell. overlap[c] is double
        self.pre_bias = np.zeros(self.n_cells)  # Prestep bias
        self.bias = np.zeros(self.n_cells)  # Bias for each cell. overlap[c] is double
        self.pre_activation = np.zeros(self.n_cells)  # Activation before inhibition for each cell.
        self.activation = np.zeros(self.n_cells)
        self.boost = np.ones(self.n_cells, dtype=float)  # Boost value for cell c
        self.overlap_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has had significant overlap (> min_overlap)
        self.active_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has been active after inhibition
        self.bias_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has been biased (distal or topdown)
        self.last_activation = None  # Hold last step in state for rendering

        # Helpers
        self.diagonal = 1.414*2*math.sqrt(n_cells)
        self.excitatory_activation_mult = np.ones(self.n_cells)  # Matrix of 1 or -1 for each cell (after initialization)


    def __str__(self):
        return "<Region inputs=%d cells=%d />" % (self.n_inputs, len(self.cells))

    def print_cells(self):
        activations = [cell.activation for cell in self.cells]
        return printarray(activations, continuous=True)

    def initialize(self):
        # Create cells
        for i in range(self.n_cells):
            c = Cell(region=self, index=i)
            c.initialize()
            if not c.excitatory:
                self.excitatory_activation_mult[i] = -1
            self.cells.append(c)
        log("Initialized %s" % self)

    def region_above(self):
        if self.index < len(self.brain.regions) - 1:
            return self.brain.regions[self.index + 1]
        return None

    def region_below(self):
        if self.index > 0:
            return self.brain.regions[self.index - 1]
        return None

    def is_top(self):
        return self.index == len(self.brain.regions) - 1

    def _input_side_len(self):
        return math.sqrt(self.n_inputs)

    def _cell_side_len(self):
        return math.sqrt(self.n_cells)

    def _get_active_cells(self):
        pass

    def _input_active(self, j):
        '''
        At index j, at time t - tMinus
        '''
        return self.input[j]

    def _kth_score(self, cells, values, k):
        '''
        Given list of cells, calculate kth highest overlap value
        '''
        if cells:
            cell_values = [values[c.index] for c in cells]
            _k = min([k, len(values)]) # TODO: Ok to pick last if k > overlaps?
            cell_values = sorted(cell_values, reverse=True) # Highest to lowest
            kth_value = cell_values[_k-1]
            # log("%s is %dth highest overlap score in sequence: %s" % (kth_value, k, overlaps))
            return kth_value
        return 0 # Shouldn't happen?

    def _max_duty_cycle(self, cells):
        if cells:
            return max([self.active_duty_cycle[c.index] for c in cells])
        else:
            return 0

    def _neighbors_of(self, cell):
        '''
        Return all cells within inhibition radius
        '''
        _neighbors = []
        for c in self.cells:
            if cell.index == c.index:
                continue
            dist = util.dist_from_indexes(c.index, cell.index, self._cell_side_len())
            if dist <= self.inhibition_radius:
                _neighbors.append(c)
        # log("Got %d neighbors. Overlaps: %s" % (len(_neighbors), [self.overlap[n.index] for n in _neighbors]))
        return _neighbors

    def _boost_function(self, c, min_duty_cycle):
        if self.active_duty_cycle[c] >= min_duty_cycle:
            b = 1.0
        else:
            b = 1 + (min_duty_cycle - self.active_duty_cycle[c]) * self.brain.config("BOOST_MULTIPLIER")
        return b

    def _increase_cell_permanences(self, c, increase, excitatory=True, type="proximal"):
        '''
        Increase the permanence value of every matching synapse of cell c by a increase factor
        Currently increases synapses across all segments

        Args:
            excitatory (bool): Increase excitatory or inhibitory synapses
            type (string): Which segments to increase

        TODO: Should this be for a specific segment
        '''
        cell = self.cells[c]
        if type == "proximal":
            for seg in self.cells[c].proximal_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if (source_cell and source_cell.excitatory == excitatory):
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

        elif type == "distal":
            for seg in self.cells[c].distal_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if source_cell.excitatory == excitatory:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

        elif type == "topdown":
            for seg in self.cells[c].topdown_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if source_cell.excitatory == excitatory:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

    def calculate_biases(self):
        '''
        For each cell calculate aggregate activations from distal segments
        Result is a bias array that will be used during overlap to increase
        chances we activate 'predicted' cells

        For PP bias array now includes top-down biases

        Updates active_before_learning for each segment for next step

        TODO: weight topdown vs distal
        '''
        bias = np.zeros(len(self.cells))  # Initialize bias to 0
        for i, c in enumerate(self.cells):
            for seg in (c.distal_segments + c.topdown_segments):
                seg.active_before_learning = seg.active()
                increment = 1 if seg.distal() else self.brain.config("TOPDOWN_BIAS_WEIGHT")
                if seg.active_before_learning:
                    bias[i] += increment
        return bias

    def do_overlap(self):
        '''
        Return overlap as a double for each cell representing boosted
        activation from proximal inputs
        '''
        overlaps = np.zeros(len(self.cells))  # Initialize overlaps to 0
        for i, c in enumerate(self.cells):
            for seg in c.proximal_segments:
                seg.active_before_learning = seg.active()
                if seg.active_before_learning:
                    overlaps[i] += 1
            # Note this boost is calculated on prior step
            overlaps[i] *= self.boost[i]
        return overlaps

    def do_inhibition(self):
        '''
        Inputs:
            - Overlaps (proximal)
            - Bias (distal/topdown)
        Sum the two (weighted) and choose winners (active outputs)

        TODO: Try modulating weighting based on recent distal/topdown vs proximal activity
        '''
        self.pre_activation = (self.brain.config("OVERLAP_WEIGHT") * self.overlap) * (1 + (self.brain.config("BIAS_WEIGHT") * self.bias))
        # self.pre_activation = (self.brain.config("OVERLAP_WEIGHT") * self.overlap) + (self.brain.config("BIAS_WEIGHT") * self.bias)
        active = np.zeros(len(self.cells))
        for c in self.cells:
            pa = self.pre_activation[c.index]
            neighbors = self._neighbors_of(c)
            kth_score = self._kth_score(neighbors, self.pre_activation, k=self.brain.config("DESIRED_LOCAL_ACTIVITY"))
            if pa > 0 and pa >= kth_score:
                active[c.index] = True
        return active

    def learn_segment(self, seg, is_activating=False, is_biased=False):
        '''Update synapse permanences for this segment.

        Synapse's permanence will increase if::
            a) it is contributing from an excitatory source, and
            b) cell is activating

        Synapse's permanence will decrease if::
            a) it is contributing from an excitatory source, and
            b) cell is not activating, but is biased

        TODO: Move all learning logic here for clarity?
        '''
        n_inc = n_dec = n_conn = n_discon = 0
        active = seg.active_before_learning
        for i in range(seg.n_synapses()):
            seg.syn_change[i] = 0
            was_connected = seg.connected(i)
            source_cell = seg.source_cell(i)
            source_excitatory = not source_cell or source_cell.excitatory # Inputs excitatory
            contribution = seg.contribution(i, absolute=True)
            learn_threshold = self.brain.config("DIST_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD") if (seg.distal() or seg.topdown()) else self.brain.config("PROX_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD")
            contributor = contribution >= learn_threshold
            seg.syn_contribution[i] = contributor
            increase_permanence = decrease_permanence = False
            if seg.proximal():
                change_permanence = seg.active_before_learning and is_activating and source_excitatory
            else:
                change_permanence = contributor and source_excitatory
            if change_permanence:
                if seg.proximal():
                    increase_permanence = contributor
                    decrease_permanence = False # (just decay) not contributor
                else:
                    increase_permanence = is_activating
                    decrease_permanence = not is_activating and is_biased
                if increase_permanence and seg.syn_permanences[i] < 1.0:
                    n_inc += 1
                    seg.syn_permanences[i] += self.brain.config("PERM_LEARN_INC")
                    seg.syn_permanences[i] = min(1.0, seg.syn_permanences[i])
                    seg.syn_change[i] += 1
                elif decrease_permanence and seg.syn_permanences[i] > 0.0:
                    n_dec +=1
                    seg.syn_permanences[i] -= self.brain.config("PERM_LEARN_DEC")
                    seg.syn_permanences[i] = max(0.0, seg.syn_permanences[i])
                    seg.syn_change[i] -= 1
                connection_changed = was_connected != seg.connected(i)
                if connection_changed:
                    connected = not was_connected
                    if connected:
                        n_conn += 1
                    else:
                        n_discon += 1
        return (n_inc, n_dec, n_conn, n_discon)

    def do_learning(self, activating):
        '''Update permanences for each segment according to learning rules.

        Proximal:
            Learn segments to activating cells

        Distal / Topdown:
            If cell activating: learn active segments, or all if none active
            If cell not activating, but is biased: learn on active segments

        Args:
            activating (np.array - bool): Activation in this step after inhibition
        '''
        n_increased_prox = n_decreased_prox = n_increased_dist = n_decreased_dist = n_conn_prox = n_discon_prox = n_conn_dist = n_discon_dist = 0
        for i, cell_is_activating in enumerate(activating):
            cell = self.cells[i]
            cell_biased = self.bias[i] # Thresh?
            any_distal_active = any([seg.active_before_learning for seg in cell.distal_segments])
            any_topdown_active = any([seg.active_before_learning for seg in cell.topdown_segments])
            for seg in cell.all_segments():
                seg_active = seg.active_before_learning
                segment_learns = False
                if seg.proximal():
                    segment_learns = cell_is_activating
                elif seg.distal():
                    segment_learns = (cell_is_activating and (seg_active or not any_distal_active)) or \
                        (cell_biased and not cell_is_activating and seg_active)
                elif seg.topdown():
                    segment_learns = (cell_is_activating and (seg_active or not any_topdown_active)) or \
                        (cell_biased and not cell_is_activating and seg_active)

                if segment_learns:
                    ni, nd, nc, ndc = self.learn_segment(seg, is_activating=cell_is_activating, is_biased=cell_biased)
                    n_increased_prox += ni
                    n_decreased_prox += nd
                    n_conn_prox += nc
                    n_discon_prox += ndc
                else:
                    # Re-initialize change state
                    seg.decay_permanences()
                    seg.syn_change = [0 for x in seg.syn_change]

        log("Distal/Topdown: +%d/-%d (%d connected, %d disconnected)" % (n_increased_dist, n_decreased_dist, n_conn_dist, n_discon_dist))

        n_boosted = 0
        all_field_sizes = []
        for i, cell in enumerate(self.cells):
            neighbors = self._neighbors_of(cell)
            min_duty_cycle = 0.01 * self._max_duty_cycle(neighbors) # Based on active duty
            cell_active = activating[i]
            sufficient_overlap = self.overlap[i] > self.brain.min_overlap
            biased = self.bias[i] > BIAS_DUTY_CUTOFF
            cell.update_duty_cycles(active=cell_active, overlap=sufficient_overlap, bias=biased)
            boost_inhibitory = self.bias_duty_cycle[i] > LIMIT_BIAS_DUTY_CYCLE
            if boost_inhibitory:
                self._increase_cell_permanences(i, self.brain.config("DISTAL_BOOST_MULT") * CONNECTED_PERM, type="distal", excitatory=False)
                self._increase_cell_permanences(i, self.brain.config("DISTAL_BOOST_MULT") * CONNECTED_PERM, type="topdown", excitatory=False)
            if T_START_PROXIMAL_BOOSTING != -1 and self.brain.t > T_START_PROXIMAL_BOOSTING:
                self.boost[i] = self._boost_function(i, min_duty_cycle)  # Updates boost value for cell (higher if below min)

                # Check if overlap duty cycle less than minimum (note: min is calculated from max *active* not overlap)
                if self.overlap_duty_cycle[i] < min_duty_cycle:
                    # log("Increasing permanences for cell %s in region %d due to overlap duty cycle below min: %s" % (i, self.index, min_duty_cycle))
                    self._increase_cell_permanences(i, 0.1 * CONNECTED_PERM, type="proximal")
                    n_boosted += 1

            if T_START_DISTAL_BOOSTING != -1 and self.brain.t > T_START_DISTAL_BOOSTING:
                if self.bias_duty_cycle[i] < min_duty_cycle:
                    # TODO: Confirm this is working
                    self._increase_cell_permanences(i, self.brain.config("DISTAL_BOOST_MULT") * CONNECTED_PERM, type="distal")
                    self._increase_cell_permanences(i, self.brain.config("DISTAL_BOOST_MULT") * CONNECTED_PERM, type="topdown")
                    n_boosted += 1
            all_field_sizes.append(self.cells[i].connected_receptive_field_size())

        if n_boosted:
            log("Boosting %d due to low overlap duty cycle" % n_boosted)

        # Update inhibition radius (based on updated active connections in each column)
        self.inhibition_radius = util.average(all_field_sizes) * self.brain.config("INHIBITION_RADIUS_DISCOUNT")
        min_positive_radius = 1.0
        if self.inhibition_radius and self.inhibition_radius < min_positive_radius:
            self.inhibition_radius = min_positive_radius
        log("Setting inhibition radius to %s" % self.inhibition_radius)

    def tempero_spatial_pooling(self, learning_enabled=True):
        '''
        Temporal-Spatial pooling routine
        --------------
        Takes input and calculates active columns (sparse representation)
        This is a representation of input in context of previous input
        (same region and regions above -- top-down predictions)
        '''

        # Phase 1: Overlap
        self.overlap = self.do_overlap()
        if VERBOSITY >= 2: log("%s << Overlap (normalized)" % printarray(self.overlap, continuous=True), level=2)

        # Phase 2: Inhibition
        activating = self.do_inhibition()
        if VERBOSITY >= 2: log("%s << Activating (inhibited)" % printarray(activating, continuous=True), level=2)


        # Phase 3: Learning
        if learning_enabled:
            if VERBOSITY >= 2: log("%s << Active Duty Cycle" % printarray(self.active_duty_cycle, continuous=True), level=2)
            if VERBOSITY >= 2: log("%s << Overlap Duty Cycle" % printarray(self.overlap_duty_cycle, continuous=True), level=2)
            self.do_learning(activating)


        # Phase 4: Calculate new activations
        # Save pre-step activations
        self.last_activation = [cell.activation for cell in self.cells]
        # Update activations
        for i, cell in enumerate(self.cells):
            if activating[i]:
                cell.activation = 1.0  # Max out
            else:
                cell.activation -= cell.fade_rate
            if cell.activation < 0:
                cell.activation = 0.0
            self.activation[i] = cell.activation
        if VERBOSITY >= 2: log("%s << Activations" % self.print_cells(), level=2)


        # Phase 5: Calculate Distal Biases (TM?)
        self.pre_bias = np.copy(self.bias)
        self.bias = self.calculate_biases()
        if VERBOSITY >= 2: log("%s << Bias" % printarray(self.bias, continuous=True), level=2)

        if VERBOSITY >= 1:
            # Log average synapse permanence in region
            permanences = []
            n_connected = 0
            n_synapses = 0
            for cell in self.cells:
                for seg in cell.proximal_segments:
                    permanences.append(util.average(seg.syn_permanences))
                    n_synapses += seg.n_synapses()
                    n_connected += len(filter(lambda x : x > CONNECTED_PERM, seg.syn_permanences))
            ave_permanence = util.average(permanences)
            p_connected = (n_connected/float(n_synapses))*100. if n_synapses else 0.0
            log("R%d - average proximal synapse permanence: %.1f (%.1f%% connected of %d)" % (self.index, ave_permanence, p_connected, n_synapses), level=1)



    ##################
    # Primary Step Function
    ##################

    def step(self, input, learning_enabled=False):
        '''Process inputs for a region.

        After step is complete:
            * self.bias is an array of biases based on activations after step
            * self.overlap is an array based on proximal connections to new inputs
            * self.last_activation is an array of activations before new input
            * self.pre_bias is an array of biases before new input
            * cell.activation for each cell is updated
        '''
        self.input = input

        self.tempero_spatial_pooling(learning_enabled=learning_enabled)  # Calculates active cells

        return [c.activation for c in self.cells]


class PPHTMBrain(object):
    '''
    Predictive Processing implementation of HTM.
    With added continuous timing strategy.
    '''

    def __init__(self, min_overlap=DEF_MIN_OVERLAP, r1_inputs=1, seed=None):
        self.regions = []
        self.t = 0
        self.active_behaviors = []
        self.inputs = None
        self.seed = seed

        # Brain config
        self.n_inputs = r1_inputs
        self.min_overlap = min_overlap # A minimum number of inputs that must be active for a column to be considered during the inhibition step
        # Defaults
        self.CONFIG = {
            "PROXIMAL_ACTIVATION_THRESHHOLD": 3,
            "DISTAL_ACTIVATION_THRESHOLD": 2,
            "BOOST_MULTIPLIER": 2.58,
            "DESIRED_LOCAL_ACTIVITY": 2,
            "DISTAL_SYNAPSE_CHANCE": 0.4,
            "TOPDOWN_SYNAPSE_CHANCE": 0.3,
            "MAX_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.4,
            "MIN_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.1,
            "CELLS_PER_REGION": 14**2,
            "N_REGIONS": 2,
            "BIAS_WEIGHT": 0.6,
            "OVERLAP_WEIGHT": 0.4,
            "FADE_RATE": 0.5,
            "DISTAL_SEGMENTS": 3,
            "PROX_SEGMENTS": 2,
            "TOPDOWN_SEGMENTS": 1,
            "TOPDOWN_BIAS_WEIGHT": 0.5,
            "SYNAPSE_DECAY_PROX": 0.00005,
            "SYNAPSE_DECAY_DIST": 0.0,
            "PERM_LEARN_INC": 0.07,
            "PERM_LEARN_DEC": 0.04,
            "CHANCE_OF_INHIBITORY": 0.1,
            "DIST_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD": 1.0,
            "PROX_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD": 0.5,
            "DISTAL_BOOST_MULT": 0.01,
            "INHIBITION_RADIUS_DISCOUNT": 0.8
        }


    def __repr__(self):
        return "<PPHTMBrain regions=%d>" % len(self.regions)

    def config(self, key):
        return self.CONFIG.get(key)

    def initialize(self, n_inputs=None, **params):
        self.CONFIG.update(params)

        if n_inputs is not None:
            self.n_inputs = n_inputs

        if self.seed is not None:
            random.seed(self.seed)

        n_inputs = self.n_inputs

        # Initialize and create regions and cells
        self.regions = []
        for i in range(self.config("N_REGIONS")):
            top_region = i == self.config("N_REGIONS") - 1
            cpr = self.config("CELLS_PER_REGION")
            r = Region(self, i, n_inputs=n_inputs, n_cells=cpr, n_cells_above=cpr if not top_region else 0)
            r.initialize()
            n_inputs = cpr  # Next region will have 1 input for each output cell
            self.regions.append(r)
        self.t = 0
        log("Initialized %s" % self)

    def get_anomaly_score(self):
        # should we be using standard bias?
        r0 = self.regions[0]
        biasedCells = r0.pre_bias > 0
        overlappedCells = r0.overlap > 0
        nBiasedCells = np.sum(biasedCells)
        nPredictedCells = np.sum(overlappedCells & biasedCells)
        matchScore = nPredictedCells / float(nBiasedCells) if nBiasedCells > 0 else 0.0
        anomalyScore = 1. - matchScore
        if anomalyScore < 0.0 or anomalyScore > 1.0:
            print biasedCells
            print overlappedCells
            print nBiasedCells
            print nPredictedCells
            print matchScore
            print anomalyScore
            raise Exception("anomalyScore out of range: %s" % anomalyScore)
        return (anomalyScore, nBiasedCells, nPredictedCells)


    def process(self, readings, learning=False):
        '''
        Step through all regions inputting output of each into next
        Returns output of last region
        '''
        log("~~~~~~~~~~~~~~~~~ Processing inputs at T%d" % self.t, level=1)
        self.inputs = readings
        _in = self.inputs
        for i, r in enumerate(self.regions):
            log("Step processing for region %d\n%s << Input" % (i, printarray(_in, continuous=True)), level=2)
            out = r.step(_in, learning_enabled=learning)
            _in = out
        self.t += 1 # Move time forward one step
        return out
