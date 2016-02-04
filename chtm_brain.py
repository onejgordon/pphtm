#!/usr/bin/env python

# Jeremy Gordon

# Basic implementation of HTM network (Neumenta, Hawkins)
# Adjusted to support continuous activations that fall off over time at different rates
# Temporal pooler removed in favor of spatial-temperal pooling
# http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf


# CURRENT PROBLEMS
# -----------------------

# Once a region has learned, bias should be a subset of proximal overlap on each step

# All segments are learning same patterns? How to choose which to learn?
# A->B, since it's less frequent, is unable to learn this transition (we
# unlearn it every time E->B happens). Should we only learn on active segments?
# If so, how do we ensure we ever have activity? Should we learn on most
# active segment? Doesn't work because we then only learn on one seg each cell

# Should we unlearn distabl predictions that never activate? Causing noise
# in bias layer that doesn't go away.

# Looks like we are learning too much on one segment (active),
# straddline multiple patterns.

# TODO
# Render proximal connections (and re-initialize changes there too)
# Should we try boosting distal synapses if active duty cycle low?
# Print a learning metric (bias/overlap alignment?) to monitor progress in a run
# Run a few more times to see if wecan consistently learn ABCD/EBCF

# Now it's time to build invariant SDRs at a higher region layer.
# These can predict all pattern members regardless of order.
# -----------------------

import numpy as np
import random
import math
import util

# Settings

VERBOSITY = 1
PROXIMAL_ACTIVATION_THRESHHOLD = 2 # Activation threshold for a segment. If the number of active connected synapses in a segment is greater than activationThreshold, the segment is said to be active.
DISTAL_ACTIVATION_THRESHOLD = 2.5
DEF_MIN_OVERLAP = 2
CONNECTED_PERM = 0.2  # If the permanence value for a synapse is greater than this value, it is said to be connected.
DUTY_HISTORY = 100
BOOST_MULTIPLIER = 1.5
INHIBITION_RADIUS_DISCOUNT = 0.8
INIT_PERMANENCE = 0.2  # New learned synapses
INIT_PERMANENCE_JITTER = 0.05  # Max offset from CONNECTED_PERM when initializing synapses
SYNAPSE_ACTIVATION_LEARN_THRESHHOLD = 0.8
INIT_PERMANENCE_LEARN_INC_CHANGE = 0.02
INIT_PERMANENCE_LEARN_DEC_CHANGE = 0.005
DESIRED_LOCAL_ACTIVITY = 2
DO_BOOSTING = True
CHANCE_OF_INHIBITORY = 0.0
DISTAL_BIAS_EFFECT = 0.8
OVERLAP_EFFECT = 0.5
T_START_BOOSTING = 50
MIN_FADE_RATE, MAX_FADE_RATE = (0.2, 0.5)

DISTAL_SYNAPSE_CHANCE = 0.2
MAX_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.3
MIN_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.05

DISTAL_SEGMENTS = 3
PROX_SEGMENTS = 2

BOOST_DISTAL = False

def log(message, level=1):
    if VERBOSITY >= level:
        print message

def printarray(array, coerce_to_int=True, continuous=False):
    if continuous:
        # Takes an array of doubles
        out = ""
        _max = max(array)
        if _max:
            normalized = [x/_max for x in array]
        else:
            normalized = array
        for item in normalized:
            if math.isnan(item):
                simplified = "?"
            else:
                if item < 0:
                    simplified = "N" # Negative
                else:
                    simplified = str(int(item*5))
                    if simplified == "0":
                        simplified = "."
            out += simplified
        out += " (max: %.1f)" % _max
        return out
    else:
        if type(array[0]) is int or coerce_to_int:
            return ''.join([str(int(x)) for x in array])
        elif type(array[0]) in [float, np.float64]:
            return '|'.join([str(int(x)) for x in array])


class Segment(object):
    '''
    Dendrite segment of cell (proximal or distal)
    Store all synapses activations / connectedness in arrays
    '''
    PROXIMAL = 1
    DISTAL = 2

    def __init__(self, cell, index, region, type=None):
        self.index = index
        self.region = region
        self.cell = cell
        self.type = type if type else self.PROXIMAL

        # Synapses
        self.syn_sources = [] # Index of source (either input or cell in region)
        self.syn_permanences = [] # (0,1)
        self.syn_last_change = [] # -1, 0 1

    def __repr__(self):
        t = self.region.brain.t
        return "<Segment type=%s index=%d potential=%d connected=%d>" % (self.print_type(), self.index, self.n_synapses(), len(self.connected_synapses()))

    def initialize(self):
        if self.type == self.PROXIMAL:
            # Setup initial potential synapses for proximal segments
            n_inputs = self.region.n_inputs
            cell_x, cell_y = util.coords_from_index(self.cell.index, self.region._cell_side_len())
            for source in range(n_inputs):
                # Loop through all inputs and randomly choose to create synapse or not
                input_x, input_y = util.coords_from_index(source, self.region._input_side_len())
                dist = util.distance((cell_x, cell_y), (input_x, input_y))
                max_distance = self.region.diagonal
                chance_of_synapse = ((MAX_PROXIMAL_INIT_SYNAPSE_CHANCE - MIN_PROXIMAL_INIT_SYNAPSE_CHANCE) * (1 - float(dist)/max_distance)) + MIN_PROXIMAL_INIT_SYNAPSE_CHANCE
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(source)
        else:
            # Distal
            cell_index = self.cell.index
            for index in range(self.region.n_cells):
                if index == cell_index:
                    # Avoid creating synapse with self
                    continue
                chance_of_synapse = DISTAL_SYNAPSE_CHANCE
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index)
        log_message = "Initialized %s" % self
        log(log_message)

    def proximal(self):
        return self.type == self.PROXIMAL

    def add_synapse(self, source_index=0, permanence=None):
        if permanence is None:
            permanence = CONNECTED_PERM + INIT_PERMANENCE_JITTER*(random.random()-0.5)
        self.syn_sources.append(source_index)
        self.syn_permanences.append(permanence)
        self.syn_last_change.append(0)

    def source_cell(self, synapse_index=0):
        source = self.source(synapse_index)
        return self.region.cells[source]

    def remove_synapse(self, index):
        pass

    def distance_from(self, coords_xy, index=0):
        source_xy = util.coords_from_index(self.syn_sources[index], self.region._input_side_len())
        return util.distance(source_xy, coords_xy)

    def connected(self, index=0, connectionPermanence=CONNECTED_PERM):
        return self.syn_permanences[index] > connectionPermanence

    def print_type(self):
        return "Proximal" if self.type == self.PROXIMAL else "Distal"

    def contribution(self, index=0, absolute=False):
        '''
        Returns a contribution (activation) [-1.0,1.0] of synapse at index
        '''
        if self.proximal():
            activation = self.region._input_active(self.syn_sources[index])
            mult = 1
        else:
            cell = self.source_cell(index)
            activation = cell.activation
            mult = 1 if cell.excitatory else -1
        if not absolute:
            activation *= mult
        return activation

    def total_activation(self):
        '''
        Returns sum of contributions from all connected synapses
        Return: double (not bounded)
        '''
        return sum([self.contribution(i) for i in self.connected_synapses()])

    def active(self):
        '''
        '''
        threshold = PROXIMAL_ACTIVATION_THRESHHOLD if self.proximal() else DISTAL_ACTIVATION_THRESHOLD
        return self.total_activation() > threshold

    def n_synapses(self):
        return len(self.syn_sources)

    def source(self, index):
        return self.syn_sources[index]

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        '''
        Return array of indexes
        '''
        connected_indexes = [i for i,p in enumerate(self.syn_permanences) if p > connectionPermanence]
        return connected_indexes


class Cell(object):
    '''
    An HTM abstraction of one or more biological neurons
    Has multiple dendrite segments connected to inputs
    '''

    def __init__(self, region, index, n_proximal_segments=PROX_SEGMENTS, n_distal_segments=DISTAL_SEGMENTS):
        self.index = index
        self.region = region
        self.n_proximal_segments = n_proximal_segments
        self.n_distal_segments = n_distal_segments
        self.distal_segments = []
        self.proximal_segments = []
        self.activation = 0.0 # [0.0, 1.0]
        self.coords = util.coords_from_index(index, self.region._cell_side_len())
        self.fade_rate = random.uniform(MIN_FADE_RATE, MAX_FADE_RATE)
        self.excitatory = random.random() > CHANCE_OF_INHIBITORY

        # History
        self.recent_active_duty = []  # After inhibition, list of bool
        self.recent_overlap_duty = []  # Overlap > min_overlap, list of bool

    def __repr__(self):
        return "<Cell index=%d activation=%.1f excitatory=%s />" % (self.index, self.activation, self.excitatory)

    def initialize(self):
        for i in range(self.n_proximal_segments):
            proximal = Segment(self, i, self.region, type=Segment.PROXIMAL)
            proximal.initialize() # Creates synapses
            self.proximal_segments.append(proximal)
        for i in range(self.n_distal_segments):
            distal = Segment(self, i, self.region, type=Segment.DISTAL)
            distal.initialize() # Creates synapses
            self.distal_segments.append(distal)
        log("Initialized %s" % self)

    def active_segments(self, type=Segment.PROXIMAL):
        segs = self.proximal_segments if type == Segment.PROXIMAL else self.distal_segments
        return filter(lambda seg : seg.active(), segs)

    def update_duty_cycles(self, active=False, overlap=False):
        '''
        Add current active state & overlap state to history and recalculate duty cycles
        '''
        self.recent_active_duty.insert(0, 1 if active else 0)
        self.recent_overlap_duty.insert(0, 1 if overlap else 0)
        if len(self.recent_active_duty) > DUTY_HISTORY:
            # Truncate
            self.recent_active_duty = self.recent_active_duty[:DUTY_HISTORY]
        if len(self.recent_overlap_duty) > DUTY_HISTORY:
            # Truncate
            self.recent_overlap_duty = self.recent_overlap_duty[:DUTY_HISTORY]
        cell_active_duty = sum(self.recent_active_duty) / float(len(self.recent_active_duty))
        cell_overlap_duty = sum(self.recent_overlap_duty) / float(len(self.recent_overlap_duty))
        self.region.active_duty_cycle[self.index] = cell_active_duty
        self.region.overlap_duty_cycle[self.index] = cell_overlap_duty

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

    def most_active_distal_segment(self):
        return sorted(self.distal_segments, key=lambda seg : seg.total_activation(), reverse=True)[0]

class Region(object):
    '''
    Made up of many columns
    '''

    def __init__(self, brain, index, permanence_inc=INIT_PERMANENCE_LEARN_INC_CHANGE, permanence_dec=INIT_PERMANENCE_LEARN_DEC_CHANGE, n_cells=10, n_inputs=20):
        self.index = index
        self.brain = brain

        # Region constants (spatial)
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.inhibition_radius = 0 # Average connected receptive field size of the columns

        # Hierarchichal setup
        self.n_cells = n_cells
        self.n_inputs = n_inputs
        self.cells = []

        # State (historical)
        self.input = None  # Inputs at time t - input[t, j] is double [0.0, 1.0]

        # State
        self.overlap = np.zeros(self.n_cells)  # Overlap for each cell. overlap[c] is double
        self.boost = np.ones(self.n_cells, dtype=float)  # Boost value for cell c
        self.bias = np.zeros(self.n_cells)
        self.pre_activation = np.zeros(self.n_cells)
        self.active_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has been active after inhibition (e.g. over the last 1000 iterations).
        self.overlap_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has had significant overlap (> min_overlap)
        self.last_activation = None  # Hold last step in state for rendering

        # Helper constants
        self.diagonal = 1.414*2*math.sqrt(n_cells)


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
            self.cells.append(c)
        print "Initialized %s" % self

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
            cell_values = []
            for c in cells:
                cell_values.append(values[c.index])
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
            b = 1 + (min_duty_cycle - self.active_duty_cycle[c]) * BOOST_MULTIPLIER
        return b

    def _increase_permanences(self, c, increase, excitatory_only=False, type="proximal"):
        '''
        Increase the permanence value of every excitatory synapse (cell c) by a increase factor
        TODO: Should this be for a specific segment?
        '''
        cell = self.cells[c]
        if type == "proximal":
            for seg in self.cells[c].proximal_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if source_cell.excitatory or not excitatory_only:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

        if type == "distal" and BOOST_DISTAL:
            # Try also increasing distal segments
            for seg in self.cells[c].distal_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if source_cell.excitatory or not excitatory_only:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

    def calculate_distal_biases(self):
        '''
        For each cell calculate aggregate activations from distal segments
        Result is a bias array that will be used during overlap to increase
        chances we activate 'predicted' cells
        '''
        bias = np.zeros(len(self.cells))  # Initialize bias to 0
        for i, c in enumerate(self.cells):
            for seg in c.distal_segments:
                if seg.active():
                    bias[i] += 1
        return bias

    def do_overlap(self):
        '''
        Return overlap as a double for each cell representing boosted, biased
        activation from proximal inputs
        '''
        overlaps = np.zeros(len(self.cells))  # Initialize overlaps to 0
        for i, c in enumerate(self.cells):
            for seg in c.proximal_segments:
                if seg.active():
                    overlaps[i] += 1
            overlaps[i] *= self.boost[i]
        return overlaps

    def do_inhibition(self):
        '''
        Inputs:
            - Overlaps (proximal)
            - Bias (distal)
        Sum the two (weighted) and choose winners (active outputs)
        '''
        PREACTIVATION_REQUIRES_PROXIMAL = True
        self.pre_activation = OVERLAP_EFFECT * self.overlap * (1 + DISTAL_BIAS_EFFECT * self.bias)
        active = np.zeros(len(self.cells))
        for c in self.cells:
            pa = self.pre_activation[c.index]
            neighbors = self._neighbors_of(c)
            kth_score = self._kth_score(neighbors, self.pre_activation, k=DESIRED_LOCAL_ACTIVITY)
            if pa > 0 and pa >= kth_score:
                active[c.index] = True
        return active

    def learn_segment(self, seg, is_activating=False, distal=False):
        n_inc = n_dec = n_conn = n_discon = 0
        for i in range(seg.n_synapses()):
            seg.syn_last_change[i] = 0
            was_connected = seg.connected(i)
            cell = seg.source_cell(i)
            cell_excitatory = cell.excitatory
            contributor = seg.contribution(i, absolute=True) >= SYNAPSE_ACTIVATION_LEARN_THRESHHOLD
            this_synapse_learns = True #contributor
            if this_synapse_learns:
                increase_permanence = (is_activating and cell_excitatory and contributor) or (not is_activating and not cell_excitatory and contributor)
                if increase_permanence and seg.syn_permanences[i] < 1.0:
                    n_inc += 1
                    seg.syn_permanences[i] += self.permanence_inc
                    seg.syn_permanences[i] = min(1.0, seg.syn_permanences[i])
                    seg.syn_last_change[i] += 1
                elif not increase_permanence and seg.syn_permanences[i] > 0.0:
                    n_dec +=1
                    seg.syn_permanences[i] -= self.permanence_dec
                    seg.syn_permanences[i] = max(0.0, seg.syn_permanences[i])
                    seg.syn_last_change[i] -= 1
            connection_changed = was_connected != seg.connected(i)
            if connection_changed:
                connected = not was_connected
                if connected:
                    n_conn += 1
                else:
                    n_discon += 1
        return (n_inc, n_dec, n_conn, n_discon)

    def do_learning(self, activating):
        '''
        Update permanences
        On activating cells, increase permenences for each excitatory synapse above a min. contribution
        On non-activating cells, increase permenences for each inhibitory synapse above a min. contribution
        '''
        n_increased_prox = n_decreased_prox = n_increased_dist = n_decreased_dist = n_conn_prox = n_discon_prox = n_conn_dist = n_discon_dist = 0
        for i, is_activating in enumerate(activating):
            cell = self.cells[i]
            # Proximal
            if is_activating:
                for seg in cell.proximal_segments:
                    ni, nd, nc, ndc = self.learn_segment(seg, is_activating=is_activating)
                    n_increased_prox += ni
                    n_decreased_prox += nd
                    n_conn_prox += nc
                    n_discon_prox += ndc
            # Distal
            any_active = any([seg.active() for seg in cell.distal_segments])
            most_active = None
            if not any_active:
                most_active = cell.most_active_distal_segment()
            for seg in cell.distal_segments:
                do_learn = is_activating and (seg.active() or (most_active and seg.index == most_active.index))
                if do_learn:
                    ni, nd, nc, ndc = self.learn_segment(seg, is_activating=is_activating, distal=True)
                    n_increased_dist += ni
                    n_decreased_dist += nd
                    n_conn_dist += nc
                    n_discon_dist += ndc
                else:
                    # Re-initialize change state
                    seg.syn_last_change = [0 for x in seg.syn_last_change]


        log("Distal: +%d/-%d (%d connected, %d disconnected)" % (n_increased_dist, n_decreased_dist, n_conn_dist, n_discon_dist))

        n_boosted = 0
        all_field_sizes = []
        for i, cell in enumerate(self.cells):
            neighbors = self._neighbors_of(cell)
            min_duty_cycle = 0.01 * self._max_duty_cycle(neighbors) # Based on active duty
            cell_active = activating[i]
            sufficient_overlap = self.overlap[i] > self.brain.min_overlap
            cell.update_duty_cycles(active=cell_active, overlap=sufficient_overlap)
            if DO_BOOSTING and self.brain.t > T_START_BOOSTING:
                self.boost[i] = self._boost_function(i, min_duty_cycle)  # Updates boost value for cell (higher if below min)

                # Check if overlap duty cycle less than minimum (note: min is calculated from max *active* not overlap)
                if self.overlap_duty_cycle[i] < min_duty_cycle:
                    # log("Increasing permanences for cell %s in region %d due to overlap duty cycle below min: %s" % (i, self.index, min_duty_cycle))
                    self._increase_permanences(i, 0.1 * CONNECTED_PERM, type="proximal")
                    n_boosted += 1

                # TODO: Boost distal here if active_duty_cycle low?

            all_field_sizes.append(self.cells[i].connected_receptive_field_size())

        if n_boosted:
            log("Boosting %d due to low overlap duty cycle" % n_boosted)

        # Update inhibition radius (based on updated active connections in each column)
        self.inhibition_radius = util.average(all_field_sizes) * INHIBITION_RADIUS_DISCOUNT
        min_positive_radius = 1.0
        if self.inhibition_radius and self.inhibition_radius < min_positive_radius:
            self.inhibition_radius = min_positive_radius
        # log("Setting inhibition radius to %s" % self.inhibition_radius)

    def tempero_spatial_pooling(self, learning_enabled=True):
        '''
        Temporal-Spatial pooling routine
        --------------
        Takes input and calculates active columns (sparse representation) for input into temporal pooling
        '''

        # Phase 1: Calculate Distal Biases (TM?)
        self.bias = self.calculate_distal_biases()
        log("%s << Bias" % printarray(self.bias, continuous=True), level=2)

        # Phase 2: Overlap
        self.overlap = self.do_overlap()
        log("%s << Overlap (normalized)" % printarray(self.overlap, continuous=True), level=2)

        # Phase 3: Inhibition
        activating = self.do_inhibition()
        log("%s << Activating (inhibited)" % printarray(activating, continuous=True), level=2)

        # Phase 4: Learning
        if learning_enabled:
            log("%s << Active Duty Cycle" % printarray(self.active_duty_cycle, continuous=True), level=2)
            log("%s << Overlap Duty Cycle" % printarray(self.overlap_duty_cycle, continuous=True), level=2)
            self.do_learning(activating)

        # Update activations
        self.last_activation = [cell.activation for cell in self.cells]
        for i, cell in enumerate(self.cells):
            if activating[i]:
                cell.activation = 1.0  # Max out
            else:
                cell.activation -= cell.fade_rate
            if cell.activation < 0:
                cell.activation = 0.0

        log("%s << Activations" % self.print_cells(), level=2)

        if VERBOSITY >= 1:
            # Log average synapse permanence in region
            permanences = []
            n_connected = 0
            n_synapses = 0
            for cell in self.cells:
                for seg in cell.distal_segments:
                    permanences.append(util.average(seg.syn_permanences))
                    n_synapses += seg.n_synapses()
                    n_connected += len(filter(lambda x : x > CONNECTED_PERM, seg.syn_permanences))
            ave_permanence = util.average(permanences)
            log("R%d - average distal synapse permanence: %.1f (%.1f%% connected of %d)" % (self.index, ave_permanence, (n_connected/float(n_synapses))*100., n_synapses), level=1)



    ##################
    # Primary Step Function
    ##################

    def step(self, input, learning_enabled=False):
        self.input = input

        self.tempero_spatial_pooling(learning_enabled=learning_enabled)  # Calculates active cells

        return [c.activation for c in self.cells]



class CHTMBrain(object):

    def __init__(self, cells_per_region=None, min_overlap=DEF_MIN_OVERLAP, r1_inputs=1):
        self.regions = []
        self.t = 0
        self.active_behaviors = []
        self.cells_per_region = cells_per_region
        self.n_inputs = r1_inputs
        self.min_overlap = min_overlap # A minimum number of inputs that must be active for a column to be considered during the inhibition step
        self.inputs = None

    def __repr__(self):
        return "<HTMBrain regions=%d>" % len(self.regions)

    def initialize(self):
        n_inputs = self.n_inputs
        for i, cpr in enumerate(self.cells_per_region):
            r = Region(self, i, n_inputs=n_inputs, n_cells=cpr)
            r.initialize()
            n_inputs = cpr  # Next region will have 1 input for each output cell
            self.regions.append(r)
        print "Initialized %s" % self

    def process(self, readings, learning=False):
        '''
        Step through all regions inputting output of each into next
        Returns output of last region
        '''
        print "~~~~~~~~~~~~~~~~~ Processing inputs at T%d" % self.t
        self.inputs = readings
        _in = self.inputs
        for i, r in enumerate(self.regions):
            log("Step processing for region %d\n%s << Input" % (i, printarray(_in, continuous=True)), level=2)
            out = r.step(_in, learning_enabled=learning)
            _in = out
        self.t += 1 # Move time forward one step
        return out
