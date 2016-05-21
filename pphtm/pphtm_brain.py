#!/usr/bin/env python

import numpy as np
import random
import math
import util
from util import printarray

# Settings (global vars, other vars set in brain.__init__)

VERBOSITY = 0
DEF_MIN_OVERLAP = 2
CONNECTED_PERM = 0.2  # If the permanence value for a synapse is greater than this value, it is said to be connected.
DUTY_HISTORY = 100
INHIBITION_RADIUS_DISCOUNT = 0.8
INIT_PERMANENCE = 0.2  # New learned synapses
INIT_PERMANENCE_JITTER = 0.05  # Max offset from CONNECTED_PERM when initializing synapses
SYNAPSE_ACTIVATION_LEARN_THRESHHOLD = 1.0
T_START_BOOSTING = 0

BOOST_DISTAL = False

def log(message, level=1):
    if VERBOSITY >= level:
        print message

class Segment(object):
    '''
    Dendrite segment of cell (proximal, distal, or top-down prediction)
    Store all synapses activations / connectedness in arrays
    '''
    PROXIMAL = 1
    DISTAL = 2
    TOPDOWN = 3 # prediction

    def __init__(self, cell, index, region, type=None):
        self.index = index
        self.region = region
        self.cell = cell
        self.type = type if type else self.PROXIMAL

        # Synapses (all lists below len == # of synapses)
        self.syn_sources = [] # Index of source (either input or cell in region, or above)
        self.syn_permanences = [] # (0,1)
        self.syn_change = [] # -1, 0 1 (after step)
        self.syn_prestep_contribution = [] # (after step)
        self.syn_contribution = [] # (after step)

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
                chance_of_synapse = ((self.region.brain.MAX_PROXIMAL_INIT_SYNAPSE_CHANCE - self.region.brain.MIN_PROXIMAL_INIT_SYNAPSE_CHANCE) * (1 - float(dist)/max_distance)) + self.region.brain.MIN_PROXIMAL_INIT_SYNAPSE_CHANCE
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(source)
        elif self.distal():
            cell_index = self.cell.index
            for index in range(self.region.n_cells):
                if index == cell_index:
                    # Avoid creating synapse with self
                    continue
                chance_of_synapse = self.region.brain.DISTAL_SYNAPSE_CHANCE
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index, permanence=0.15)
        else:
            # Top down connections
            for index in range(self.region.n_cells_above):
                chance_of_synapse = self.region.brain.TOPDOWN_SYNAPSE_CHANCE
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index, permanence=0.15)
        log_message = "Initialized %s" % self
        log(log_message)

    def proximal(self):
        return self.type == self.PROXIMAL

    def distal(self):
        return self.type == self.DISTAL

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

    def remove_synapse(self, index):
        pass

    def decay_permanences(self):
        factor = self.region.brain.SYNAPSE_DECAY
        self.syn_permanences = [max(p - factor, 0.0) for p in self.syn_permanences]

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

    def contribution(self, index=0, absolute=False):
        '''
        index is a cell index (within segment, not full region)
        Returns a contribution (activation) [-1.0,1.0] of synapse at index
        '''
        if self.proximal():
            activation = self.region._input_active(self.syn_sources[index])
            mult = 1
        else:
            # distal or top-down
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
        threshold = self.region.brain.PROXIMAL_ACTIVATION_THRESHHOLD if self.proximal() else self.region.brain.DISTAL_ACTIVATION_THRESHOLD
        return self.total_activation() > threshold

    def n_synapses(self):
        return len(self.syn_sources)

    def source(self, index):
        return self.syn_sources[index]

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        '''
        Return array of cell indexes (in segment)
        '''
        connected_indexes = [i for i,p in enumerate(self.syn_permanences) if p > connectionPermanence]
        return connected_indexes


class Cell(object):
    '''
    An HTM abstraction of one or more biological neurons
    Has multiple dendrite segments connected to inputs
    '''

    def __init__(self, region, index):
        self.index = index
        self.region = region
        self.n_proximal_segments = self.region.brain.PROX_SEGMENTS
        self.n_distal_segments = self.region.brain.DISTAL_SEGMENTS
        self.n_topdown_segments = self.region.brain.TOPDOWN_SEGMENTS if not self.region.is_top() else 0
        self.distal_segments = []
        self.proximal_segments = []
        self.topdown_segments = []
        self.activation = 0.0 # [0.0, 1.0]
        self.coords = util.coords_from_index(index, self.region._cell_side_len())
        self.fade_rate = self.region.brain.FADE_RATE
        self.excitatory = random.random() > self.region.brain.CHANCE_OF_INHIBITORY

        # History
        self.recent_active_duty = []  # After inhibition, list of bool
        self.recent_overlap_duty = []  # Overlap > min_overlap, list of bool

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

    def active_segments(self, type=Segment.PROXIMAL):
        segs = {
            Segment.PROXIMAL: self.proximal_segments,
            Segment.DISTAL: self.distal_segments,
            Segment.TOPDOWN: self.topdown_segments
        }.get(type)
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

    def count_distal_connections(self, cell_index):
        synapses = self.connected_synapses(type=Segment.DISTAL)
        return synapses.count(cell_index)

    def most_active_distal_segment(self):
        return sorted(self.distal_segments, key=lambda seg : seg.total_activation(), reverse=True)[0]

class Region(object):
    '''
    Made up of many columns
    '''

    def __init__(self, brain, index, n_cells=10, n_inputs=20, n_cells_above=0):
        self.index = index
        self.brain = brain

        # Region constants (spatial)
        self.permanence_inc = self.brain.INIT_PERMANENCE_LEARN_INC_CHANGE
        self.permanence_dec = self.brain.INIT_PERMANENCE_LEARN_DEC_CHANGE
        self.inhibition_radius = 0 # Average connected receptive field size of the columns

        # Hierarchichal setup
        self.n_cells = n_cells
        self.n_cells_above = n_cells_above
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
            b = 1 + (min_duty_cycle - self.active_duty_cycle[c]) * self.brain.BOOST_MULTIPLIER
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
                    if (source_cell and source_cell.excitatory) or not excitatory_only:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

        if type == "distal" and BOOST_DISTAL:
            # Try also increasing distal segments
            for seg in self.cells[c].distal_segments:
                for i, perm in enumerate(seg.syn_permanences):
                    source_cell = seg.source_cell(i)
                    if source_cell.excitatory or not excitatory_only:
                        seg.syn_permanences[i] = min([perm+increase, 1.0])

    def calculate_biases(self):
        '''
        For each cell calculate aggregate activations from distal segments
        Result is a bias array that will be used during overlap to increase
        chances we activate 'predicted' cells

        For PP bias array now includes top-down biases
        '''
        bias = np.zeros(len(self.cells))  # Initialize bias to 0
        for i, c in enumerate(self.cells):
            for seg in (c.distal_segments + c.topdown_segments):
                seg.active_before_learning = seg.active()
                if seg.active_before_learning:
                    bias[i] += 1

        return bias

    def do_overlap(self):
        '''
        Return overlap as a double for each cell representing boosted
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
        self.pre_activation = self.brain.OVERLAP_EFFECT * self.overlap * (1 + self.brain.DISTAL_BIAS_EFFECT * self.bias)
        active = np.zeros(len(self.cells))
        for c in self.cells:
            pa = self.pre_activation[c.index]
            neighbors = self._neighbors_of(c)
            kth_score = self._kth_score(neighbors, self.pre_activation, k=self.brain.DESIRED_LOCAL_ACTIVITY)
            if pa > 0 and pa >= kth_score:
                active[c.index] = True
        return active

    def learn_segment(self, seg, is_activating=False, distal=False):
        '''Update synapse permanences for this segment.

        Synapse's permanence will increase if::
          a) it is contributing from an excitatory source, to an activating cell
          b) it is contributing from an inhibitory source, to a non-activating cell

        Otherwise decrease permanence
        '''
        n_inc = n_dec = n_conn = n_discon = 0
        for i in range(seg.n_synapses()):
            seg.syn_change[i] = 0
            was_connected = seg.connected(i)
            cell = seg.source_cell(i)
            cell_excitatory = not cell or cell.excitatory # Inputs excitatory
            seg.syn_prestep_contribution[i] = seg.syn_contribution[i]
            contribution = seg.contribution(i, absolute=True)
            contributor = contribution >= SYNAPSE_ACTIVATION_LEARN_THRESHHOLD
            seg.syn_contribution[i] = contribution
            increase_permanence = (is_activating and cell_excitatory and contributor) or (not is_activating and not cell_excitatory and contributor)
            if increase_permanence and seg.syn_permanences[i] < 1.0:
                n_inc += 1
                seg.syn_permanences[i] += self.permanence_inc
                seg.syn_permanences[i] = min(1.0, seg.syn_permanences[i])
                seg.syn_change[i] += 1
            elif not increase_permanence and seg.syn_permanences[i] > 0.0:
                n_dec +=1
                seg.syn_permanences[i] -= self.permanence_dec
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
        '''Update permanences.

        On activating cells, increase permenences for each excitatory synapse above a min. contribution
        On non-activating cells, increase permenences for each inhibitory synapse above a min. contribution

        # TODO: top-down learning
        '''
        n_increased_prox = n_decreased_prox = n_increased_dist = n_decreased_dist = n_conn_prox = n_discon_prox = n_conn_dist = n_discon_dist = 0
        for i, is_activating in enumerate(activating):
            cell = self.cells[i]
            # Proximal
            # TODO: Also hold pre-learn segment activation for proximal
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
                active = seg.active_before_learning
                do_learn = is_activating and (active or not any_active)
                if do_learn:
                    ni, nd, nc, ndc = self.learn_segment(seg, is_activating=is_activating, distal=True)
                    n_increased_dist += ni
                    n_decreased_dist += nd
                    n_conn_dist += nc
                    n_discon_dist += ndc
                else:
                    # Re-initialize change state
                    seg.decay_permanences()
                    seg.syn_change = [0 for x in seg.syn_change]


        log("Distal: +%d/-%d (%d connected, %d disconnected)" % (n_increased_dist, n_decreased_dist, n_conn_dist, n_discon_dist))

        n_boosted = 0
        all_field_sizes = []
        for i, cell in enumerate(self.cells):
            neighbors = self._neighbors_of(cell)
            min_duty_cycle = 0.01 * self._max_duty_cycle(neighbors) # Based on active duty
            cell_active = activating[i]
            sufficient_overlap = self.overlap[i] > self.brain.min_overlap
            cell.update_duty_cycles(active=cell_active, overlap=sufficient_overlap)
            if self.brain.DO_BOOSTING and self.brain.t > T_START_BOOSTING:
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
        Takes input and calculates active columns (sparse representation)
        This is a representation of input in context of previous input
        (same region and regions above -- top-down predictions)
        '''

        # Phase 1: Overlap
        self.overlap = self.do_overlap()
        log("%s << Overlap (normalized)" % printarray(self.overlap, continuous=True), level=2)

        # Phase 2: Inhibition
        activating = self.do_inhibition()
        log("%s << Activating (inhibited)" % printarray(activating, continuous=True), level=2)

        # Phase 3: Learning
        if learning_enabled:
            log("%s << Active Duty Cycle" % printarray(self.active_duty_cycle, continuous=True), level=2)
            log("%s << Overlap Duty Cycle" % printarray(self.overlap_duty_cycle, continuous=True), level=2)
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
        log("%s << Activations" % self.print_cells(), level=2)


        # Phase 5: Calculate Distal Biases (TM?)
        self.last_bias = np.copy(self.bias)
        self.bias = self.calculate_biases()
        log("%s << Bias" % printarray(self.bias, continuous=True), level=2)

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
        '''Process inputs for a region.

        After step is complete:
            * self.bias is an array of biases based on activations after step
            * self.overlap is an array based on proximal connections to new inputs
            * self.last_activation is an array of activations before new input
            * self.last_bias is an array of biases before new input
            * cell.activation for each cell is updated
        '''
        self.input = input

        self.tempero_spatial_pooling(learning_enabled=learning_enabled)  # Calculates active cells

        return [c.activation for c in self.cells]



class PPHTMBrain(object):
    '''
    Predictive Processing implementation of HTM.
    '''

    def __init__(self, min_overlap=DEF_MIN_OVERLAP, r1_inputs=1):
        self.regions = []
        self.t = 0
        self.active_behaviors = []
        self.inputs = None

        # Brain config
        self.n_inputs = r1_inputs
        self.min_overlap = min_overlap # A minimum number of inputs that must be active for a column to be considered during the inhibition step
        # Defaults
        self.PROXIMAL_ACTIVATION_THRESHHOLD = 3 # Activation threshold for a segment. If the number of active connected synapses in a segment is greater than activationThreshold, the segment is said to be active.
        self.DISTAL_ACTIVATION_THRESHOLD = 2
        self.BOOST_MULTIPLIER = 1.3
        self.DESIRED_LOCAL_ACTIVITY = 2
        self.DO_BOOSTING = 1
        self.DISTAL_SYNAPSE_CHANCE = 0.5
        self.TOPDOWN_SYNAPSE_CHANCE = 0.4
        self.MAX_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.4
        self.MIN_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.1
        self.CELLS_PER_REGION = 9**2
        self.N_REGIONS = 1
        self.DISTAL_BIAS_EFFECT = 0.6
        self.OVERLAP_EFFECT = 0.6
        self.FADE_RATE = 0.7
        self.DISTAL_SEGMENTS = 3
        self.PROX_SEGMENTS = 2
        self.TOPDOWN_SEGMENTS = 1 # Only relevant if >1 region
        self.SYNAPSE_DECAY = 0.0005
        self.INIT_PERMANENCE_LEARN_INC_CHANGE = 0.03
        self.INIT_PERMANENCE_LEARN_DEC_CHANGE = 0.003
        self.CHANCE_OF_INHIBITORY = 0.2

    def __repr__(self):
        return "<PPHTMBrain regions=%d>" % len(self.regions)

    def initialize(self, n_inputs=None, **params):
        if 'PROXIMAL_ACTIVATION_THRESHHOLD' in params:
            self.PROXIMAL_ACTIVATION_THRESHHOLD = params.get('PROXIMAL_ACTIVATION_THRESHHOLD')
        if 'DISTAL_ACTIVATION_THRESHOLD' in params:
            self.DISTAL_ACTIVATION_THRESHOLD = params.get('DISTAL_ACTIVATION_THRESHOLD')
        if 'BOOST_MULTIPLIER' in params:
            self.BOOST_MULTIPLIER = params.get('BOOST_MULTIPLIER')
        if 'DESIRED_LOCAL_ACTIVITY' in params:
            self.DESIRED_LOCAL_ACTIVITY = params.get('DESIRED_LOCAL_ACTIVITY')
        if 'DO_BOOSTING' in params:
            self.DO_BOOSTING = params.get('DO_BOOSTING')
        if 'DISTAL_SYNAPSE_CHANCE' in params:
            self.DISTAL_SYNAPSE_CHANCE = params.get('DISTAL_SYNAPSE_CHANCE')
        if 'TOPDOWN_SYNAPSE_CHANCE' in params:
            self.TOPDOWN_SYNAPSE_CHANCE = params.get('TOPDOWN_SYNAPSE_CHANCE')
        if 'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE' in params:
            self.MAX_PROXIMAL_INIT_SYNAPSE_CHANCE = params.get('MAX_PROXIMAL_INIT_SYNAPSE_CHANCE')
        if 'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE' in params:
            self.MIN_PROXIMAL_INIT_SYNAPSE_CHANCE = params.get('MIN_PROXIMAL_INIT_SYNAPSE_CHANCE')
        if 'CELLS_PER_REGION' in params:
            self.CELLS_PER_REGION = params.get('CELLS_PER_REGION')
        if 'N_REGIONS' in params:
            self.N_REGIONS = params.get('N_REGIONS')
        if 'DISTAL_BIAS_EFFECT' in params:
    	   self.DISTAL_BIAS_EFFECT = params.get('DISTAL_BIAS_EFFECT')
        if 'OVERLAP_EFFECT' in params:
    	   self.OVERLAP_EFFECT = params.get('OVERLAP_EFFECT')
        if 'FADE_RATE' in params:
    	   self.FADE_RATE = params.get('FADE_RATE')
        if 'DISTAL_SEGMENTS' in params:
    	   self.DISTAL_SEGMENTS = params.get('DISTAL_SEGMENTS')
        if 'PROX_SEGMENTS' in params:
    	   self.PROX_SEGMENTS = params.get('PROX_SEGMENTS')
        if 'TOPDOWN_SEGMENTS' in params:
            self.TOPDOWN_SEGMENTS = params.get('TOPDOWN_SEGMENTS')
        if 'SYNAPSE_DECAY' in params:
        	self.SYNAPSE_DECAY = params.get('SYNAPSE_DECAY')
        if 'INIT_PERMANENCE_LEARN_INC_CHANGE' in params:
        	self.INIT_PERMANENCE_LEARN_INC_CHANGE = params.get('INIT_PERMANENCE_LEARN_INC_CHANGE')
        if 'INIT_PERMANENCE_LEARN_DEC_CHANGE' in params:
        	self.INIT_PERMANENCE_LEARN_DEC_CHANGE = params.get('INIT_PERMANENCE_LEARN_DEC_CHANGE')
        if 'CHANCE_OF_INHIBITORY' in params:
            self.CHANCE_OF_INHIBITORY = params.get('CHANCE_OF_INHIBITORY')

        if n_inputs is not None:
            self.n_inputs = n_inputs

        n_inputs = self.n_inputs

        # Initialize and create regions and cells
        self.regions = []
        for i in range(self.N_REGIONS):
            top_region = i == self.N_REGIONS - 1
            cpr = self.CELLS_PER_REGION
            r = Region(self, i, n_inputs=n_inputs, n_cells=cpr, n_cells_above=cpr if not top_region else 0)
            r.initialize()
            n_inputs = cpr  # Next region will have 1 input for each output cell
            self.regions.append(r)
        self.t = 0
        log("Initialized %s" % self)

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
