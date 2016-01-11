#!/usr/bin/env python

# Jeremy Gordon

# Basic implementation of HTM network (Neumenta, Hawkins)
# Adjusted to support continuous activations that fall off over time at different rates
# Temporal pooler removed in favor of spatial-temperal pooling
# TODO: Should we support distal dendrites in same region?
# http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf

import numpy as np
import random
import math
import util

# Settings

VERBOSITY = 1
DEF_ACTIVATION_THRESHHOLD = 1 # Activation threshold for a segment. If the number of active connected synapses in a segment is greater than activationThreshold, the segment is said to be active. 
DEF_MIN_OVERLAP = 3
CONNECTED_PERM = 0.2  # If the permanence value for a synapse is greater than this value, it is said to be connected.
DUTY_HISTORY = 1000
BOOST_MULTIPLIER = 3
OVERLAP_ACTIVATION_THRESHHOLD = 3
INHIBITION_RADIUS_DISCOUNT = 0.4
INIT_PERMANENCE = 0.2  # New learned synapses
INIT_PERMANENCE_JITTER = 0.05  # Max offset from CONNECTED_PERM when initializing synapses
SYNAPSE_ACTIVATION_LEARN_THRESHHOLD = 0.5
INIT_PERMANENCE_LEARN_CHANGE = 0.01
INIT_DESIRED_LOCAL_ACTIVITY = 3
DO_BOOSTING = True
CHANCE_OF_INHIBITORY = 0.3

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
                simplified = str(int(item * 10))
                if simplified == "10":
                    simplified = "X"
                elif simplified == "0":
                    simplified = "."
            out += simplified
        return out
    else:
        if type(array[0]) is int or coerce_to_int:
            return ''.join([str(int(x)) for x in array])
        elif type(array[0]) in [float, np.float64]:
            return '|'.join([str(int(x)) for x in array])

class Synapse(object):
    '''
    A synapse part of a dendrite segment of a cell
    Passes activation of source to dendrite segment
    If inhibitory, subtracts from activation
    Connected if permanence > threshhold
    '''

    def __init__(self, region, source, permanence, column=None):
        self.region = region
        self.permanence = permanence  # (0,1) Only used for proximal
        self.excitatory = random.random() > CHANCE_OF_INHIBITORY # Inhibitory if False
        # Source
        self.source = source  # Index of source input

    def distance_from(self, coords_xy):
        source_xy = util.coords_from_index(self.source, self.region._input_side_len())
        return util.distance(source_xy, coords_xy)

    def connected(self, connectionPermanence=CONNECTED_PERM):
        return self.permanence > connectionPermanence

    def contribution(self, tMinus=0, absolute=False):
        '''
        Returns a contribution [-1.0,1.0]
        '''
        activation = self.region._get_activation(self.source, tMinus)
        mult = 1 if self.excitatory else -1
        if not absolute:
            activation *= mult
        return activation


class Segment(object):
    '''
    Dendrite segment of cell
    '''
    def __init__(self, cell, index, region):
        self.index = index
        self.region = region
        self.activation_threshhold = DEF_ACTIVATION_THRESHHOLD
        self.cell = cell

        # State 
        self.potential_synapses = []  # List of Synapse() objects

    def __repr__(self):
        t = self.region.brain.t
        return "<Segment index=%d potential=%d connected=%d>" % (self.index, len(self.potential_synapses), len(self.connected_synapses()))

    def initialize(self):
        # Setup initial potential synapses for proximal segments
        MAX_INIT_SYNAPSE_CHANCE = 0.5
        MIN_INIT_SYNAPSE_CHANCE = 0.05
        n_inputs = self.region.n_inputs
        cell_x, cell_y = util.coords_from_index(self.cell.index, self.region._cell_side_len())
        for source in range(n_inputs):
            input_x, input_y = util.coords_from_index(source, self.region._input_side_len())
            dist = util.distance((cell_x, cell_y), (input_x, input_y))
            chance_of_synapse = (MAX_INIT_SYNAPSE_CHANCE * (1 - float(dist)/self.region._input_side_len())) + MIN_INIT_SYNAPSE_CHANCE
            add_synapse = random.random() < chance_of_synapse
            if add_synapse:
                perm = CONNECTED_PERM + INIT_PERMANENCE_JITTER*(random.random()-0.5)
                s = Synapse(self.region, source, perm)
                self.potential_synapses.append(s)
        log_message = "Initialized %s" % self
        log(log_message)

    def total_activation(self):
        '''
        Returns sum of contributions, double (not bounded)
        '''
        return sum([syn.contribution() for syn in self.connected_synapses()])

    def active(self, tMinus, state='active'):  # state in ['active','learn']
        '''
        '''
        return self.total_activation() > self.activation_threshhold

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        return filter(lambda s : s.connected(connectionPermanence=connectionPermanence), self.potential_synapses)


class Cell(object):
    '''
    An HTM abstraction of one or more biological neurons
    Has multiple dendrite segments connected to inputs
    '''

    def __init__(self, region, index, n_segments=5):
        self.index = index
        self.region = region
        self.n_segments = n_segments
        self.segments = []
        self.activation = 0.0 # [0.0, 1.0]
        self.coords = util.coords_from_index(index, self.region._cell_side_len())
        self.fade_rate = random.uniform(0.1, 0.3)

        # History
        self.recent_active_duty = []  # After inhibition, list of bool
        self.recent_overlap_duty = []  # Overlap > min_overlap, list of bool

    def __repr__(self):
        return "<Cell index=%d activation=%.1f />" % (self.index, self.activation)

    def initialize(self):
        for i in range(self.n_segments):
            seg = Segment(self, i, self.region)
            seg.initialize() # Creates synapses
            self.segments.append(seg)

    def active_segments(self):
        return filter(lambda seg : seg.active(), self.segments)

    def get_active_segment(self, state='active'):
        # Get active segments sorted by sequence boolean, then activity level
        ordered = sorted(self.active_segments(state=state), key=lambda seg : seg.total_activation())
        if ordered:
            return ordered[0]
        else:
            return None

    def adapt_segments(self, positive_reinforcement=True):
        log("Adapting segments for %s - %d segment updates" % (self, len(self.segment_update_list)), level=4)
        # TODO: Impl

    def update_duty_cycles(self, active=False, overlap=False):
        '''
        Add current active state & overlap state to history and recalculate duty cycles
        '''
        self.recent_active_duty.insert(0, active)
        self.recent_overlap_duty.insert(0, overlap)
        if len(self.recent_active_duty) > DUTY_HISTORY:
            # Truncate
            self.recent_active_duty = self.recent_active_duty[:DUTY_HISTORY]  
        if len(self.recent_overlap_duty) > DUTY_HISTORY:
            # Truncate
            self.recent_overlap_duty = self.recent_overlap_duty[:DUTY_HISTORY]  
        x = sum(self.recent_active_duty) / len(self.recent_active_duty)
        self.region.active_duty_cycle[self.index] = x
        self.region.overlap_duty_cycle[self.index] = sum(self.recent_overlap_duty) / len(self.recent_overlap_duty)        

    def connected_receptive_field_size(self):
        '''
        Returns max distance (radius) among currently connected synapses
        '''
        connected = self.connected_synapses()
        distances = [s.distance_from(self.coords) for s in connected]
        if distances:
            return max(distances)
        else:
            return 0

    def connected_synapses(self):
        synapses = []
        for seg in self.segments:
            synapses.extend(seg.connected_synapses())
        return synapses


class Region(object):
    '''
    Made up of many columns
    '''

    def __init__(self, brain, index, desired_local_activity=INIT_DESIRED_LOCAL_ACTIVITY, permanence_inc=INIT_PERMANENCE_LEARN_CHANGE, permanence_dec=INIT_PERMANENCE_LEARN_CHANGE, n_cells=10, n_inputs=20):
        self.index = index
        self.brain = brain

        # Region constants (spatial)
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.desired_local_activity = desired_local_activity  # Desired winners after input step
        self.inhibition_radius = 0 # Average connected receptive field size of the columns

        # Hierarchichal setup
        self.n_cells = n_cells
        self.n_inputs = n_inputs
        self.cells = []

        # State (historical)
        self.input = np.zeros((0, self.n_inputs))  # Inputs at time t - input[t, j] is double [0.0, 1.0]

        # State
        self.overlap = np.zeros(self.n_cells)  # Overlap for each cell. overlap[c] is double
        self.boost = np.ones(self.n_cells, dtype=float)  # Boost value for cell c
        self.active_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has been active after inhibition (e.g. over the last 1000 iterations).
        self.overlap_duty_cycle = np.zeros((self.n_cells))  # Sliding average: how often column c has had significant overlap (> min_overlap)
        # self.min_duty_cycle = np.zeros((self.n_cells))  # Minimum desired firing rate for a cell. If a cell's firing rate falls below this value, it will be boosted. This value is calculated as 1% of the maximum firing rate of its neighbors.
        
    def __str__(self):
        return "<Region inputs=%d cells=%d />" % (self.n_inputs, len(self.cells))

    def print_cells(self):
        activations = [cell.activation for cell in self.cells]
        return printarray(activations, continuous=True)

    def initialize(self):
        # Create cells & get neighbors
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

    def _get_activation(self, j, tMinus=0):
        '''
        At index j, at time t - tMinus
        tMinus 0 is now, positive is steps back in time
        TODO: We can speed this up by keeping a shorter history
        '''
        present = self.input.shape[0] - 1  # Number of time step rows in input ndarray
        t = present - tMinus
        if t >= 0:
            return self.input[t][j]
        else:
            raise Exception("History error - cant get input at T%d" % t)

    def _kth_score(self, cells, k):
        '''
        Given list of cells, calculate kth highest overlap value
        '''
        if cells:
            overlaps = []
            for c in cells:
                overlaps.append(self.overlap[c.index])
            _k = min([k, len(overlaps)]) # TODO: Ok to pick last if k > overlaps?
            overlaps = sorted(overlaps, reverse=True) # Highest to lowest
            kth_overlap = overlaps[_k-1]            
            # log("%s is %dth highest overlap score in sequence: %s" % (kth_overlap, k, overlaps))
            return kth_overlap
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
        # log("Got %d neighbors of %s - inhibition radius: %s" % (len(_neighbors), cell, self.inhibition_radius))
        return _neighbors

    def _boost_function(self, c, min_duty_cycle):
        if self.active_duty_cycle[c] >= min_duty_cycle:
            b = 1.0
        else:
            b = 1 + (min_duty_cycle - self.active_duty_cycle[c]) * BOOST_MULTIPLIER
        return b

    def _increase_permanences(self, c, increase):
        '''
        Increase the permanence value of every synapse (cell c) by a increase factor
        TODO: Should this be for a specific segment?
        '''
        for seg in self.cells[c].segments:
            for syn in seg.potential_synapses:
                syn.permanence += increase
                syn.permanence = min([1.0, syn.permanence])


    ##################
    # Spatial Pooling
    ##################

    def do_overlap(self):
        '''
        Return overlap as a double for each cell representing boosted activation from inputs
        '''
        overlaps = np.zeros(len(self.cells))  # Initialize overlaps to 0
        for i, c in enumerate(self.cells):
            for seg in c.segments:
                overlaps[i] += seg.total_activation()
            # Do floor, boosting, and digitize
            if overlaps[i] < self.brain.min_overlap:
                overlaps[i] = 0
            else:
                boosted = overlaps[i] * self.boost[i]
        return overlaps

    def do_inhibition(self):
        '''
        Get active cells after inhibition around strongly overlapped cells
        '''
        active = np.zeros(len(self.cells))
        for c in self.cells:
            kth_highest_overlap = self._kth_score(self._neighbors_of(c), self.desired_local_activity)
            minLocalActivity = kth_highest_overlap
            ovlp = self.overlap[c.index]
            if minLocalActivity > 0 and ovlp >= minLocalActivity:
                # log("Activating %d because overlap %s > %s" % (c.index, ovlp, minLocalActivity))
                active[c.index] = 1
        return active

    def do_learning(self, activating):
        n_increased = n_decreased = n_changed = 0
        for i, is_activating in enumerate(activating):
            if is_activating:
                cell = self.cells[i]
                for seg in cell.segments:
                    for syn in seg.potential_synapses:
                        was_connected = syn.connected()
                        if syn.contribution(absolute=True) >= SYNAPSE_ACTIVATION_LEARN_THRESHHOLD:
                            n_increased += 1
                            syn.permanence += self.permanence_inc
                            syn.permanence = min(1.0, syn.permanence)
                        else:
                            n_decreased +=1 
                            syn.permanence -= self.permanence_dec
                            syn.permanence = max(0.0, syn.permanence)
                        connection_changed = was_connected != syn.connected()
                        if connection_changed:
                            n_changed += 1

        log("Increased %d and decreased %d permanences. %d changes" % (n_increased, n_decreased, n_changed))

        n_boosted = 0
        all_field_sizes = []
        for i, cell in enumerate(self.cells):
            neighbors = self._neighbors_of(cell)
            min_duty_cycle = 0.01 * self._max_duty_cycle(neighbors) # Based on active duty
            cell_active = activating[i]
            sufficient_overlap = self.overlap[i] > self.brain.min_overlap
            cell.update_duty_cycles(active=cell_active, overlap=sufficient_overlap)
            if DO_BOOSTING:
                self.boost[i] = self._boost_function(i, min_duty_cycle)  # Updates boost value for cell (higher if below min)

                # Check if overlap duty cycle less than minimum (note: min is calculated from max *active* not overlap)
                if self.overlap_duty_cycle[i] < min_duty_cycle:
                    log("Increasing permanences for cell %s in region %d due to overlap duty cycle below min: %s" % (i, self.index, min_duty_cycle))
                    self._increase_permanences(i, 0.1 * CONNECTED_PERM)
                    n_boosted += 1

            all_field_sizes.append(self.cells[i].connected_receptive_field_size())
    
        if n_boosted:
            log("Boosting %d due to low duty cycle" % n_boosted)

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
        # Phase 1: Overlap
        self.overlap = self.do_overlap()
        log("%s << Overlap (normalized)" % printarray(self.overlap, continuous=True), level=2)

        # Phase 2: Inhibition    
        activating = self.do_inhibition()
        log("%s << Activating (inhibited)" % printarray(activating), level=2)

        # Update activations
        for i, cell in enumerate(self.cells):
            if activating[i]:
                cell.activation = 1.0  # Max out
            else:
                cell.activation -= cell.fade_rate
            if cell.activation < 0:
                cell.activation = 0.0

        log("%s << Activations" % self.print_cells(), level=2)

        permanences = []
        for cell in self.cells:
            for seg in cell.segments:
                for syn in seg.potential_synapses:
                    permanences.append(syn.permanence)

        ave_permanence = util.average(permanences)
        n_connected = len(filter(lambda x : x > CONNECTED_PERM, permanences))
        log("Average synapse permanence: %.1f (%d connected of %d)" % (ave_permanence, n_connected, len(permanences)))

        # Phase 3: Learning
        if learning_enabled:
            self.do_learning(activating)


    ##################
    # Primary Step Function
    ##################

    def step(self, input, learning_enabled=False):
        # Add input for time t to input historical state
        self.input = np.vstack((self.input, input))
        
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
        _in = readings
        for i, r in enumerate(self.regions):
            log("Step processing for region %d\n%s << Input" % (i, printarray(_in, continuous=True)), level=2)
            out = r.step(_in, learning_enabled=learning)
            _in = out
        self.t += 1 # Move time forward one step
        return out
