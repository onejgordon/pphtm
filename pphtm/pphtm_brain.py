#!/usr/bin/env python

import numpy as np
import random
import math
from pphtm import util
from pphtm.util import printarray

# Settings (global vars, other vars set in brain.__init__)

VERBOSITY = 1
DEF_MIN_OVERLAP = 2
# If the permanence value for a synapse is greater than this value, it is
# said to be connected.
CONNECTED_PERM = 0.2


INIT_PERMANENCE = 0.2  # New learned synapses
# Max offset from CONNECTED_PERM when initializing synapses
INIT_PERMANENCE_JITTER = 0.05
PROXIMITY_WEIGHTING = 1  # (bool) For proximal connection init.

# Continuous timing
TIME_CONSTANT = 1  # Distance traveled per step
SPIKE_ACTIVATION_INCR = 0.25
CELL_ACTIVATION_THRESH = 5
AVE_SYN_DIST_DISTAL = 2.5
AVE_SYN_DIST_TOPDOWN = 3.5
AVE_SYN_DIST_PROX = 1.5
LEARN_SPIKE_DIST = TIME_CONSTANT  # Radius within which to learn
LEARN_DIST_CHANGE = .05
AXON_MIN_LEN = 1
AXON_MAX_LEN = 4
POST_SPIKE_RESET_ACTIVATION = 0.0
EXPIRE_SPIKE_DISTANCE = max([
    AVE_SYN_DIST_PROX,
    AVE_SYN_DIST_DISTAL,
    AVE_SYN_DIST_TOPDOWN])*1.5  # After which all spikes are removed (max dendrite length?)


def log(message, level=1):
    if VERBOSITY >= level:
        print message


class Spike(object):

    '''
    Represent an action potential spike initiated at a particular
    time for a particular cell.

    Dendrites sourced from this cell will periodically check
    whether these spikes have reached their cell body,
    at which point the activation of the down-stream cell is incremented.

    '''

    def __init__(self, t):
        self.t_init = t  # Time step

    def __repr__(self):
        return "<Spike term=%d />" % self.t_term

    def distance_travelled(self, t_now):
        '''
        How far spike has traveled from cell body
        '''
        return (t_now - self.t_init) * TIME_CONSTANT


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

    Unmodeled: Speed through each dendrite (time constant varies with diameter, etc)
    '''
    PROXIMAL = 1
    DISTAL = 2
    TOPDOWN = 3  # feedback

    def __init__(self, cell, index, region, type=None):
        self.index = index  # Segment index
        self.region = region
        self.cell = cell
        self.type = type if type else self.PROXIMAL

        # Synapses (all lists below len == # of synapses)
        # Index of source (either input or cell in region, or above)
        self.syn_sources = []
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
            cell_x, cell_y = util.coords_from_index(
                self.cell.index, self.region._cell_side_len())
            for source in range(n_inputs):
                # Loop through all inputs and randomly choose to create synapse
                # or not
                input_x, input_y = util.coords_from_index(
                    source, self.region._input_side_len())
                dist = util.distance((cell_x, cell_y), (input_x, input_y))
                max_distance = self.region.diagonal
                chance_of_synapse = ((self.region.brain.config("MAX_PROXIMAL_INIT_SYNAPSE_CHANCE") - self.region.brain.config(
                    "MIN_PROXIMAL_INIT_SYNAPSE_CHANCE")) * (1 - float(dist)/max_distance)) + self.region.brain.config("MIN_PROXIMAL_INIT_SYNAPSE_CHANCE")
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(source)
        elif self.distal():
            cell_index = self.cell.index
            for index in range(self.region.n_cells):
                if index == cell_index:
                    # Avoid creating synapse with self
                    continue
                chance_of_synapse = self.region.brain.config(
                    "DISTAL_SYNAPSE_CHANCE")
                add_synapse = random.random() < chance_of_synapse
                if add_synapse:
                    self.add_synapse(index)
        else:
            # Top down connections
            for index in range(self.region.n_cells_above):
                chance_of_synapse = self.region.brain.config(
                    "TOPDOWN_SYNAPSE_CHANCE")
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

    def add_synapse(self, source_index=0, permanence=None, distance=None):
        if permanence is None:
            permanence = util.jitter(CONNECTED_PERM, INIT_PERMANENCE_JITTER/2)
        if distance is None:
            if self.proximal():
                ave = AVE_SYN_DIST_PROX
            elif self.distal():
                ave = AVE_SYN_DIST_DISTAL
            elif self.topdown():
                ave = AVE_SYN_DIST_TOPDOWN
            distance = util.jitter(ave)
        self.syn_sources.append(source_index)
        self.syn_permanences.append(permanence)
        self.syn_distances.append(distance)
        self.syn_change.append(0)
        self.syn_prestep_contribution.append(0)
        self.syn_contribution.append(0)
        log("Adding synapse at index %d, dist: %s" % (source_index, distance))

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
            Cell() object in own region or region above/below
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
        source_xy = util.coords_from_index(
            self.syn_sources[index], self.region._input_side_len())
        return util.distance(source_xy, coords_xy)

    def connected(self, index=0, connectionPermanence=CONNECTED_PERM):
        return self.syn_permanences[index] > connectionPermanence

    def print_type(self):
        return {
            self.PROXIMAL: "Proximal",
            self.DISTAL: "Distal",
            self.TOPDOWN: "Top-Down"
        }.get(self.type)

    def n_synapses(self):
        return len(self.syn_sources)

    def source(self, index):
        return self.syn_sources[index]

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        '''
        Return array of cell indexes (in segment)
        '''
        connected_indexes = [
            i for i, p in enumerate(self.syn_permanences) if p >= connectionPermanence]
        return connected_indexes

    def nearest_spike(self, synapse_index):
        '''
        Return relative distance to nearest (to synapse_index's distance/location)
        active spike on this segment.
        Positive: nearest spike is farther from cell body
        '''
        min_rel_dist = None
        distance = self.syn_distances[synapse_index]
        for j in range(self.n_synapses()):
            if j == synapse_index:
                continue  # Skip target synapse
            cell = self.source_cell(j)
            if cell:
                comp_dist = self.syn_distances[j]
                dist_between_synapses = comp_dist - distance
                # Positive = comparison synapse farther
                # Negative = comparison synapse closer, find most recent spike
                # (closest to 0 distance travelled)
                near_spike, post_synapse_dist = cell.nearest_postsynaptic_spike_to(
                    dist_between_synapses)
                if near_spike:
                    rel_dist = dist_between_synapses - post_synapse_dist
                    if min_rel_dist is None or rel_dist < min_rel_dist:
                        min_rel_dist = rel_dist
        return min_rel_dist


class Cell(object):

    '''
    Has multiple dendrite segments connected to inputs

    Stores a list of active spikes, which will move along any segments
    sourced from this cell. We assume the full distance is captured by
    the dendrite (we don't represent cell axons). Synapses can occur
    at any location along a dendrite segment, which represents the initial
    distance of the spike to the post-synaptic cell body.
    '''

    def __init__(self, region, index):
        self.index = index
        self.region = region
        self.n_proximal_segments = self.region.brain.config("PROX_SEGMENTS")
        self.n_distal_segments = self.region.brain.config("DISTAL_SEGMENTS")
        self.n_topdown_segments = self.region.brain.config(
            "TOPDOWN_SEGMENTS") if not self.region.is_top() else 0
        self.distal_segments = []
        self.proximal_segments = []
        self.topdown_segments = []
        self.spikes = []  # Active spikes
        self.activation = 0.0  # [0.0, 1.0]
        self.spiking = False
        self.last_spike_t = None  # Time step
        self.axon_length = random.random() * (AXON_MAX_LEN - AXON_MIN_LEN) + \
            AXON_MIN_LEN
        self.coords = util.coords_from_index(
            index, self.region._cell_side_len())
        self.fade_rate = self.region.brain.config("FADE_RATE")
        self.excitatory = random.random() > self.region.brain.config(
            "CHANCE_OF_INHIBITORY")

    def __repr__(self):
        return "<Cell index=%d activation=%.1f spiking=%s spikes=%d />" % (self.index, self.activation, self.region.spiking[self.index], len(self.spikes))

    def initialize(self):
        for i in range(self.n_proximal_segments):
            proximal = Segment(self, i, self.region, type=Segment.PROXIMAL)
            proximal.initialize()  # Creates synapses
            self.proximal_segments.append(proximal)
        for i in range(self.n_distal_segments):
            distal = Segment(self, i, self.region, type=Segment.DISTAL)
            distal.initialize()  # Creates synapses
            self.distal_segments.append(distal)
        for i in range(self.n_topdown_segments):
            topdown = Segment(self, i, self.region, type=Segment.TOPDOWN)
            topdown.initialize()  # Creates synapses
            self.topdown_segments.append(topdown)
        log("Initialized %s" % self)

    def all_segments(self):
        return self.proximal_segments + self.distal_segments + self.topdown_segments

    def terminating_spikes(self, segment_distance):
        '''
        Return Spike objects that should terminate now
        '''
        terminating = []
        n_spikes = len(self.spikes)
        for i, spike in enumerate(reversed(self.spikes)):
            spike_distance = spike.distance_travelled(self.region.brain.t)
            distance = segment_distance + self.axon_length
            # Within 1-step window
            dmin, dmax = distance, distance + TIME_CONSTANT
            if spike_distance >= dmin and spike_distance < dmax:
                # Arrival of spike
                terminating.append(spike)
            elif spike_distance >= EXPIRE_SPIKE_DISTANCE:
                # All remaining spikes are expired
                self.spikes = self.spikes[n_spikes - i:]
        return terminating

    def postsynaptic_distance_travelled(self, spike):
        return max([0, spike.distance_travelled(self.region.brain.t) - self.axon_length])

    def nearest_postsynaptic_spike_to(self, distance):
        '''Find the spike that has, at present, travelled a distance
        nearest to 'distance' after the synapse at end of axon.

        Args:
            distance (float): target distance travelled (post synapse)

        Returns:
            tuple (2):
                Spike: nearest spike
                float: distance from synapse for nearest spike

        '''
        nearest_spike_post_synapse_dist = None
        nearest_spike = None
        nearest_spike_delta = None
        for spike in self.spikes:
            dist_travelled = spike.distance_travelled(self.region.brain.t)
            post_synapse_dist = dist_travelled - self.axon_length
            if post_synapse_dist > 0:
                delta = abs(post_synapse_dist - distance)
                if not nearest_spike_delta or delta < nearest_spike_delta:
                    nearest_spike_delta = delta
                    nearest_spike_post_synapse_dist = post_synapse_dist
                    nearest_spike = spike
        return (nearest_spike, nearest_spike_post_synapse_dist)

    def active_segments(self, type=Segment.PROXIMAL):
        segs = {
            Segment.PROXIMAL: self.proximal_segments,
            Segment.DISTAL: self.distal_segments,
            Segment.TOPDOWN: self.topdown_segments
        }.get(type)
        return filter(lambda seg: seg.active(), segs)

    def connected_receptive_field_size(self):
        '''
        Returns max distance (radius) among currently connected proximal synapses
        '''
        connected_indexes = self.connected_synapses(type=Segment.PROXIMAL)
        side_len = self.region._input_side_len()
        distances = []
        for i in connected_indexes:
            distance = util.distance(
                util.coords_from_index(i, side_len), self.coords)
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

    def spike(self, t):
        log("%s is spiking" % self)
        self.spikes.append(Spike(t))
        self.spiking = True
        self.activation = POST_SPIKE_RESET_ACTIVATION
        self.last_spike_t = t


class Region(object):

    '''
    Made up of many columns
    '''

    def __init__(self, brain, index, n_cells=10, n_inputs=20, n_cells_above=0):
        self.index = index
        self.brain = brain

        # Region constants (spatial)
        self.inhibition_radius = 2

        # Hierarchichal setup
        self.n_cells = n_cells
        self.n_cells_above = n_cells_above
        self.n_inputs = n_inputs
        self.cells = []

        # State (historical)
        # Inputs at time t - input[t, j] is double [0.0, 1.0]
        self.input = None

        # State
        # Overlap for each cell. overlap[c] is double
        self.overlap = np.zeros(self.n_cells)
        self.pre_bias = np.zeros(self.n_cells)  # Prestep bias
        # Bias for each cell. overlap[c] is double
        self.bias = np.zeros(self.n_cells)
        # Activation before inhibition for each cell.
        self.pre_activation = np.zeros(self.n_cells)
        # Each cell that's spiking in this time step
        self.spiking = np.zeros(self.n_cells)
        # Boost value for cell c
        self.boost = np.ones(self.n_cells, dtype=float)
        self.bias = np.zeros(self.n_cells)
        self.last_activation = None  # Hold last step in state for rendering

        # Helpers
        self.diagonal = 1.414*2*math.sqrt(n_cells)
        # Matrix of 1 or -1 for each cell (after initialization)
        self.excitatory_activation_mult = np.ones(self.n_cells)

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
            # TODO: Ok to pick last if k > overlaps?
            _k = min([k, len(values)])
            # Highest to lowest
            cell_values = sorted(cell_values, reverse=True)
            kth_value = cell_values[_k-1]
            # log("%s is %dth highest overlap score in sequence: %s" % (kth_value, k, overlaps))
            return kth_value
        return 0  # Shouldn't happen?

    def _neighbors_of(self, cell):
        '''
        Return all cells within inhibition radius
        '''
        _neighbors = []
        for c in self.cells:
            if cell.index == c.index:
                continue
            dist = util.dist_from_indexes(
                c.index, cell.index, self._cell_side_len())
            if dist <= self.inhibition_radius:
                _neighbors.append(c)
        # log("Got %d neighbors. Overlaps: %s" % (len(_neighbors), [self.pre_activation[n.index] for n in _neighbors]))
        return _neighbors

    def do_terminate_spikes(self):
        '''
        Check each cell for segments connected to sources
        with spikes termating on this time step.
        '''
        total_spikes = 0
        total_cells = 0
        self.bias = np.zeros(self.n_cells)
        for i, c in enumerate(self.cells):
            if c.activation > 0:
                # Fade activations
                c.activation = max([c.activation - c.fade_rate, 0])
            cell_terminated = False
            for seg in c.all_segments():
                for j in seg.connected_synapses():
                    source = seg.source_cell(j)
                    n_spikes = 0
                    if source:
                        dist = seg.syn_distances[j]
                        n_spikes = len(source.terminating_spikes(dist))
                        total_spikes += n_spikes
                        if n_spikes:
                            cell_terminated = True
                    else:
                        # Region 1 inputs act as 0-distance connections --
                        # activate immediately
                        n_spikes = 1 if self._input_active(
                            seg.syn_sources[j]) else 0
                    if n_spikes:
                        # log("Terminating %d spikes on %s" % (n_spikes, c))
                        c.activation += n_spikes * \
                            SPIKE_ACTIVATION_INCR + (random.random() / 20.)
                        if seg.distal() or seg.topdown():
                            self.bias[i] = 1
            if cell_terminated:
                total_cells += 1
            self.pre_activation[i] = c.activation
        log("Terminating %d spikes on %d cells" % (total_spikes, total_cells))

    def do_inhibition(self):
        '''
        Choose winners of cells with activation > threshold.
        Populates pre_activation

        Returns:
            Boolean array of which cells to activate (generate spikes)
        '''
        active = np.zeros(self.n_cells)
        for c in self.cells:
            pa = c.activation
            neighbors = self._neighbors_of(c)
            kth_score = self._kth_score(
                neighbors, self.pre_activation, k=self.brain.config("DESIRED_LOCAL_ACTIVITY"))
            if pa > 0 and pa >= kth_score:
                active[c.index] = 1
        return active

    def do_generate_spikes(self, spiking_after_inhibition):
        '''
        For each cell:
            - If activation above threshold, generate spikes
                which will be monitored by all downstream cells.

        '''
        self.spiking = np.zeros(self.n_cells)
        for i, c in enumerate(self.cells):
            c.spiking = False
            spiking = spiking_after_inhibition[i]
            if spiking:
                c.spike(self.brain.t)
                self.spiking[i] = 1

    def region_process(self):
        '''
        Region processing routine
        --------------
        Takes input and calculates active columns (sparse representation)
        This is a representation of input in context of previous input
        (same region and regions above -- top-down predictions)
        '''

        # TODO: Do we have a multi-region processing order or ops issue?
        t = self.brain.t

        # Phase 1: Terminate spikes that have reached cell body, and increment
        # activation
        self.do_terminate_spikes()

        # Phase 2: Inhibit activating cells > activation threshold (generate
        # SDR)
        if True:
            spiking_after_inhibition = self.do_inhibition()
        else:
            spiking_after_inhibition = np.zeros(self.n_cells)
            for i, c in enumerate(self.cells):
                if c.activation > CELL_ACTIVATION_THRESH:
                    spiking_after_inhibition[i] = 1

        # Phase 3: Generate spikes from cells active after inhibition
        self.do_generate_spikes(spiking_after_inhibition)

        self.last_activation = [cell.activation for cell in self.cells]

    ##################
    # Primary Processing Function
    ##################

    def process(self, input, learning_enabled=False):
        '''Process inputs for a region.

        After step is complete:
            * self.pre_activation...
            * self.last_activation is an array of activations before new input
            * cell.activation for each cell is updated
        '''
        self.input = input
        self.region_process()  # Calculates active cells
        return [c.activation for c in self.cells]

    ##################
    # Learning Rules
    ##################

    def learn(self):
        '''Update permanences and distances for each segment in region
        according to learning rules.

        For each synapse connected to a currently spiking source:
            - Calculate distance between spikes (and order, before/after)
            - If distance is within learning radius:
                - Move synapse towards nearest spike (either towards or away from cell body)

        Implications:
            First spike doesn't shift/learn (since no closest spike yet exists)
        '''
        for i, cell in enumerate(self.cells):
            for seg in cell.all_segments():
                for j in seg.connected_synapses():
                    source = seg.source_cell(j)
                    dist = seg.syn_distances[j]
                    if source:
                        if source.spiking:
                            nearest_spike_rel_dist = seg.nearest_spike(j)
                            if nearest_spike_rel_dist is not None:
                                if abs(nearest_spike_rel_dist) < LEARN_SPIKE_DIST:
                                    if nearest_spike_rel_dist != 0.0:
                                        # Spike has passed synapse
                                        passed = nearest_spike_rel_dist < 0
                                        move_toward = passed
                                        change = min(
                                            [LEARN_DIST_CHANGE, nearest_spike_rel_dist])
                                        if move_toward:
                                            # Distance reduces (toward cell
                                            # body)
                                            change *= -1
                                        seg.syn_distances[j] += change
                                        log("Synapse %d on %s @ %s moving %s towards %s" % (
                                            j, seg, dist, change, nearest_spike_rel_dist))
                                    else:
                                        log("Synapse %d on %s @ %s coincident!" % (
                                            j, seg, dist))
                                    seg.syn_permanences[
                                        j] += self.brain.config("PERM_LEARN_INC")
                                    if seg.syn_permanences[j] > 1:
                                        seg.syn_permanences[j] = 1

                seg.decay_permanences()


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
        # A minimum number of inputs that must be active for a column to be
        # considered during the inhibition step
        self.min_overlap = min_overlap
        # Defaults
        self.CONFIG = {
            "DESIRED_LOCAL_ACTIVITY": 1,
            "DISTAL_SYNAPSE_CHANCE": 0.4,
            "TOPDOWN_SYNAPSE_CHANCE": 0.3,
            "MAX_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.4,
            "MIN_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.1,
            "CELLS_PER_REGION": 13**2,
            "N_REGIONS": 2,
            "BIAS_WEIGHT": 0.6,
            "OVERLAP_WEIGHT": 0.4,
            "FADE_RATE": 0.9,
            "DISTAL_SEGMENTS": 1,
            "PROX_SEGMENTS": 2,
            "TOPDOWN_SEGMENTS": 0,
            "TOPDOWN_BIAS_WEIGHT": 0.5,
            "SYNAPSE_DECAY_PROX": 0.00005,
            "SYNAPSE_DECAY_DIST": 0.0,
            "PERM_LEARN_INC": 0.07,
            "PERM_LEARN_DEC": 0.04,
            "CHANCE_OF_INHIBITORY": 0.0,
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
            r = Region(self, i, n_inputs=n_inputs, n_cells=cpr,
                       n_cells_above=cpr if not top_region else 0)
            r.initialize()
            # Next region will have 1 input for each output cell
            n_inputs = cpr
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
        matchScore = nPredictedCells / \
            float(nBiasedCells) if nBiasedCells > 0 else 0.0
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

    def step(self, readings, learning=False):
        '''
        Step through all regions inputting output of each into next
        Returns output of last region
        '''
        log("~~~~~~~~~~~~~~~~~ Processing inputs at T%d" % self.t, level=1)
        self.inputs = readings
        _in = self.inputs
        for i, r in enumerate(self.regions):
            log("Step processing for region %d\n%s << Input" %
                (i, printarray(_in, continuous=True)), level=2)
            out = r.process(_in, learning_enabled=learning)
            _in = out
        if learning:
            for i, r in enumerate(self.regions):
                log("Learning for region %d" % (i), level=2)
                r.learn()
        self.t += 1  # Move time forward one step
        return out
