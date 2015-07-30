#!/usr/bin/env python

# Jeremy Gordon

# Basic implementation of HTM network (Neumenta, Hawkins)
# http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf

import numpy as np
import random

# Settings

VERBOSE = True

CONNECTED_PERM = 0.2  # If the permanence value for a synapse is greater than this value, it is said to be connected.
MIN_OVERLAP = 2  # A minimum number of inputs that must be active for a column to be considered during the inhibition step
DUTY_HISTORY = 1000
BOOST_MULTIPLIER = 10
MIN_THRESHHOLD = 2  # Minimum segment activity for learning
INIT_PERMANENCE = 0.2  # New learned synapses
INIT_PERMANENCE_JITTER = 0.05  # Max offset from CONNECTED_PERM when initializing synapses
NEW_SYNAPSE_COUNT = 5

RETAIN_STEPS = 3

class SegmentUpdate(object):

    def __init__(self, segment_index=-1, active_synapses=None, sequence=False):
        self.segment_index = segment_index
        if active_synapses:
            self.active_synapses = [(s.column.index, s.source) for s in active_synapses]
        else:
            self.active_synapses = []
        self.sequence = sequence


class Synapse(object):
    '''
    A synapse part of a dendrite segment of a column or cell
    Active if its input is active
    Connected if permanence > threshhold
    '''

    def __init__(self, region, source, permanence, column=None):
        self.region = region
        self.permanence = permanence  # (0,1)
        # Source
        self.source = source  # Index of source input (or cell in column, if distal)
        self.column = column  # None if proximal


    def proximal(self):
        return self.column is None

    def distance_from(self, location):
        return abs(self.source - location)

    def connected(self, connectionPermanence=CONNECTED_PERM):
        return self.permanence > connectionPermanence

    def active(self, t, state='active'):
        if self.proximal():
            input = self.region._get_input(self.source, t)
            return input
        else:
            # Distal
            if state == 'active':
                return self.column.cells[self.source]._in_active_state(t)
            elif state == 'learn':
                return self.column.cells[self.source]._in_learn_state(t)
            else:
                raise Exception("State not allowed")

class Segment(object):
    '''
    Dendrite segment of column or cell
    Distal (horizontal) or proximal (feed forward)
    TODO: How to initialize potential synapses?
    '''
    def __init__(self, index, region):
        self.index = index
        self.region = region
        self.sequence = False
        self.activation_threshhold = 1

        # State
        self.potential_synapses = []  # List of Synapse() objects.

    def __repr__(self):
        t = self.region.brain.t
        return "<Segment potential=%d connected=%d>" % (len(self.potential_synapses), len(self.connected_synapses()))

    def initialize(self, column=None):
        # Setup initial potential synapses
        MAX_INIT_SYNAPSE_CHANCE = 0.5
        MIN_INIT_SYNAPSE_CHANCE = 0.05
        n_inputs = self.region.inputs_per_region
        for source in range(n_inputs):
            dist = abs(column.center - source)            
            chance_of_synapse = (MAX_INIT_SYNAPSE_CHANCE * (1 - float(dist)/n_inputs)) + MIN_INIT_SYNAPSE_CHANCE
            add_synapse = random.random() < chance_of_synapse
            if add_synapse:
                perm = CONNECTED_PERM + INIT_PERMANENCE_JITTER*(random.random()-0.5)
                s = Synapse(self.region, source, perm)
                self.potential_synapses.append(s)
        print "Initialized %s" % self

    def active(self, t, state='active'):  # state in ['active','learn']
        '''
        This routine returns true if the number of connected synapses on segment
        s that are active due to the given state at time t is greater than
        activationThreshold. The parameter state can be activeState, or
        learnState.
        '''
        n_active = len(self.active_synapses(t, state=state))
        return n_active > self.activation_threshhold

    def active_synapses(self, t, state='active', connectionPermanence=CONNECTED_PERM):
        return filter(lambda s : s.active(t, state=state), self.connected_synapses(connectionPermanence=connectionPermanence))

    def connected_synapses(self, connectionPermanence=CONNECTED_PERM):
        return filter(lambda s : s.connected(connectionPermanence=connectionPermanence), self.potential_synapses)

    def get_segment_active_synapses(self, t, new_synapses=False):
        '''
        Return SegmentUpdate() object
        '''
        active_synapses = self.active_synapses(t)
        n_new_synapses = NEW_SYNAPSE_COUNT - len(active_synapses) if new_synapses else 0
        if n_new_synapses > 0:
            learn_synapses = self.active_synapses(t, state='learn')
            active_synapses.extend(random.sample(learn_synapses, min(len(learn_synapses), n_new_synapses)))
        return SegmentUpdate(segment_index=self.index, active_synapses=active_synapses, sequence=False)

class Cell(object):
    '''
    An HTM abstraction of one or more biological neurons
    Has multiple distal dendrite segments with connections from other cells in the region
    '''

    def __init__(self, region, col_index, index, n_segments=3):
        self.col_index = col_index  # Column index
        self.index = index  # In column
        self.region = region
        self.segments = [Segment(s, self.region) for s in range(n_segments)]

        self.segment_update_list = []  # List of SegmentUpdate() objects

    def _in_active_state(self, tMinus=0):
        present = self.region.brain.t
        return self.region.active_state[self.col_index][self.index][present - tMinus]

    def _in_predictive_state(self, tMinus=0):
        present = self.region.brain.t
        return self.region.predictive_state[self.col_index][self.index][present - tMinus]

    def _in_learn_state(self, tMinus=0):
        present = self.region.brain.t
        return self.region.learn_state[self.col_index][self.index][present - tMinus]
    
    def _set_active_state(self):
        present = self.region.brain.t
        self.region.active_state[self.col_index][self.index][present] = 1

    def _set_predictive_state(self):
        present = self.region.brain.t
        self.region.predictive_state[self.col_index][self.index][present] = 1

    def _set_learn_state(self):
        present = self.region.brain.t
        self.region.learn_state[self.col_index][self.index][present] = 1

    def active_segments(self, state='active'):
        return filter(lambda seg : seg.active(state=state), self.segments)

    def get_active_segment(self, state='active'):
        # Get active segments sorted by sequence boolean, then activity level
        ordered = sorted(self.active_segments(state=state), key=lambda seg : (seg.sequence, seg.active_synapses()))
        if ordered:
            return ordered[0]
        else:
            return None

    def get_best_matching_segment(self, t):
        '''
        Return segment index or -1 and # of active synapses
        '''
        best_seg_index = -1
        most_active_synapses = 0
        for i, seg in enumerate(self.segments):
            active_synapses = seg.active_synapses(t, connectionPermanence=0) # NOTE, via active/learn?
            if active_synapses > most_active_synapses and active_synapses > MIN_THRESHHOLD:
                most_active_synapses = active_synapses
                best_seg_index = i
        return (best_seg_index, most_active_synapses)

    def adapt_segments(positive_reinforcement=True):
        # Prepare update dictionary
        update_active_synapses_by_segment = {}  # segment_index -> list of synapse sources (c, i)
        # TODO: Do active_synapses need column index?
        for su in self.segment_update_list:
            if su.segment_index not in update_active_synapses_by_segment:
                update_active_synapses_by_segment[su.segment_index] = []
            update_active_synapses_by_segment[su.segment_index].extend(su.active_synapses)
        for seg in self.segments:
            active_synapses = update_active_synapses_by_segment[seg.index]
            for s in seg.potential_synapses:
                if s.source in active_synapses:
                    if positive_reinforcement:
                        s.permanence += self.region.permanence_inc
                    else:
                        s.permanence -= self.region.permanence_dec
                    active_synapses.remove(s.index)  # Remove so we remain with only new synapses
                elif positive_reinforcement:
                    s.permanence -= self.region.permanence_dec
            if len(active_synapses):
                # Any synapses in segmentUpdate that do yet exist get added with a permanence count of initialPerm.                
                for c, i in active_synapses:
                    col = self.region.columns[c]
                    seg.potential_synapses.append(Synapse(self.region, i, permanence=INIT_PERMANENCE, column=col))
                    # TODO: Index vs source?

        # Clear queued upates
        self.segment_update_list = []


class Column(object):
    '''
    Made up of one or more 'vertically stacked' cells
    Has a proiximal 'shared' dendrite receiving feed-forward input
    '''

    def __init__(self, region, index):
        # Setup
        self.index = index
        self.region = region
        self.center = np.random.random() * region.inputs_per_region  # In 1D, this is a single integer index above a particular input. TODO: Convert to 2D?
        self.segment = Segment(index=0, region=region)  # Proximal dendrite segment with synapses

        # State
        self.cells = []

    def initialize(self):
        for c in range(self.region.cells_per_column):
            cell = Cell(self.region, self.index, c)
            self.cells.append(cell)
        self.segment.initialize(column=self)

    def get_best_matching_cell(self, t):
        '''
        Return the cell with the best matching 
        segment (as defined above). If no cell has a matching segment, 
        then return the cell with the fewest number of segments.
        '''
        best_cell = None
        best_cell_active_synapses = 0
        best_seg_index = -1
        n_segments = []
        for c in self.cells:
            n_segments.append(len(c.segments))
            seg_index, active_synapses = c.get_best_matching_segment(t)
            if seg_index > 0:
                if active_synapses > best_cell_active_synapses:
                    best_cell = c
                    best_cell_active_synapses = active_synapses
                    best_seg_index = seg_index
        if best_cell:
            # Return best matching cell (if found)
            return (best_cell, best_seg_index)
        else:
            # Return cell with least # of segments
            return (self.cells[n_segments.index(min(n_segments))], best_seg_index)


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
        self.active_duty_cycle = sum(self.recent_active_duty) / len(self.recent_active_duty)
        self.overlap_duty_cycle = sum(self.recent_overlap_duty) / len(self.recent_overlap_duty)        

    def connected_receptive_field_size(self):
        '''
        Returns max distance (radius) among currently connected synapses
        '''
        connected = self.connected_synapses()
        return max([s.distance_from(self.center) for s in connected])


    def connected_synapses(self):
        return self.segment.connected_synapses()



class Region(object):
    '''
    Made up of many columns
    '''

    def __init__(self, brain, desired_local_activity=5, permanence_inc=0.1, permanence_dec=0.1, columns_per_region=10, cells_per_column=3, inputs_per_region=20):
        self.brain = brain

        # Region constants (spatial)
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.desired_local_activity = desired_local_activity  # Desired winners after input step
        self.inhibition_radius = 0 # Average connected receptive field size of the columns

        # Region constants (temporal)
        self.cells_per_column = cells_per_column

        # Hierarchichal setup
        self.columns_per_region = columns_per_region
        self.inputs_per_region = inputs_per_region
        self.columns = []

        # Current state

        # New form data structures - Spatial
        self.active_columns = np.zeros((1, self.columns_per_region))  # Column activity at time t, column c - active_columns(t, c) is bool
        self.input = np.zeros((1, self.inputs_per_region))  # Inputs at time t - input[t, j] is int
        self.overlap = np.zeros(self.columns_per_region)  # Overlap for each column. overlap[c] is int
        self.neighbors = self._init_neighbors()  # List of lists (column neighbors list for each column neighbors[c])
        self.boost = np.ones(self.columns_per_region, dtype=float)  # Boost value for column c
        self.potential_synapses = [c.segment.potential_synapses for c in self.columns] # potential_synapses(c)
        self.connected_synapses = [c.segment.connected_synapses() for c in self.columns]  # TODO: Need a getter function?
        self.active_duty_cycle = np.array(self.columns_per_region, dtype=float)  # Sliding average: how often column c has been active after inhibition (e.g. over the last 1000 iterations).
        self.overlap_duty_cycle = np.array(self.columns_per_region, dtype=float)  # Sliding average: how often column c has had significant overlap (> minOverlap)
        self.min_duty_cycle = np.array(self.columns_per_region, dtype=float)  # Minimum desired firing rate for a cell. If a cell's firing rate falls below this value, it will be boosted. This value is calculated as 1% of the maximum firing rate of its neighbors.
        # History
        self.recent_active_duty = []  # After inhibition, list of bool
        self.recent_overlap_duty = []  # Overlap > minOverlap, list of bool

        # New form data structures - Temporal        
        self.active_state = np.zeros((self.columns_per_region, self.cells_per_column, 1))  # active_state(c, i, t)
        self.predictive_state = np.zeros((self.columns_per_region, self.cells_per_column, 1))  # predictive_state(c, i, t)
        self.learn_state = np.zeros((self.columns_per_region, self.cells_per_column, 1))  # learn_state(c, i, t)

    def initialize(self):
        # Create columns
        for i in range(self.columns_per_region):
            col = Column(region=self, index=i)
            col.initialize()
            self.columns.append(col)

        
    def _get_input(self, j, tMinus=0):
        '''
        At index j, at time t - tMinus
        tMinus 0 is now, positive is steps back in time
        '''
        present = self.input.shape[0]  # Number of time step rows in input ndarray
        t = present - 1 - tMinus
        if t >= 0:
            return self.input[t][j]
        else:
            raise Exception("History error - cant get input at time %s" % t)

    def _kth_score(self, cols, k):
        '''
        Given list of columns, calculate kth highest overlap value
        '''
        return sorted(self.overlap)[k-1]

    def _max_duty_cycle(self, cols):
        return max([c.active_duty_cycle for c in cols])

    def _init_neighbors(self):
        neighbors = []
        for c in self.columns:
            cneighbors = self._neighbors_of(c)
            neighbors.append(cneighbors)
        return neighbors

    def _neighbors_of(self, column):
        '''
        Return all columns within inhibition radius
        '''
        _neighbors = []
        for c in self.columns:
            if column.index == c.index:
                continue
            dist = abs(c.center - column.center)
            if dist < self.inhibition_radius:
                _neighbors.append(c)
        return _neighbors

    def _boost_function(self, c, min_duty_cycle):
        if self.active_duty_cycle[c] >= min_duty_cycle:
            b = 1.0
        else:
            b = 1 + (min_duty_cycle - self.active_duty_cycle[c]) * BOOST_MULTIPLIER
        return b

    def _increase_permanences(self, c, scale):
        '''
        Increase the permanence value of every synapse (column c) by a scale factor
        '''
        for s in self.columns[c].segment.potential_synapses:
            s.permanence *= scale


    def _get_active_columns(self):
        # At most recent time step (t=-1)
        return filter(lambda c : bool(self.active_columns[-1][c.index]), self.columns)


    ##################
    # Spatial Pooling
    ##################

    def do_overlap(self):
        '''
        Return overlap as a number for each column representing ovlerap (above floor) with current input
        '''
        overlaps = np.zeros(len(self.columns))  # Initialize overlaps to 0
        for c, col in enumerate(self.columns):
            overlaps[c] = 0
            for s in col.connected_synapses():
                overlaps[c] += self.inputs[s.source]
            if overlaps[c] < MIN_OVERLAP:
                overlaps[c] = 0
            else:
                overlaps[c] = overlaps[c] * self.boost[c]
        if VERBOSE:
            print "Overlap: ", overlaps
        return overlaps

    def do_inhibition(self):
        '''
        Get active columns after inhibition around strongly overlapped columns
        '''
        active = np.zeros(len(self.columns))
        for c in self.columns:
            minLocalActivity = self._kth_score(self._neighbors_of(c), self.desired_local_activity)
            ovlp = self.overlap[c.index]
            if ovlp > 0 and ovlp >= minLocalActivity:
                active[c.index] = 1
        return active

    def do_learning(self):
        for c in active:
            for s in c.potential_synapses:
                if s.active(self.inputs):
                    s.permanence += self.permanence_inc
                    s.permanence = min(1.0, s.permanence)
                else:
                    s.permanence -= self.permanence_dec
                    s.permanence = max(0.0, s.permanence)

        all_field_sizes = []
        for c, col in enumerate(self.columns):
            min_duty_cycle = 0.01 * self._max_duty_cycle(self.neighbors[c])
            column_active = self._get_active_columns()[c]
            sufficient_overlap = self.overlap[c] > MIN_OVERLAP
            self.update_duty_cycles(c, active=column_active, overlap=sufficient_overlap)
            self.boost[c] = self._boost_function(c, min_duty_cycle)  # Updates boost value for column (higher if below min)

            # Check if overlap duty cycle less than minimum (note: min is calculated from max *active* not overlap)
            if self.overlap_duty_cycle[c] < min_duty_cycle:
                self._increase_permanences(c, 0.1 * CONNECTED_PERM)

            all_field_sizes.append(self.columns[c].connected_receptive_field_size())

        # Update inhibition radius (based on updated active connections in each column)
        self.inhibition_radius = tools.average(all_field_sizes)

    def spatial_pooling(self, inputs, learning_enabled=True):
        '''
        Spatial pooling routine
        --------------
        Takes inputs and calculates active columns (sparse representation) for input into temporal pooling
        '''
        self.inputs = inputs

        # Phase 1: Overlap
        self.overlap = self.do_overlap()

        # Phase 2: Inhibition    
        # Add present time's active columns to active_columns history
        self.active_columns = np.vstack((self.active_columns, self.do_inhibition()))

        # Phase 3: Learning
        if learning_enabled:
            self.do_learning()

    ##################
    # Temporal Pooling
    ##################

    # Phase 1
    def do_active_state(self):
        '''
        Calculate active state for each cell in each active column based on prior step's states
        '''
        t = self.brain.t
        for col in self._get_active_columns():
            buPredicted = False
            lcChosen = False
            prior_predictive_state = [c._in_predictive_state() for c in col.cells]
            for i, c in enumerate(col.cells):
                if prior_predictive_state[i]:
                    # Cell was in predictive state
                    seg = c.get_active_segment('active')
                    if seg.sequence:
                        # Cell was predicted due to active sequence segment
                        buPredicted = True
                        c._set_active_state()
                        if seg.active(self.brain.t - 1, 'learn'):
                            # Active in previous step from learn-state cells
                            lcChosen = True
                            c._set_learn_state()

            if not buPredicted:
                # Activate all
                for c in col.cells:
                    c._set_active_state()

            if not lcChosen:
                # Choose best matching cell
                c, seg_index = col.get_best_matching_cell(t-1)
                # Set learn state...
                c._set_learn_state()
                # ...and add a segment to this cell
                sUpdate = c.segments[seg_index].get_segment_active_synapses(t-1, new_synapses=True)
                sUpdate.sequence = True
                c.segment_update_list.append(sUpdate)

    # Phase 2
    def do_predictive_state(self):
        t = self.brain.t
        output = np.zeros(len(self.columns))
        for i, col in enumerate(self.columns):
            for cell in col.cells:
                for seg in cell.segments:
                    if seg.active(t):
                        cell._set_predictive_state()
                        
                        activeUpdate = seg.get_segment_active_synapses(t)
                        cell.segment_update_list.append(activeUpdate)

                        predSegment = cell.get_best_matching_segment(t-1)
                        predUpdate = predSegment.get_segment_active_synapses(t-1, new_synapses=True)
                        cell.segment_update_list.append(predUpdate)
                    
                # TODO: Is output calculation right, boolean OR of any predictive/active cell makes column output 1?
                if cell._in_predictive_state() or cell._in_active_state():
                    output[i] = 1
        
        return output

    # Phase 3
    def do_update_synapses(self):
        for i, col in enumerate(self.columns):
            for cell in col.cells:
                if cell._in_learn_state():
                    cell.adapt_segments(positive_reinforcement=True)
                else:
                    stopped_predicting = cell._in_predictive_state(t-1) and not cell._in_predictive_state()
                    if stopped_predicting:
                        self.adapt_segments(positive_reinforcement=False)


    def temporal_pooling(self, learning_enabled=True):
        '''
        Temporal pooling routine
        --------------
        Takes inputs of active columns from spatial pooler and calculates active cells (predictive or active)
        These form input for next level (how? columns with any active?)
        '''
        # Phase 1: Active state
        self.do_active_state()

        # Phase 2: Predictive state    
        out = self.do_predictive_state()

        # Phase 3: Learning
        if learning_enabled:
            self.do_update_synapses()

        return out

    ##################
    # Primary Step Function
    ##################

    def step(self, input, learning_enabled=False):
        if VERBOSE:
            print "Step... Inputs: %s" % input
        self.spatial_pooling(input, learning_enabled=learning_enabled)  # Calculates active columns
        if VERBOSE:
            print "Active columns: %s" % self.active_columns[-1]
        out = self.temporal_pooling(learning_enabled=learning_enabled)
        if VERBOSE:
            print "Output: %s" % out
        return out

class HTMBrain():

    def __init__(self):
        self.regions = []
        self.t = 0

    def __repr__(self):
        return "<HTMBrain regions=%d>" % len(self.regions)

    def initialize(self, n_regions=1, input_length=5):
        for i in range(n_regions):
            r = Region(self, inputs_per_region=input_length)
            r.initialize()
            self.regions.append(r)
        print "Initialized %s" % self

    def process(self, input):
        print "Processing inputs in time t = %d" % self.t
        _in = input
        for r in self.regions:
            out = r.step(_in)
            _in = out
        self.t += 1 # Move time forward one step
        return out

if __name__ == "__main__":
    INPUT_LEN = 10
    DURATION = 1
    b = HTMBrain()
    # Run white noise
    b.initialize(n_regions=2, input_length=INPUT_LEN)
    for i in range(DURATION):
        out = b.process(np.random.random_integers(0, 1, INPUT_LEN))
