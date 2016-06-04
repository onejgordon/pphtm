# TODO

## Next experiments

* Are we unlearning patterns too quickly for segments? Should we fade slower?
* See invariant structures in R2 (maintaining activation during sequence, prximal learning rules)
* Proximal synapses never unlearning (what's the rule? check htm)
* Decrease fade rate to enable multi-step proximal learning
* Experiment with integration function (allow distal activation?)

## Implementation TODO

* Export swarm results to CV for easier analysis
* Topdown: 2X is learned to precede A, B', C'
* Distal: A is learned to precede B' (75% of time), and B (25% of the time)
* Proximal: 2X learns AB'C' via slowness?
* Why isn't distal prediction working? 2 synapses activating, but no bias
* Implement unboost for cells active (e.g. from bias) too frequently?
	- Unboost -> increase inhibitory synapse permanences
* Proximal cell detail connection grid not sized properly
* How do we form invariant structures?
	- Do higher regions learn multi-step sub-patterns after biases turn on? AB'C' -> 2
* Consider differences required by 1R proximal dynamic and 2R.
	- 1R needs to learn 1 SDR per input
	- 2R needs to learn multi-step combinations of inputs?
	- Do we need to enable distal connections (bias) to activate even in absence of ff?
	- Or is activation slowness sufficient?
* Linear regression on swarm runs to show stat. sig. of any relationships
* Redo printer with custom Widgets()

## Algorithm TODO

* How do inhibitory neurons learn?
	- Learning can occur due to post-synaptic activity only.
	- If post-synapse active too frequently, synapse from inh. cell strengthens?
* Prediction may be undermined by topdown. Consider relative bias strength in prediction matching?
* Does formation of invariant structures require lower-level slowness?
	- E.g. AB'C retains enough of AB' to fire higher level SDR representing AB'C
* Do we need more distal activation? Overlap instead of subset?
* Should we try boosting distal synapses if active duty cycle low?
* We need to unlearn synapses that lead to bias noise.

## PP TODO

* How is precision weighting modeled?
* How to code error signal? Predictions? (error is only output of a region, not neurono-based?)
* What is difference between prediction and activation?
	- HTM says this is NMDA spike from distal connections -> predicted cell state
	- *new* firing at higher level region indicates 'unpdredicted' lower-level activation
	- so new firing is probably weighted somehow -- passes on information more directly
	- when a predicted incoming signal is passed upwards, no change occurs...


## Current Problems

* All segments are learning same patterns? How to choose which to learn?
* A->B, since it's less frequent, is unable to learn this transition (we
* unlearn it every time E->B happens). Should we only learn on active segments?
* If so, how do we ensure we ever have activity? Should we learn on most
 active segment? Doesn't work because we then only learn on one seg each cell
* Should we unlearn distabl predictions that never activate? Causing noise
* in bias layer that doesn't go away.
* Looks like we are learning too much on one segment (active)
* straddline multiple patterns.