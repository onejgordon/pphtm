# TODO

## Implementation TODO

* Show run summary at end of swarm with 3d chart of params and outcome to help learning

## Algorithm TODO

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
* Can we simply extend distal bias from regions above?

## Other TODO



## Eventual

* Render proximal connections (and re-initialize changes there too)
* Now it's time to build invariant SDRs at a higher region layer.
* These can predict all pattern members regardless of order.

## Current Problems

* Once a region has learned, bias should be a subset of proximal overlap on each step

* All segments are learning same patterns? How to choose which to learn?
* A->B, since it's less frequent, is unable to learn this transition (we
* unlearn it every time E->B happens). Should we only learn on active segments?
* If so, how do we ensure we ever have activity? Should we learn on most
 active segment? Doesn't work because we then only learn on one seg each cell
* Should we unlearn distabl predictions that never activate? Causing noise
* in bias layer that doesn't go away.
* Looks like we are learning too much on one segment (active)
* straddline multiple patterns.