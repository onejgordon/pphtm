# NOTES

## Timing Theory

https://docs.google.com/drawings/d/1XyqZDm_cXR_zIAthYpix3TU70FuP2WEI--8nsgmgbF4/edit

* Overall principle, synapses learn when spike terminates (end of axon) and we move synapse's location to try to co-locate with a nearby spike on the same segment, such that next time the same inputs fire, they will reach cell body closer together (more powerful input).
* Representations:
	- Distance between synapses (axons different length / time constant): correction between actual learned timing sequence and difference between axon length.
	- Distance between synapses (axons same length): delay between learned firing
	- Axon length: ??
	- Synapse location from cell body: ??

### Problems

* How can second synapse to fire learn? Can it be done without reverse spikes?
	- If spike intervals are consistent, could synchronicity between downstream spike and upstream feedback help reinforce the interval / distance?
* Can we do proximal feed-forward learning with the same mechanism? Move synapses towards cell body?
	- If cell body produces spike upstream towards synapse when it fires, we may be able to use the same mechanism (move synapse towards closest spike, including cell body spike)
	- But... While we've learned that input and output cell fire together, the input cell wont actually be influencing the output cell's firing. Problem?
	- Does this solve a problem of simultanaity in hebbian learning?
	- Illustrate these cases (connection becoming more proximal?)
* Does this work with many synapses on a segment?
* How to handle cases where spikes arrive at same time (simultaneous activation, same axon length) to different parts of segment.
* How does multi-branching affect this

### Neuro confirmations

* How often do action potentials move both ways through dendrite?
* Timing to differentiate subtle speech sound differences may be important, e.g. ABA vs AAABAAA. Neuro evidence to support humans differentiate these?


## TODO

* Revert proximal handling to HTM
	- Spike/distal activation can be separated for use as bias & predictions?
	- Proximal activates as normal
* Model axon length
* How do we predict within the continuous paradigm?
	- Split distal/proximal/topdown spikes & activation?
	- Map distal/topdown to input activation patterns?

