# What this is

* Basic implementation of HTM network (Neumenta, Hawkins)
* Adjusted to support continuous activations that fall off over time at different rates
* Temporal pooler removed in favor of spatial-temperal pooling
* http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf


# CURRENT PROBLEMS

* Once a region has learned, bias should be a subset of proximal overlap on each step

* All segments are learning same patterns? How to choose which to learn?
* A->B, since it's less frequent, is unable to learn this transition (we
* unlearn it every time E->B happens). Should we only learn on active segments?
* If so, how do we ensure we ever have activity? Should we learn on most
* active segment? Doesn't work because we then only learn on one seg each cell

* Should we unlearn distabl predictions that never activate? Causing noise
* in bias layer that doesn't go away.

* Looks like we are learning too much on one segment (active),
* straddline multiple patterns.