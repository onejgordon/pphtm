# What this is

* Basic implementation of HTM network (See Numenta, Hawkins)
* Adjusted to support continuous activations that fall off over time at different rates
* Temporal pooler removed in favor of spatial-temperal pooling
* pphtm includes top-down predictions
* http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf

Run python /tests/test_chtm.py to run tests/visualization on sample data
Run python /tests/swarm_pphtm.py to swarm over a range of configuration vars