# NOTES

- Present setup learns simple pattern 2 in 120 steps.
	* 9x9
	* DISTAL_SYNAPSE_CHANCE = 0.5
	* TOPDOWN_SYNAPSE_CHANCE = 0.4
	* MAX_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.4
	* MIN_PROXIMAL_INIT_SYNAPSE_CHANCE = 0.1
- Is bias used for prediction from right step?
- We probably need to move bias calculation to end of step()