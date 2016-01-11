#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from htm_brain import HTMBrain
from intell import Agent, Behavior, Sensor
from SimpleHeatGame import SimpleHeatGame, SimpleHeatWorld, SimpleHeatSource, SimpleHeatAgent
import numpy as np

WORLD_SIDE_LEN = 30

def main():
	game = SimpleHeatGame(animate=True, timeout=50)

	world = SimpleHeatWorld([WORLD_SIDE_LEN, WORLD_SIDE_LEN])
	world.addSource(SimpleHeatSource(1.5,[5,5], heat=5))

	brain = HTMBrain(columns_per_region=[20,10,3])  
	agent = SimpleHeatAgent(world, brain)

	game.setWorld(world)
	game.setAgent(agent)

	game.run()


if __name__ == "__main__":
    main()