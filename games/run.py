#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from agent_world import Agent, Behavior, Sensor
from SimpleHeatGame import SimpleHeatGame, SimpleHeatWorld, SimpleHeatSource, SimpleHeatAgent
from pphtm.pphtm_brain import PPHTMBrain

WORLD_SIDE_LEN = 30

def main():
	game = SimpleHeatGame(animate=True, timeout=200)

	world = SimpleHeatWorld([WORLD_SIDE_LEN, WORLD_SIDE_LEN])
	world.addSource(SimpleHeatSource(1.5,[5,5], heat=5))

	brain = PPHTMBrain()
	agent = SimpleHeatAgent(world, brain)

	game.setWorld(world)
	game.setAgent(agent)

	game.run()


if __name__ == "__main__":
    main()