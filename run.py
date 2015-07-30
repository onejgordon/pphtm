#!/usr/bin/env python

from intell import Agent, Behavior, Sensor
from SimpleHeatGame import SimpleHeatGame, SimpleHeatWorld, SimpleHeatSource, SimpleHeatAgent

WORLD_SIDE_LEN = 30

def main():
	game = SimpleHeatGame(animate=True, timeout=200)

	world = SimpleHeatWorld([WORLD_SIDE_LEN, WORLD_SIDE_LEN])
	world.addSource(SimpleHeatSource(1.5,[5,5], heat=5))

	agent = SimpleHeatAgent(world)

	game.setWorld(world)
	game.setAgent(agent)

	game.run()


if __name__ == "__main__":
    main()