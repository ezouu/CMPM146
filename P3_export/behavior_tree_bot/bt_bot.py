#!/usr/bin/env python
#

"""
// There is already a basic strategy in place here. You can use it as a
// starting point, or you can throw it out entirely and replace it with your
// own.
"""
import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from behavior_tree_bot.behaviors import *
from behavior_tree_bot.checks import *
from behavior_tree_bot.bt_nodes import Selector, Sequence, Action, Check

from planet_wars import PlanetWars, finish_turn


# You have to improve this tree or create an entire new one that is capable
# of winning against all the 5 opponent bots
def setup_behavior_tree():

    # We want multiple phases each turn (defend + expand + attack).
    # Use Selectors with a final do_nothing() action to make each phase always succeed.
    root = Sequence(name='Turn Strategy (Defense -> Expand -> Attack -> Reinforce)')

    defense_phase = Selector(name='Defense Phase')
    defend_if_threatened = Sequence(name='Defend if threatened')
    defend_if_threatened.child_nodes = [
        Check(is_any_my_planet_under_threat),
        Action(defend_most_threatened_planet),
    ]
    evacuate_logic = Sequence(name='Evacuate if doomed')
    evacuate_logic.child_nodes = [
        Check(is_any_my_planet_under_threat),
        Action(evacuate_protocol)
    ]
    defense_phase.child_nodes = [defend_if_threatened, evacuate_logic, Action(do_nothing)]

    expand_phase = Selector(name='Expansion Phase')
    expand_if_possible = Sequence(name='Expand to valuable neutrals')
    expand_if_possible.child_nodes = [
        Check(if_neutral_planet_available),
        Action(expand_aggressively),
    ]
    expand_phase.child_nodes = [expand_if_possible, Action(do_nothing)]

    attack_phase = Selector(name='Attack Phase')
    attack_if_profitable = Sequence(name='Attack when profitable')
    attack_if_profitable.child_nodes = [
        Check(if_enemy_planet_available),
        Action(attack_opportunistically),
    ]
    attack_phase.child_nodes = [attack_if_profitable, Action(do_nothing)]

    reinforce_phase = Selector(name='Reinforcement Phase')
    reinforce_if_multi = Sequence(name='Reinforce weakest when multi-planet')
    reinforce_if_multi.child_nodes = [
        Check(have_multiple_planets),
        Action(reinforce_weakest_my_planet),
    ]
    reinforce_phase.child_nodes = [reinforce_if_multi, Action(do_nothing)]

    root.child_nodes = [defense_phase, expand_phase, attack_phase, reinforce_phase]

    logging.info('\n' + root.tree_to_string())
    return root

# You don't need to change this function
def do_turn(state):
    behavior_tree.execute(planet_wars)

if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    behavior_tree = setup_behavior_tree()
    try:
        map_data = ''
        while True:
            current_line = input()
            if len(current_line) >= 2 and current_line.startswith("go"):
                planet_wars = PlanetWars(map_data)
                do_turn(planet_wars)
                finish_turn()
                map_data = ''
            else:
                map_data += current_line + '\n'

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error in bot.")
