import pyhop
import json

def check_enough(state, ID, item, num):
	if getattr(state,item)[ID] >= num: return []
	return False

def produce_enough(state, ID, item, num):
	return [('produce', ID, item), ('have_enough', ID, item, num)]

pyhop.declare_methods('have_enough', check_enough, produce_enough)

def produce(state, ID, item):
	return [('produce_{}'.format(item), ID)]

pyhop.declare_methods('produce', produce)

def make_method(name, rule):
	def method(state, ID):
		# your code here
		pass

	return method

def declare_methods(data):
	# some recipes are faster than others for the same product even though they might require extra tools
	# sort the recipes so that faster recipes go first

	# your code here
	# hint: call make_method, then declare the method to pyhop using pyhop.declare_methods('foo', m1, m2, ..., mk)	
	pass			

def make_operator(rule):
	def operator(state, ID):
		time_cost = rule.get('Time', 0)
		if state.time[ID] < time_cost:
			return False

		requires = rule.get('Requires', {})
		for item, num in requires.items():
			if getattr(state, item)[ID] < num:
				return False

		consumes = rule.get('Consumes', {})
		for item, num in consumes.items():
			if getattr(state, item)[ID] < num:
				return False

		# apply effects
		for item, num in consumes.items():
			getattr(state, item)[ID] -= num

		produces = rule.get('Produces', {})
		for item, num in produces.items():
			getattr(state, item)[ID] += num

		state.time[ID] -= time_cost
		return state
	return operator

def declare_operators(data):
	ops = []
	for recipe_name, rule in data['Recipes'].items():
		op = make_operator(rule)
		# Name operators so plans are readable and stable.
		safe = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in recipe_name.strip())
		op.__name__ = f'op_{safe}'
		ops.append(op)

	pyhop.declare_operators(*ops)

def add_heuristic(data, ID):
	# prune search branch if heuristic() returns True
	# do not change parameters to heuristic(), but can add more heuristic functions with the same parameters: 
	# e.g. def heuristic2(...); pyhop.add_check(heuristic2)
	def heuristic(state, curr_task, tasks, plan, depth, calling_stack):
		# your code here
		return False # if True, prune this branch

	pyhop.add_check(heuristic)

def define_ordering(data, ID):
	# if needed, use the function below to return a different ordering for the methods
	# note that this should always return the same methods, in a new order, and should not add/remove any new ones
	def reorder_methods(state, curr_task, tasks, plan, depth, calling_stack, methods):
		return methods
	
	pyhop.define_ordering(reorder_methods)

def set_up_state(data, ID):
	state = pyhop.State('state')
	setattr(state, 'time', {ID: data['Problem']['Time']})

	for item in data['Items']:
		setattr(state, item, {ID: 0})

	for item in data['Tools']:
		setattr(state, item, {ID: 0})

	for item, num in data['Problem']['Initial'].items():
		setattr(state, item, {ID: num})

	return state

def set_up_goals(data, ID):
	goals = []
	for item, num in data['Problem']['Goal'].items():
		goals.append(('have_enough', ID, item, num))

	return goals

if __name__ == '__main__':
	import sys
	rules_filename = 'crafting.json'
	if len(sys.argv) > 1:
		rules_filename = sys.argv[1]

	with open(rules_filename) as f:
		data = json.load(f)

	state = set_up_state(data, 'agent')
	goals = set_up_goals(data, 'agent')

	declare_operators(data)
	declare_methods(data)
	add_heuristic(data, 'agent')
	define_ordering(data, 'agent')

	# pyhop.print_operators()
	# pyhop.print_methods()

	# Hint: verbose output can take a long time even if the solution is correct; 
	# try verbose=1 if it is taking too long
	pyhop.pyhop(state, goals, verbose=1)
	# pyhop.pyhop(state, [('have_enough', 'agent', 'cart', 1),('have_enough', 'agent', 'rail', 20)], verbose=3)
