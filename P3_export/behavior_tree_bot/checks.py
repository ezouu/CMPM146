

def if_enemy_planet_available(state):
    return any(state.enemy_planets())


def if_neutral_planet_available(state):
    return any(state.neutral_planets())


def have_fleet_in_flight(state):
    """True if we already have any fleet traveling."""
    return len(state.my_fleets()) > 0


def total_ships(player_planets, player_fleets):
    """Utility for quickly comparing players."""
    return sum(p.num_ships for p in player_planets) + sum(f.num_ships for f in player_fleets)


def have_largest_fleet(state):
    return total_ships(state.my_planets(), state.my_fleets()) > total_ships(state.enemy_planets(), state.enemy_fleets())


def _incoming_enemy_ships_to_planet(state, planet_id, within_turns=None):
    incoming = 0
    for f in state.enemy_fleets():
        if f.destination_planet != planet_id:
            continue
        if within_turns is not None and f.turns_remaining > within_turns:
            continue
        incoming += f.num_ships
    return incoming


def is_any_my_planet_under_threat(state, within_turns=10):
    """
    Conservative threat check: returns True if any of our planets has enemy ships inbound soon.
    """
    for p in state.my_planets():
        if _incoming_enemy_ships_to_planet(state, p.ID, within_turns=within_turns) > 0:
            return True
    return False


def can_capture_any_neutral_planet(state):
    """
    True if we have enough ships on at least one planet to capture at least one neutral planet (ignoring travel-time).
    """
    my_planets = state.my_planets()
    neutrals = state.neutral_planets()
    if not my_planets or not neutrals:
        return False
    strongest = max(my_planets, key=lambda p: p.num_ships, default=None)
    if strongest is None:
        return False
    return any(strongest.num_ships > (n.num_ships + 1) for n in neutrals)


def can_attack_any_enemy_planet(state):
    """
    True if we have enough ships on at least one planet to likely win an attack on at least one enemy planet
    (simple growth-aware estimate).
    """
    my_planets = state.my_planets()
    enemies = state.enemy_planets()
    if not my_planets or not enemies:
        return False

    source = max(my_planets, key=lambda p: p.num_ships, default=None)
    if source is None:
        return False

    for target in enemies:
        d = state.distance(source.ID, target.ID)
        required = target.num_ships + target.growth_rate * d + 1
        if source.num_ships > required:
            return True
    return False


def have_multiple_planets(state):
    """True if we control 2+ planets."""
    return len(state.my_planets()) >= 2
