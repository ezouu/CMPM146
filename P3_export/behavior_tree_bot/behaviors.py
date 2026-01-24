from planet_wars import issue_order


def do_nothing(state):
    """Convenience action for 'always succeed' fallbacks in the behavior tree."""
    return True


def _incoming_ships(state, planet_id, owner, within_turns=None):
    fleets = state.my_fleets() if owner == 1 else state.enemy_fleets()
    total = 0
    for f in fleets:
        if f.destination_planet != planet_id:
            continue
        if within_turns is not None and f.turns_remaining > within_turns:
            continue
        total += f.num_ships
    return total


def _net_incoming_ships(state, planet_id, within_turns=None):
    return _incoming_ships(state, planet_id, owner=1, within_turns=within_turns) - \
           _incoming_ships(state, planet_id, owner=2, within_turns=within_turns)


def _reserve_ships(state, planet_id, horizon=10, base_reserve=6):
    """
    Keep some ships behind for defense.
    If enemy fleets are inbound soon, reserve more.
    """
    incoming_enemy = _incoming_ships(state, planet_id, owner=2, within_turns=horizon)
    incoming_my = _incoming_ships(state, planet_id, owner=1, within_turns=horizon)
    net_threat = max(0, incoming_enemy - incoming_my)
    return max(1, base_reserve + net_threat)


def _strongest_my_planet(state):
    return max(state.my_planets(), key=lambda p: p.num_ships, default=None)


def _weakest_my_planet(state):
    return min(state.my_planets(), key=lambda p: p.num_ships, default=None)


def _weakest_enemy_planet(state):
    return min(state.enemy_planets(), key=lambda p: p.num_ships, default=None)


def _weakest_neutral_planet(state):
    return min(state.neutral_planets(), key=lambda p: p.num_ships, default=None)


def _best_neutral_planet(state, source):
    """
    Pick a neutral target by simple ROI heuristic.
    Higher growth, fewer ships, closer distance are preferred.
    """
    best = None
    best_score = None
    for t in state.neutral_planets():
        d = state.distance(source.ID, t.ID)
        score = (t.growth_rate + 1) / ((t.num_ships + 1) * (d + 1))
        if best is None or score > best_score:
            best, best_score = t, score
    return best


def _best_enemy_planet(state, source):
    """
    Pick an enemy target by simple ROI heuristic that accounts for growth during travel.
    """
    best = None
    best_score = None
    for t in state.enemy_planets():
        d = state.distance(source.ID, t.ID)
        required = t.num_ships + t.growth_rate * d + 1
        score = (t.growth_rate + 1) / (required * (d + 1))
        if best is None or score > best_score:
            best, best_score = t, score
    return best


def attack_weakest_enemy_planet(state):
    # (1) If we currently have a fleet in flight, abort plan.
    if len(state.my_fleets()) >= 1:
        return False

    # (2) Find my strongest planet.
    strongest_planet = _strongest_my_planet(state)

    # (3) Find the weakest enemy planet.
    weakest_planet = _weakest_enemy_planet(state)

    if not strongest_planet or not weakest_planet:
        # No legal source or destination
        return False
    else:
        # (4) Send half the ships from my strongest planet to the weakest enemy planet.
        return issue_order(state, strongest_planet.ID, weakest_planet.ID, strongest_planet.num_ships / 2)


def spread_to_weakest_neutral_planet(state):
    # (1) If we currently have a fleet in flight, just do nothing.
    if len(state.my_fleets()) >= 1:
        return False

    # (2) Find my strongest planet.
    strongest_planet = _strongest_my_planet(state)

    # (3) Find the weakest neutral planet.
    weakest_planet = _weakest_neutral_planet(state)

    if not strongest_planet or not weakest_planet:
        # No legal source or destination
        return False
    else:
        # (4) Send half the ships from my strongest planet to the weakest enemy planet.
        return issue_order(state, strongest_planet.ID, weakest_planet.ID, strongest_planet.num_ships / 2)


def spread_to_best_neutral_planet(state):
    """
    Expand to a good neutral planet (growth/ships/distance heuristic).
    Sends just enough ships to capture, capped by available ships.
    """
    if len(state.my_fleets()) >= 1:
        return False

    source = _strongest_my_planet(state)
    if not source:
        return False

    target = _best_neutral_planet(state, source)
    if not target:
        return False

    ships_to_send = min(source.num_ships // 2, target.num_ships + 1)
    if ships_to_send <= 0:
        return False
    return issue_order(state, source.ID, target.ID, ships_to_send)


def attack_best_enemy_planet(state):
    """
    Attack an enemy planet that looks profitable.
    Uses a simple travel-time growth estimate for required ships.
    """
    if len(state.my_fleets()) >= 1:
        return False

    source = _strongest_my_planet(state)
    if not source:
        return False

    target = _best_enemy_planet(state, source)
    if not target:
        return False

    d = state.distance(source.ID, target.ID)
    required = target.num_ships + target.growth_rate * d + 1
    if source.num_ships - 1 < required:
        return False
    return issue_order(state, source.ID, target.ID, required)


def reinforce_weakest_my_planet(state):
    """
    Move ships internally: send from strongest planet to weakest planet.
    Useful when we own multiple planets and want to reduce vulnerability.
    """
    if len(state.my_fleets()) >= 1:
        return False

    source = _strongest_my_planet(state)
    dest = _weakest_my_planet(state)
    if not source or not dest or source.ID == dest.ID:
        return False

    # Only reinforce if the gap is meaningful.
    if source.num_ships <= dest.num_ships + 10:
        return False

    ships_to_send = source.num_ships // 3
    if ships_to_send <= 0:
        return False
    return issue_order(state, source.ID, dest.ID, ships_to_send)


def defend_most_threatened_planet(state, horizon=12, max_orders=3):
    """
    If an enemy fleet is inbound, reinforce the planet that is most likely to fall.
    We estimate needed ships at the earliest enemy-arrival within the horizon.
    """
    my_planets = state.my_planets()
    if not my_planets:
        return False

    # Identify the most threatened planet by (needed ships, soonest arrival).
    threatened = None
    threatened_need = 0
    threatened_eta = None
    for p in my_planets:
        enemy_fleets = [f for f in state.enemy_fleets()
                        if f.destination_planet == p.ID and f.turns_remaining <= horizon]
        if not enemy_fleets:
            continue
        eta = min(f.turns_remaining for f in enemy_fleets)
        incoming_enemy = sum(f.num_ships for f in enemy_fleets if f.turns_remaining <= eta)
        incoming_my = sum(f.num_ships for f in state.my_fleets()
                          if f.destination_planet == p.ID and f.turns_remaining <= eta)
        expected_defenders = p.num_ships + p.growth_rate * eta + incoming_my
        need = incoming_enemy - expected_defenders + 1
        if need > threatened_need:
            threatened = p
            threatened_need = need
            threatened_eta = eta

    if threatened is None or threatened_need <= 0:
        return False

    # Send reinforcements from closest planets with surplus.
    orders = 0
    remaining = threatened_need
    donors = [p for p in my_planets if p.ID != threatened.ID]
    donors.sort(key=lambda p: state.distance(p.ID, threatened.ID))

    for donor in donors:
        if orders >= max_orders or remaining <= 0:
            break
        donor_now = state.planets[donor.ID].num_ships
        reserve = _reserve_ships(state, donor.ID, horizon=horizon)
        available = donor_now - reserve
        if available <= 0:
            continue
        send = min(available, remaining)
        if send <= 0:
            continue
        if issue_order(state, donor.ID, threatened.ID, send):
            orders += 1
            remaining -= send

    return orders > 0


def expand_aggressively(state, max_orders=4):
    """
    Capture multiple neutral planets per turn using 'just enough' ships.
    Targets are chosen by a growth/ships/distance heuristic and we avoid double-targeting.
    """
    my_planets = state.my_planets()
    neutrals = state.neutral_planets()
    if not my_planets or not neutrals:
        return False

    already_targeted = {f.destination_planet for f in state.my_fleets()}

    # Score neutrals by value per cost (using nearest distance from any of our planets).
    scored = []
    for n in neutrals:
        if n.ID in already_targeted:
            continue
        nearest_d = min(state.distance(p.ID, n.ID) for p in my_planets)
        score = (n.growth_rate + 1) / ((n.num_ships + 1) * (nearest_d + 1))
        scored.append((score, n, nearest_d))
    scored.sort(key=lambda t: t[0], reverse=True)

    orders = 0
    for _, target, _ in scored:
        if orders >= max_orders:
            break

        # Find best donor: close + has enough surplus for "just enough" capture.
        required = target.num_ships + 1
        best_donor = None
        best_dist = None
        for donor in my_planets:
            donor_now = state.planets[donor.ID].num_ships
            reserve = _reserve_ships(state, donor.ID, horizon=12)
            available = donor_now - reserve
            if available < required:
                continue
            d = state.distance(donor.ID, target.ID)
            if best_donor is None or d < best_dist:
                best_donor, best_dist = donor, d

        if best_donor is None:
            continue

        if issue_order(state, best_donor.ID, target.ID, required):
            orders += 1
            already_targeted.add(target.ID)

    return orders > 0


def attack_opportunistically(state, max_orders=3):
    """
    Attack enemy planets when we can send the full growth-aware required ships.
    """
    my_planets = state.my_planets()
    enemies = state.enemy_planets()
    if not my_planets or not enemies:
        return False

    already_targeted = {f.destination_planet for f in state.my_fleets()}

    # Build all feasible (donor,target) options and pick best few by ROI.
    options = []
    for donor in my_planets:
        donor_now = state.planets[donor.ID].num_ships
        reserve = _reserve_ships(state, donor.ID, horizon=12)
        available = donor_now - reserve
        if available <= 0:
            continue

        for target in enemies:
            if target.ID in already_targeted:
                continue
            d = state.distance(donor.ID, target.ID)
            required = target.num_ships + target.growth_rate * d + 1
            if available < required:
                continue
            score = (target.growth_rate + 1) / (required * (d + 1))
            options.append((score, donor.ID, target.ID, required))

    options.sort(key=lambda t: t[0], reverse=True)

    orders = 0
    used_donors = set()
    for _, donor_id, target_id, required in options:
        if orders >= max_orders:
            break
        if donor_id in used_donors:
            continue
        if target_id in already_targeted:
            continue

        # Re-check availability after earlier orders.
        donor_now = state.planets[donor_id].num_ships
        reserve = _reserve_ships(state, donor_id, horizon=12)
        if donor_now - reserve < required:
            continue

        if issue_order(state, donor_id, target_id, required):
            orders += 1
            used_donors.add(donor_id)
            already_targeted.add(target_id)

    return orders > 0
