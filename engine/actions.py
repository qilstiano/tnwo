import random
from typing import Optional, List, Dict
from core.constants import Resource, DiplomaticState, Tech, Civic
from core.models import Nation
from core.game_state import GameState

class ActionHandler:
    def __init__(self, state: GameState):
        self.state = state
        self.turn_events = [] # stores log strings for the UI to consume

    def queue_action(self, player_id: int, command: str) -> bool:
        """Validates and queues an action for a specific player"""
        nation = self.state.nations[player_id]
        if nation.is_defeated or len(nation.queued_actions) >= nation.max_action_points:
            return False
            
        parts = command.strip().split()
        cmd = parts[0].upper()
        
        # Basic validation
        if cmd == "HARVEST":
            if len(parts) < 2 or parts[1].upper() not in ["GOLD", "MANPOWER", "PRODUCTION", "SCIENCE", "CIVICS"]: return False
        elif cmd in ["DECLARE_WAR", "MILITARY_STRIKE", "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE", "CANCEL_ALLIANCE", "SABOTAGE", "SKIRMISH", "PROPOSE_TRADE", "ACCEPT_TRADE", "PROPOSE_RESEARCH", "ACCEPT_RESEARCH"]:
            if len(parts) < 2: return False
            try: 
                tid = int(parts[1])
                if tid not in self.state.nations or tid == player_id or self.state.nations[tid].is_defeated:
                    return False
            except ValueError:
                return False
                
        elif cmd in ["PROPOSE_JOINT_WAR", "ACCEPT_JOINT_WAR"]:
            if len(parts) < 3: return False
            try:
                tid = int(parts[1]) # ally
                eid = int(parts[2]) # enemy
                if tid not in self.state.nations or tid == player_id or self.state.nations[tid].is_defeated: return False
                if eid not in self.state.nations or eid == player_id or self.state.nations[eid].is_defeated: return False
            except ValueError:
                return False
        
        elif cmd == "INVEST":
            # INVEST <resource> -- must have 200 gold
            if len(parts) < 2 or parts[1].upper() not in ["MANPOWER", "INDUSTRY", "SCIENCE", "CIVICS", "MILITARY"]: return False
            if self.state.nations[player_id].gold < 200: return False
                
        # Grace Period: Block external operations for the configured
        # number of turns (default 25, but can be varied per experiment
        # via GameState.grace_period_turns).
        foreign_cmds = ["DECLARE_WAR", "MILITARY_STRIKE", "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE", "CANCEL_ALLIANCE", "SABOTAGE", "SKIRMISH", "PROPOSE_TRADE", "ACCEPT_TRADE", "PROPOSE_RESEARCH", "ACCEPT_RESEARCH", "PROPOSE_JOINT_WAR", "ACCEPT_JOINT_WAR"]
        grace_turns = getattr(self.state, "grace_period_turns", 25)
        if self.state.turn <= grace_turns and cmd in foreign_cmds:
            return False

        nation.queued_actions.append(command.strip())
        nation.action_points -= 1
        return True
        
    def cancel_last_action(self, player_id: int) -> bool:
        nation = self.state.nations[player_id]
        if nation.queued_actions:
            nation.queued_actions.pop()
            nation.action_points += 1
            return True
        return False

    def resolve_simultaneous_turn(self):
        """Processes all queued actions across all nations simultaneously"""
        self.turn_events = []
        self.turn_events.append(f"--- RESOLVING TURN {self.state.turn} ---")

        # 0. Global Random Events
        if random.random() < 0.20:
            events = [
                ("🌍 GLOBAL PLAGUE: All nations lose 30 Manpower.", lambda n: setattr(n, 'manpower', max(0, n.manpower - 30))),
                ("🌍 ECONOMIC BOOM: Trade routes flourish! +100 Gold.", lambda n: setattr(n, 'gold', n.gold + 100)),
                ("🌍 SCIENTIFIC BREAKTHROUGH: +50 Science.", lambda n: setattr(n, 'science', n.science + 50)),
                ("🌍 NATURAL DISASTER: Infrastructure damaged across the continent.", lambda n: setattr(n, 'infrastructure_health', max(10, n.infrastructure_health - 15))),
                ("🌍 RENAISSANCE: +50 Civics.", lambda n: setattr(n, 'civics', n.civics + 50)),
                ("🌍 MARKET CRASH: Global recession drains treasuries by 50 Gold.", lambda n: setattr(n, 'gold', max(0, n.gold - 50))),
                ("🌍 IDEOLOGICAL SHIFT: Widespread protests harm production.", lambda n: setattr(n, 'production', max(0, n.production - 30))),
                ("🌍 GOLDEN HARVEST: Bountiful yields grant +50 Manpower.", lambda n: setattr(n, 'manpower', n.manpower + 50)),
                ("🌍 CYBER WARFARE PANIC: Random intellectual property theft reduces Science.", lambda n: setattr(n, 'science', max(0, n.science - 20))),
                ("🌍 ARMS RACE: Military industry subsidies grant +50 Production.", lambda n: setattr(n, 'production', n.production + 50))
            ]
            chosen_event = random.choice(events)
            self.turn_events.append(chosen_event[0])
            for n in self.state.nations.values():
                if not n.is_defeated:
                    chosen_event[1](n)
        
        # 1. Diplomacy (Alliances & Pacts)
        alliance_proposals = []
        trade_proposals = []
        research_proposals = []
        joint_war_proposals = []
        
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for cmd in n.queued_actions:
                if cmd.startswith("PROPOSE_ALLIANCE"):
                    target = int(cmd.split()[1])
                    alliance_proposals.append((n.id, target))
                    self.turn_events.append(f"{n.name} proposed an Alliance to {self.state.nations[target].name}.")
                elif cmd.startswith("PROPOSE_TRADE"):
                    target = int(cmd.split()[1])
                    trade_proposals.append((n.id, target))
                    self.turn_events.append(f"{n.name} proposed a Trade Agreement to {self.state.nations[target].name}.")
                elif cmd.startswith("PROPOSE_RESEARCH"):
                    target = int(cmd.split()[1])
                    research_proposals.append((n.id, target))
                    self.turn_events.append(f"{n.name} proposed a Research Pact to {self.state.nations[target].name}.")
                elif cmd.startswith("PROPOSE_JOINT_WAR"):
                    parts = cmd.split()
                    target = int(parts[1])
                    enemy = int(parts[2])
                    joint_war_proposals.append((n.id, target, enemy))
                    self.turn_events.append(f"{n.name} requested {self.state.nations[target].name} to join them in a war against {self.state.nations[enemy].name}!")
                    
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for cmd in n.queued_actions:
                if cmd.startswith("ACCEPT_ALLIANCE"):
                    target = int(cmd.split()[1])
                    if (target, n.id) in alliance_proposals or self.state.get_diplomatic_state(n.id, target) == DiplomaticState.ALLIANCE_PENDING:
                        self.state.set_diplomatic_state(n.id, target, DiplomaticState.ALLIED)
                        n.achievements.alliances_formed += 1
                        self.state.nations[target].achievements.alliances_formed += 1
                        n.manpower += 150
                        self.state.nations[target].manpower += 150
                        self.turn_events.append(f"{n.name} and {self.state.nations[target].name} formed an Alliance! (+150🪖 mobilization)")
                
                elif cmd.startswith("ACCEPT_TRADE"):
                    target = int(cmd.split()[1])
                    if (target, n.id) in trade_proposals or target in n.pending_trade_agreements:
                        if target not in n.active_trade_agreements:
                            n.active_trade_agreements.append(target)
                            self.state.nations[target].active_trade_agreements.append(n.id)
                            # Remove from pending if applicable
                            if target in n.pending_trade_agreements: n.pending_trade_agreements.remove(target)
                            n.gold += 200
                            self.state.nations[target].gold += 200
                            self.turn_events.append(f"{n.name} and {self.state.nations[target].name} signed a Trade Agreement (+200💰 / +15% / t)!")
                
                elif cmd.startswith("ACCEPT_RESEARCH"):
                    target = int(cmd.split()[1])
                    if (target, n.id) in research_proposals or target in n.pending_research_pacts:
                        if target not in n.active_research_pacts:
                            n.active_research_pacts.append(target)
                            self.state.nations[target].active_research_pacts.append(n.id)
                            # Remove from pending if applicable
                            if target in n.pending_research_pacts: n.pending_research_pacts.remove(target)
                            n.science += 150
                            self.state.nations[target].science += 150
                            self.turn_events.append(f"{n.name} and {self.state.nations[target].name} signed a Research Pact (+150🔬 / +15% / t)!")
                
                elif cmd.startswith("ACCEPT_JOINT_WAR"):
                    parts = cmd.split()
                    proposer = int(parts[1])
                    enemy = int(parts[2])
                    
                    valid = False
                    if (proposer, n.id, enemy) in joint_war_proposals:
                         valid = True
                    else:
                         for p in n.pending_joint_wars:
                              if p["proposer"] == proposer and p["enemy"] == enemy:
                                  valid = True
                                  n.pending_joint_wars.remove(p)
                                  break
                    
                    if valid:
                         self.state.set_diplomatic_state(n.id, enemy, DiplomaticState.WAR)
                         self.state.set_diplomatic_state(proposer, enemy, DiplomaticState.WAR)
                         self.turn_events.append(f"JOINT WAR: {n.name} answered {self.state.nations[proposer].name}'s call and declared war on {self.state.nations[enemy].name}!")
                            
        # Store unmet proposals as pending
        for proposer, target in alliance_proposals:
            if self.state.get_diplomatic_state(proposer, target) != DiplomaticState.ALLIED:
                self.state.set_diplomatic_state(proposer, target, DiplomaticState.ALLIANCE_PENDING)
                
        for proposer, target in trade_proposals:
            if target not in self.state.nations[proposer].active_trade_agreements:
                if proposer not in self.state.nations[target].pending_trade_agreements:
                    self.state.nations[target].pending_trade_agreements.append(proposer)

        for proposer, target in research_proposals:
            if target not in self.state.nations[proposer].active_research_pacts:
                if proposer not in self.state.nations[target].pending_research_pacts:
                    self.state.nations[target].pending_research_pacts.append(proposer)
                    
        for proposer, target, enemy in joint_war_proposals:
            if self.state.get_diplomatic_state(target, enemy) != DiplomaticState.WAR:
                already_pending = any(p["proposer"] == proposer and p["enemy"] == enemy for p in self.state.nations[target].pending_joint_wars)
                if not already_pending:
                    self.state.nations[target].pending_joint_wars.append({"proposer": proposer, "enemy": enemy})
        
        # 2. Economy & Development (Harvest, Research, Civic)
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for cmd in n.queued_actions:
                parts = cmd.split()
                if parts[0] == "HARVEST":
                    res = parts[1].upper()
                    base_yield = 0
                    
                    if res == "GOLD":
                        base_yield = n.gold_yield * 2
                        if n.achievements.fertile_lands: base_yield = int(base_yield * 1.1)
                        n.gold += base_yield
                        n.achievements.harvest_gold_count += 1
                        n.achievements.total_gold_earned += base_yield
                        if n.achievements.harvest_gold_count >= 5: n.achievements.fertile_lands = True
                        if n.achievements.total_gold_earned >= 2000: n.achievements.trade_routes = True
                        
                    elif res == "MANPOWER":
                        base_yield = n.manpower_yield * 2
                        n.manpower += base_yield
                        n.achievements.draft_conscripts_count += 1
                        if n.achievements.draft_conscripts_count >= 5: n.achievements.levies = True
                        
                    elif res == "PRODUCTION":
                        base_yield = n.production_yield * 2
                        if n.achievements.workshops: base_yield = int(base_yield * 1.15)
                        n.production += base_yield
                        n.achievements.mobilize_industry_count += 1
                        n.achievements.total_production_earned += base_yield
                        if n.achievements.mobilize_industry_count >= 5: n.achievements.workshops = True
                        if n.achievements.total_production_earned >= 2000: n.achievements.factory_system = True
                        
                    elif res == "SCIENCE":
                        n.science += 25
                        n.achievements.fund_academies_count += 1
                        if n.achievements.fund_academies_count >= 5: n.achievements.literacy = True
                        
                    elif res == "CIVICS":
                        n.civics += 25
                        n.achievements.civics_generated += 25
                        if n.achievements.civics_generated >= 100: n.achievements.common_law = True

                    self.turn_events.append(f"{n.name} harvested {res}.")

                elif parts[0] == "INVEST":
                    target_res = parts[1].upper()
                    cost = 200
                    n.gold -= cost
                    if target_res == "MANPOWER":
                        n.manpower_yield += 5
                        n.manpower += 100
                        self.turn_events.append(f"{n.name} invested in population growth (+5🪖/t, +100🪖 now).")
                    elif target_res == "INDUSTRY":
                        n.production_yield += 3
                        n.production += 75
                        self.turn_events.append(f"{n.name} built new factories (+3🏭/t, +75🏭 now).")
                    elif target_res == "SCIENCE":
                        n.science_yield += 3
                        n.science += 75
                        self.turn_events.append(f"{n.name} funded academies (+3🔬/t, +75🔬 now).")
                    elif target_res == "CIVICS":
                        n.civic_yield += 3
                        n.civics += 50
                        self.turn_events.append(f"{n.name} reformed its institutions (+3🏛️/t, +50🏛️ now).")
                    elif target_res == "MILITARY":
                        n.military += 150
                        n.manpower += 50
                        self.turn_events.append(f"{n.name} expanded its standing army (+150⚔️, +50🪖).")

                elif parts[0] == "RESEARCH":
                    tech = " ".join(parts[1:])
                    n.current_tech = tech
                    self.turn_events.append(f"{n.name} began researching {tech}.")
                    
                elif parts[0] == "PURSUE_CIVIC":
                    civic = " ".join(parts[1:])
                    n.current_civic = civic
                    self.turn_events.append(f"{n.name} began pursuing {civic}.")

        # 3. War Declarations
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for cmd in n.queued_actions:
                if cmd.startswith("DECLARE_WAR"):
                    target = int(cmd.split()[1])
                    if self.state.get_diplomatic_state(n.id, target) != DiplomaticState.WAR:
                        self.state.set_diplomatic_state(n.id, target, DiplomaticState.WAR)
                        self.turn_events.append(f"WAR! {n.name} declared war on {self.state.nations[target].name}!")
                        
                        # Generate Grievances globally
                        for oid, onat in self.state.nations.items():
                            if oid != n.id and oid != target and not onat.is_defeated:
                                onat.grievances[n.id] = onat.grievances.get(n.id, 0) + 50

        # 3.5 Covert Ops and Skirmishes
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for cmd in n.queued_actions:
                if cmd.startswith("CANCEL_ALLIANCE"):
                    target = int(cmd.split()[1])
                    if self.state.get_diplomatic_state(n.id, target) == DiplomaticState.ALLIED:
                        self.state.set_diplomatic_state(n.id, target, DiplomaticState.NEUTRAL)
                        self.state.nations[target].grievances[n.id] = self.state.nations[target].grievances.get(n.id, 0) + 150
                        self.turn_events.append(f"BETRAYAL: {n.name} broke their Alliance with {self.state.nations[target].name}!")

                elif cmd.startswith("SABOTAGE"):
                    target = int(cmd.split()[1])
                    cost_gold = 50
                    is_ally = self.state.get_diplomatic_state(n.id, target) == DiplomaticState.ALLIED
                    if n.gold >= cost_gold:
                        n.gold -= cost_gold
                        t_nat = self.state.nations[target]
                        stolen_sci = min(t_nat.science, 30)
                        t_nat.science -= stolen_sci
                        n.science += stolen_sci
                        t_nat.production = max(0, t_nat.production - 50)
                        t_nat.infrastructure_health = max(0, t_nat.infrastructure_health - 5)
                        # Allied betrayal: massively inflated penalties
                        if is_ally:
                            t_nat.grievances[n.id] = t_nat.grievances.get(n.id, 0) + 150
                            n.grievances[target] = n.grievances.get(target, 0) + 50  # reputation cost
                            self.turn_events.append(f"BETRAYAL: {n.name} sabotaged their ally {t_nat.name}! (+150 grievances)")
                        else:
                            t_nat.grievances[n.id] = t_nat.grievances.get(n.id, 0) + 25
                            self.turn_events.append(f"SHADOW WAR: {n.name} sabotaged {t_nat.name}'s industry and stole {stolen_sci}🔬!")
                            
                elif cmd.startswith("SKIRMISH"):
                    target = int(cmd.split()[1])
                    cost_mp = 20
                    is_ally = self.state.get_diplomatic_state(n.id, target) == DiplomaticState.ALLIED
                    if n.manpower >= cost_mp:
                        n.manpower -= cost_mp
                        t_nat = self.state.nations[target]
                        t_nat.manpower = max(0, t_nat.manpower - 40)
                        stolen_gold = min(t_nat.gold, 50)
                        t_nat.gold -= stolen_gold
                        n.gold += stolen_gold
                        t_nat.infrastructure_health = max(0, t_nat.infrastructure_health - 5)
                        # Allied betrayal: massively inflated penalties
                        if is_ally:
                            t_nat.grievances[n.id] = t_nat.grievances.get(n.id, 0) + 200
                            n.grievances[target] = n.grievances.get(target, 0) + 75  # own rep hurt too
                            self.turn_events.append(f"BETRAYAL: {n.name} raided their ally {t_nat.name}'s border! (+200 grievances)")
                        else:
                            t_nat.grievances[n.id] = t_nat.grievances.get(n.id, 0) + 40
                            self.turn_events.append(f"BORDER CONFLICT: {n.name}'s forces skirmished with {t_nat.name}! Looted {stolen_gold}💰.")

        # 4. Military Strikes (Simultaneous Damage)
        # Calculate all damage first so nations don't die mid-resolution preventing their own strikes
        damages = {} # target_id -> list of (attacker_name, manpower_dmg, infra_dmg)
        
        for n in self.state.nations.values():
            if n.is_defeated: continue
            strike_count = sum(1 for c in n.queued_actions if c.startswith("MILITARY_STRIKE"))
            if strike_count == 0: continue
            
            for cmd in n.queued_actions:
                if cmd.startswith("MILITARY_STRIKE"):
                    target = int(cmd.split()[1])
                    if self.state.get_diplomatic_state(n.id, target) != DiplomaticState.WAR:
                        continue
                        
                    # deduct cost
                    cost_manpower = 100
                    cost_prod = 50
                    if "Cannons" in n.unlocked_techs: cost_prod -= 10
                    
                    if n.manpower >= cost_manpower and n.production >= cost_prod:
                        n.manpower -= cost_manpower
                        n.production -= cost_prod
                        
                        target_nat = self.state.nations[target]
                        
                        tier = len(n.unlocked_techs) + 1
                        target_tier = len(target_nat.unlocked_techs) + 1
                        adv = tier / target_tier
                        if "Steel" in n.unlocked_techs: adv *= 1.2
                        if "Engineering" in target_nat.unlocked_techs: adv *= 0.8
                        
                        dmg_mp = int(150 * adv * (0.8 + random.random() * 0.4))
                        dmg_in = int(20 * adv * (0.8 + random.random() * 0.4))
                        
                        if target not in damages: damages[target] = []
                        damages[target].append((n, dmg_mp, dmg_in))
                        
        # Apply damages
        for tid, incoming in damages.items():
            target = self.state.nations[tid]
            for attacker, d_mp, d_in in incoming:
                target.manpower -= d_mp
                target.infrastructure_health -= d_in
                
                # Looting Calculation
                stolen_gold = min(int(d_in * 10), max(0, target.gold))
                stolen_sci = min(int(d_in * 2), max(0, target.science))
                
                target.gold -= stolen_gold
                target.science -= stolen_sci
                attacker.gold += stolen_gold
                attacker.science += stolen_sci
                
                self.turn_events.append(f"{attacker.name} struck {target.name}! Lost {d_mp}🪖 / {d_in}%🏭. Looted {stolen_gold}💰 / {stolen_sci}🔬!")
                attacker.achievements.military_strikes_won += 1
                if attacker.achievements.military_strikes_won >= 3:
                    attacker.achievements.war_college = True
                
        # Resolve deaths and Annexation
        for tid, incoming in damages.items():
            target = self.state.nations[tid]
            if target.infrastructure_health <= 0 and not target.is_defeated:
                target.is_defeated = True
                target.infrastructure_health = 0
                target.manpower = 0
                
                killer = incoming[-1][0] # Last attacker gets the claim
                abs_g = int(target.gold_yield * 0.5)
                abs_p = int(target.production_yield * 0.5)
                abs_s = int(target.science_yield * 0.5)
                
                killer.absorbed_gold_yield += abs_g
                killer.absorbed_prod_yield += abs_p
                killer.absorbed_sci_yield += abs_s
                
                self.turn_events.append(f"*** {target.name} HAS FALLEN! {killer.name} annexed their territory (+{abs_g}💰/t, +{abs_p}🏭/t) ***")

        # 6. Alliance and Treaty Breakage from Grievances
        for n in self.state.nations.values():
            if n.is_defeated: continue
            for target_id, target in self.state.nations.items():
                if target.is_defeated or n.id == target_id: continue
                
                grievances = target.grievances.get(n.id, 0)
                if grievances >= 50 and self.state.get_diplomatic_state(n.id, target_id) == DiplomaticState.ALLIED:
                    self.state.set_diplomatic_state(n.id, target_id, DiplomaticState.NEUTRAL)
                    if target_id in n.active_trade_agreements:
                        n.active_trade_agreements.remove(target_id)
                    if n.id in target.active_trade_agreements:
                        target.active_trade_agreements.remove(n.id)
                    if target_id in n.active_research_pacts:
                        n.active_research_pacts.remove(target_id)
                    if n.id in target.active_research_pacts:
                        target.active_research_pacts.remove(n.id)
                        
                    self.turn_events.append(f"ALLIANCE SHATTERED: {n.name}'s hostility has exhausted {target.name}'s patience!")

        # 6.5 Failed State Collapse Check
        for n in self.state.nations.values():
            if not n.is_defeated and n.infrastructure_health <= 0:
                n.is_defeated = True
                n.infrastructure_health = 0
                n.manpower = 0
                self.turn_events.append(f"*** {n.name} HAS COLLAPSED! Continual devastation has fractured the nation. ***")

        # 7. Clean up and Next Turn
        self.state.turn += 1
        self.state.process_turn_updates()
        
        return self.turn_events