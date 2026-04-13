# Grand Strategy Geopolitical Simulation: Game Mechanics Summary

This document outlines the complete ruleset, mechanics, mathematically derived formulas, and interconnected systems of the simulation engine. The game uses a simultaneous-turn architecture where nations queue actions that are resolved concurrently.

## 1. Core Mechanics & The State Space

At its core, the simulation manages a state composed of sovereign **Nations**. Each nation possesses resources, yields, technologies, and interpersonal diplomatic statuses. 

### 1.1. Resources and Yields
Nations accumulate resources passively per turn and can actively harvest them:
*   **Gold (💰):** Economic currency used for sabotage, investments, and supporting army infrastructure. 
*   **Manpower (🪖):** Represents the population and military recruits available. Spent on strikes and skirmishes.
*   **Production (🏭):** Industry capacity spent when staging military offensives.
*   **Science (🔬):** Used to research technologies.
*   **Civics (🏛️):** Used to adopt civic policies.
*   **Military (⚔️):** Abstract score representing standing army prestige.
*   **Infrastructure Health (🏗️):** HP of the nation. It starts at `100`. If it reaches `0`, the nation is completely **defeated** and collapses.

Every nation starts with asymmetric, randomized base yields:
*   Gold Yield: 10 - 20
*   Manpower Yield: 10 - 20
*   Production Yield: 2 - 8
*   Science & Civic Yields are fixed starting at 5.

### 1.2. The Simultaneous Turn System
Each turn, a nation starts with a maximum of **3 Action Points (AP)**.
1.  **Queueing Actions:** Commands are stored in a queue. A nation can queue up to 3 actions per turn.
2.  **Grace Period:** During turns 1 to 25, no external interactions (war, alliances, skirmishes, trade) are permitted to allow for foundational development.
3.  **Simultaneous Resolution:** At the end of the turn, all queued events are triggered in a specific phase order (Global Events -> Diplomacy -> Economy -> Hostilities -> Strikes -> Defeats/Upkeep).

---

## 2. Economic & Development Mechanics

### 2.1. Passive Yield Generation Formulas
Yields are granted automatically at the end of every turn based on formulas incorporating base yields, annexed yields, and multiplicative modifiers.

*   **Gold Generation =** `(gold_yield + absorbed_gold_yield) * gold_mod`
    *   `gold_mod` = `1.0` + `(0.25 if Golden Age)` + `(0.15 * Number of Active Trade Agreements)` + `(0.10 if gold > 2000)`
*   **Manpower Generation =** `manpower_yield * manpower_mod`
    *   `manpower_mod` = `1.0` + `(0.20 if National Identity)` + `(0.10 if Levies)` - `(0.05 * War Exhaustion level)`. 
    *   *Note: Manpower modifier cannot fall below `0.1`.*
*   **Production Generation =** `(production_yield + absorbed_production_yield) * prod_mod`
    *   `prod_mod` = `1.0` + `(0.25 if Steam Power tech unlocked)`
*   **Science Generation =** `(science_yield + absorbed_sci_yield) * sci_mod`
    *   `sci_mod` = `1.0` + `(0.15 * Number of Active Research Pacts)` + `(0.20 if 3+ Techs unlocked)`
    *   *Also adds `(+ Production // 10)` if Industrial Revolution.*
    *   *Also adds `(+ Manpower // 10)` if Enlightenment.*
*   **Civics Generation =**  `civic_yield`
    *   *(Boosts to Civic yield happen via base updates, e.g., unlocking 3 civics increases base civic yield by 20%)*.

### 2.2. Actions: Harvesting
Nations can use the `HARVEST <Resource>` command to instantly produce a burst of resources.
*   **Gold Harvesting:** Yields `(Gold Yield * 2)`. If *Fertile Lands* achievement unlocked, this output increases by `10%`.
*   **Manpower Harvesting:** Yields `(Manpower Yield * 2)`.
*   **Production Harvesting:** Yields `(Production Yield * 2)`. If *Workshops* achievement unlocked, this output increases by `15%`.
*   **Science & Civics Harvesting:** Provides a flat `25` amount of the respective resource.

### 2.3. Actions: Investments
Nations can use the `INVEST <Target>` command. This always costs **200 Gold** and provides permanent boosts out of native resource generation:
*   **MANPOWER:** +5 Manpower Yield/turn and an immediate +100 Manpower.
*   **INDUSTRY:** +3 Production Yield/turn and an immediate +75 Production.
*   **SCIENCE:** +3 Science Yield/turn and an immediate +75 Science.
*   **CIVICS:** +3 Civic Yield/turn and an immediate +50 Civics.
*   **MILITARY:** Immediate +150 Military score and +50 Manpower.

### 2.4. Science & Civics Systems (Native Exchange)
A nation spends Science and Civics via an underlying continuous-check system. By queuing `RESEARCH <Tech>` or `PURSUE_CIVIC <Civic>`, the engine automatically deducts the necessary currency when the threshold is met at the end of the turn.
*   **Tech Discount:** If a nation unlocks the *Scientific Method* achievement, all Tech costs are multiplied by `0.85`.

---

## 3. Diplomacy Engine

### 3.1. Diplomatic Actions
Pacts strictly require mutual consent. Proposer queues `PROPOSE_X`, Target queues `ACCEPT_X`. 
*   **Alliance:** Upon acceptance, both nations instantly gain **+150 Manpower**.
*   **Trade Agreement:** Upon acceptance, both nations instantly gain **+200 Gold**.
*   **Research Pact:** Upon acceptance, both nations instantly gain **+150 Science**.
*   **Joint Wars:** Nations can coordinate by proposing a joint war (`PROPOSE_JOINT_WAR <Ally> <Enemy>`).

### 3.2. Betrayal & Grievances
Breaking a pact has severe geopolitical consequences governed by a "Grievances" hidden score.
*   **Cancel Alliance:** Target gains **150 Grievance points** against the betrayer.
*   **Shattering an Alliance:** If any nation accumulates **50+ Grievance points** against its ally, the alliance dynamically collapses, instantly voiding all associated trade agreements and research pacts between the two.

---

## 4. Conflict, Covert Ops, and Warfare

### 4.1. Skirmishes & Sabotage (No War Declaration Needed)
These are covert actions that bypass formal war declarations, but cause massive grievances (especially against allies).

**SABOTAGE <Target>**
*   **Cost:** 50 Gold
*   **Effect on Target:** Loses `50` Production, `5` Infrastructure Health, and up to `30` Science.
*   **Effect on Attacker:** Steals the up to `30` Science from the target.
*   **Diplomatic Penalty:** Target gains `+25` Grievances against the attacker. 
*   **Betrayal Penalty:** If target is an Ally, target gains `+150` Grievances, attacker suffers `+50` reputational Grievance penalty.

**SKIRMISH <Target>**
*   **Cost:** 20 Manpower
*   **Effect on Target:** Loses `40` Manpower, `5` Infrastructure Health, and up to `50` Gold.
*   **Effect on Attacker:** Steals the up to `50` Gold from the target.
*   **Diplomatic Penalty:** Target gains `+40` Grievances.
*   **Betrayal Penalty:** If target is an Ally, target gains `+200` Grievances, attacker suffers `+75` reputational Grievance penalty.

### 4.2. Formal Warfare & Military Strikes
To execute strikes, a formal `DECLARE_WAR` action must be executed. This naturally generates `+50` Grievances for all neutral/uninvolved active nations globally, branding the declarer as a Warmonger.

**MILITARY_STRIKE <Target>**
*   **Execution Cost:** `100` Manpower and `50` Production per strike (`40` Production if "Cannons" tech is unlocked).
*   **Tech Tier Multipliers:** Calculate combat advantage `adv`. 
    *   `Attacker Tech Tier` = (Attacker Unlocked Techs count + 1)
    *   `Target Tech Tier` = (Target Unlocked Techs count + 1)
    *   `base_adv` = `Attacker Tech Tier / Target Tech Tier`
    *   `adv` = `base_adv * (1.2 if Attacker has Steel) * (0.8 if Target has Engineering)`
*   **Damage Formulas:** Calculates damage using randomized rolls (0.8 to 1.2 bounds).
    *   `Manpower Damage` = `150 * adv * (0.8 + rand(0, 0.4))`
    *   `Infrastructure Damage (HP)` = `20 * adv * (0.8 + rand(0, 0.4))`
*   **Looting Formulas (Resource Transfer per strike):**
    *   `Stolen Gold` = min(`Infrastructure Damage * 10`, `Target's Max Gold`)
    *   `Stolen Science` = min(`Infrastructure Damage * 2`, `Target's Max Science`)

### 4.3. Annexation & War Exhaustion
When a nation's Infrastructure Health reaches `0` during conflict resolution:
*   The nation collapses and is **Defeated**.
*   The last attacking unit to strike them claims the territory.
*   **Annexation Yield:** The claiming nation permanently adds exactly **50% of the defeated nation's base yield modifiers** to their own generation pool for Gold, Production, and Science. `(absorbed_yield += target.yield * 0.5)`

If a nation is currently at war with an active nation, their **War Exhaustion** ticks up by `+1` per turn. This places a heavy negative multiplier on Manpower generation (up to `-0.05` per Exhaustion point). In peacetime, this score decays by `2` per turn.

---

## 5. Random Global Events
At the start of turn resolution, there is a **20% chance** for a server-wide random event to occur. Examples include:
*   **Plague:** Everyone loses 30 Manpower.
*   **Economic Boom:** Everyone gains 100 Gold.
*   **Natural Disaster:** Everyone loses 15 Infrastructure.
*   **Market Crash:** Everyone loses 50 Gold.

---

## 6. End-Game & Victory Conditions
The game determines winners based on three victory conditions, checked dynamically:

1.  **Domination Victory:** Only one nation remains alive with infrastructure above > 0.
2.  **Peace Victory:** Past Turn 100, if multiple nations survive and **all surviving nations are strictly ALLIED with each other**, they win a shared victory.
3.  **Score Victory:** Past Turn 100, if peace is not achieved, the game is forcefully concluded and a winner is chosen via a summation of overall influence:
    *   `Score Formula` = `Gold + Manpower + Production + (Unlocked Techs * 500) + (Unlocked Civics * 500)`
