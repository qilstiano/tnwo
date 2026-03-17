from engine.models import CountryState, StructureType, LandType

class ProductionResolverMixin:
    def _resolve_production_and_consumption(self, country: CountryState):
        """calculate and apply resource changes."""
        # production
        total_food_produced = 0.0
        total_energy_produced = 0.0
        total_minerals_produced = 0.0
        
        for sq in country.land_squares:
            if not sq.structure:
                continue
                
            rates = self.config.production_rates.get(sq.structure, {})
            
            # apply land type bonuses (simple logic: +50% if matching)
            multiplier = 1.0 + (0.5 * sq.improvement_level)
            if sq.structure == StructureType.FARM and sq.land_type == LandType.AGRICULTURAL:
                multiplier *= 1.5
            elif sq.structure == StructureType.POWER_PLANT and sq.land_type == LandType.ENERGY:
                multiplier *= 1.5
            elif sq.structure == StructureType.MINE and sq.land_type == LandType.MINERAL:
                multiplier *= 1.5

            total_food_produced += rates.get("food", 0.0) * multiplier
            total_energy_produced += rates.get("energy", 0.0) * multiplier
            total_minerals_produced += rates.get("minerals", 0.0) * multiplier

        # consumption
        food_consumed = country.population * self.config.food_per_capita
        energy_consumed = country.population * self.config.energy_per_capita
        
        # apply net changes
        country.food += total_food_produced - food_consumed
        country.energy += total_energy_produced - energy_consumed
        country.minerals += total_minerals_produced
        
        # starvation mechanics
        if country.food < 0:
            country.population -= int(country.population * self.config.malthusian_decay)
            country.food = 0
            
        # wealth mechanics W_gen = (P * T * I) * min(1, E_avail / E_req)
        # we simplify infra to number of structures
        infra = sum(1 for sq in country.land_squares if sq.structure)
        energy_required_for_wealth = infra * 10  # arbitrary logic
        energy_available = max(0.0, country.energy)
        energy_ratio = min(1.0, energy_available / max(1.0, energy_required_for_wealth))
        
        wealth_generated = (country.population * max(1.0, country.tech_level) * max(1, infra)) * energy_ratio * 0.01
        country.wealth += wealth_generated
        
        # tech growth mechanics (simple)
        # assuming each school produces 0.1 tech points, each lab 0.5
        tech_growth = sum(0.1 for sq in country.land_squares if sq.structure == StructureType.SCHOOL)
        tech_growth += sum(0.5 for sq in country.land_squares if sq.structure == StructureType.RESEARCH_LAB)
        country.tech_level += (tech_growth * (1 - country.tech_level / 100)) # approaching asymptotic limit 100
        
        # military power update
        base_military_power = sum(10 for sq in country.land_squares if sq.structure == StructureType.MILITARY_BASE)
        base_military_power *= max(1.0, country.tech_level * 0.5)
        country.military_power = base_military_power * energy_ratio
