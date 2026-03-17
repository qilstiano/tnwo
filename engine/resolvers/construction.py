class ConstructionResolverMixin:
    def _resolve_construction(self):
        """process 'build' intents and progress active construction."""
        # 1. progress active constructions for all countries
        for country in self.world.countries.values():
            for sq in country.land_squares:
                if sq.under_construction_structure:
                    sq.build_time_remaining -= 1
                    if sq.build_time_remaining <= 0:
                        sq.structure = sq.under_construction_structure
                        sq.under_construction_structure = None
                        sq.build_time_remaining = 0

        # 2. process new build intents
        for country_name, actions in self.pending_intents.items():
            country = self.world.countries.get(country_name)
            if not country: continue
            
            for action in actions:
                if action.action_type == "build" and action.structure_type and action.square_id:
                    # find the square
                    target_sq = next((sq for sq in country.land_squares if sq.square_id == action.square_id), None)
                    if target_sq and not target_sq.structure and not target_sq.under_construction_structure:
                        cost = self.config.structure_costs.get(action.structure_type, 0)
                        if country.minerals >= cost:
                            country.minerals -= cost
                            target_sq.under_construction_structure = action.structure_type
                            target_sq.build_time_remaining = self.config.structure_build_times.get(action.structure_type, 1)
