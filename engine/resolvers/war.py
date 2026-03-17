class WarResolverMixin:
    def _resolve_war_and_sabotage(self):
        # gather war intents
        wars = []
        for country_name, actions in self.pending_intents.items():
            for action in actions:
                if action.action_type == "war" and action.target_country:
                    wars.append((country_name, action.target_country, action.amount))

        for attacker_name, defender_name, minerals_spent in wars:
            attacker = self.world.countries.get(attacker_name)
            defender = self.world.countries.get(defender_name)
            if not attacker or not defender:
                continue

            attacker.minerals = max(0.0, attacker.minerals - minerals_spent)

            # calculate Power: Power = (P * T * I) + M_spent
            infra_a = sum(1 for sq in attacker.land_squares if sq.structure)
            infra_d = sum(1 for sq in defender.land_squares if sq.structure)
            
            power_a = (attacker.population * max(1.0, attacker.tech_level) * max(1, infra_a)) + minerals_spent
            power_d = (defender.population * max(1.0, defender.tech_level) * max(1, infra_d)) # basic defense power

            # calculate attrition using Lanchester's approach
            attrition_rate = 0.1
            p_loss_a = int(power_d * attrition_rate)
            p_loss_d = int(power_a * attrition_rate)
            
            # apply population losses
            attacker.population = max(0, attacker.population - p_loss_a)
            defender.population = max(0, defender.population - p_loss_d)
            
            # destroy some infrastructure based on intensity
            if attacker.population > 0:
                destroy_chance_a = p_loss_a / float(attacker.population + p_loss_a)
                for sq in attacker.land_squares:
                    if sq.structure and destroy_chance_a > 0.1:
                        sq.structure = None
            if defender.population > 0:
                destroy_chance_d = p_loss_d / float(defender.population + p_loss_d)
                for sq in defender.land_squares:
                    if sq.structure and destroy_chance_d > 0.1:
                        sq.structure = None

            # determine winner and transfer resources
            if power_a > power_d:
                stolen_m = defender.minerals * 0.2
                stolen_e = defender.energy * 0.2
                defender.minerals -= stolen_m
                defender.energy -= stolen_e
                attacker.minerals += stolen_m
                attacker.energy += stolen_e
            elif power_d > power_a:
                stolen_m = attacker.minerals * 0.2
                stolen_e = attacker.energy * 0.2
                attacker.minerals -= stolen_m
                attacker.energy -= stolen_e
                defender.minerals += stolen_m
                defender.energy += stolen_e
