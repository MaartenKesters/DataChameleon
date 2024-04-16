from plugin import Plugin
from protectionLevel import ProtectionLevel
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator

from synthcity.plugins.core.dataloader import DataLoader

class GeneratorCreator():
    def __init__(self, private: DataLoader, plugin_class: Plugin, privacy_calc: PrivacyCalculator, utility_calc: UtilityCalculator):
        self.size = 1000
        self.plugin_class = plugin_class
        self.private_data = private
        self.privacy_calc = privacy_calc
        self.utility_calc = utility_calc

    def create_generator(self, generators, protection_level: ProtectionLevel) -> Plugin:
        self.protection_level = protection_level
        ## if protection level is specified by an epsilon value, we can easily create the right generator
        if protection_level.epsilon is not None:
            self.generator = self.plugin_class(protection_level=protection_level, epsilon=protection_level.epsilon)
            self.generator.fit(self.private_data)
            return self.generator
        ## Find the generator with the most similar protection level to use as a starting point
        self.generator = self.find_initial_generator(generators, protection_level)
        if self.generator is None:
            ## Create new starting point if no other generators exist
            self.generator = self.plugin_class(protection_level=protection_level, epsilon=1)
            self.generator.fit(self.private_data)

        ## Generate synthetic data as the starting point to check the privacy/utility
        syn = self.generator.generate(count=self.size)
        
        ## Adapt generator untill requirements are met
        self.eps_high_bound = 10
        self.eps_low_bound = 0.1

        satisfied = False

        while not satisfied:

            ## Fine tune privacy
            ## Calculate the current privacy to use during fine tuning
            self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)

            if protection_level.privacy_metric is not None:
                priv_satisfied = False
                while not priv_satisfied:
                    ## Calculate the privacy with given metric for requirement
                    val = protection_level.privacy_metric.calculate(self.private_data, syn)
                    print('calc req: ' + str(val))

                    if protection_level.privacy_metric.satisfied(protection_level.privacy_value, val, protection_level.range):
                        priv_satisfied = True
                        break

                    ## Check how much the privacy has to change
                    amount = protection_level.privacy_metric.amount(protection_level.privacy_value, val)

                    ## Check if we need to increase or decrease privacy
                    if protection_level.privacy_metric.privacy() == 1:
                        if val < protection_level.privacy_value:
                            new = self.increase_privacy(syn, amount)
                        else:
                            new = self.decrease_privacy(syn, amount)
                    else:
                        if val > protection_level.privacy_value:
                            new = self.increase_privacy(syn, amount)
                        else:
                            new = self.decrease_privacy(syn, amount)

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        print('Automatically creating a generator for this protection level did not work. try to add it manually.')
                        return None

            ## Fine tune utility
            ## Calculate the current utility to use during fine tuning
            self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

            if protection_level.utility_metric is not None:
                util_satisfied = False
                while not util_satisfied:
                    ## Calculate the utility
                    val = protection_level.utility_metric.calculate(self.private_data, syn)
                    print('calc req: ' + str(val))

                    if protection_level.utility_metric.satisfied(protection_level.utility_value, val, protection_level.range):
                        util_satisfied = True
                        break
                        
                    ## Check how much the utility has to change
                    amount = protection_level.utility_metric.amount(protection_level.utility_value, val)
                    
                    ## Check if we need to increase or decrease utility
                    if protection_level.utility_metric.utility() == 1:
                        if val < protection_level.utility_value:
                            new = self.increase_utility(syn, amount)
                        else:
                            new = self.decrease_utility(syn, amount)
                    else:
                        if val > protection_level.utility_value:
                            new = self.increase_utility(syn, amount)
                        else:
                            new = self.decrease_utility(syn, amount)

                    ## If new is None, no improvements are made
                    if new is not None:
                        syn = new
                    else:
                        print('Automatically creating a generator for this protection level did not work. try to add it manually, slightly adapt the protection level or use other generators.')
                        return None

            ## Calculate the privacy and utility, check if both requirements are still met
            print('final check')
            priv_val = protection_level.privacy_metric.calculate(self.private_data, syn)
            util_val = protection_level.utility_metric.calculate(self.private_data, syn)
            print('calc priv req: ' + str(priv_val))
            print('calc util req: ' + str(util_val))
            if protection_level.privacy_metric.satisfied(protection_level.privacy_value, priv_val, protection_level.range) and protection_level.privacy_metric.satisfied(protection_level.utility_value, util_val, protection_level.range):
                return self.generator

    def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase privacy')
        ## set current generator eps as high bound
        self.eps_high_bound = self.generator.get_dp_epsilon()

        if self.eps_low_bound >= self.eps_high_bound:
            ## No improvements can be made
            return None
        
        ## Increase privacy by decreasing epsilon
        new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_low_bound) * amount), 1))

        print(new_generator.get_dp_epsilon())

        new_generator.fit(self.private_data)
        
        new_syn = new_generator.generate(count=self.size)

        ## Check if the privacy increased
        new_privacy = self.privacy_calc.calculatePrivacy(self.private_data, new_syn)
        if new_privacy > self.current_privacy:
            self.current_privacy = new_privacy
            self.generator = new_generator
            return new_syn
        else:
            ## Increasing the privacy did not work
            return None
        
    def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease privacy')
        ## set current generator eps as low bound
        self.eps_low_bound = self.generator.get_dp_epsilon()

        if self.eps_low_bound >= self.eps_high_bound:
            ## No improvements can be made
            return None
        
        ## Decrease privacy by increasing epsilon
        new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() + ((self.eps_high_bound - self.generator.get_dp_epsilon()) * amount), 1))

        print(new_generator.get_dp_epsilon())

        new_generator.fit(self.private_data)
        
        new_syn = new_generator.generate(count=self.size)

        ## Check if the privacy decreased
        new_privacy = self.privacy_calc.calculatePrivacy(self.private_data, new_syn)
        if new_privacy < self.current_privacy:
            self.current_privacy = new_privacy
            self.generator = new_generator
            return new_syn
        else:
            ## Decreasing the privacy did not work
            return None

    def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Increase utility')
        ## set current generator eps as low bound
        self.eps_low_bound = self.generator.get_dp_epsilon()

        if self.eps_low_bound >= self.eps_high_bound:
            ## No improvements can be made
            return None
        
        ## Increase utility by increasing epsilon
        new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() + ((self.eps_high_bound - self.generator.get_dp_epsilon()) * amount), 1))

        print(new_generator.get_dp_epsilon())

        new_generator.fit(self.private_data)
        
        new_syn = new_generator.generate(count=self.size)

        ## Check if the utility increased
        new_utility = self.utility_calc.calculateUtility(self.private_data, new_syn)
        if new_utility > self.current_utility:
            self.current_utility = new_utility
            self.generator = new_generator
            return new_syn
        else:
            ## Increasing the utility did not work
            return None

    def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
        print('Decrease utility')
        ## set current generator eps as low bound
        self.eps_high_bound = self.generator.get_dp_epsilon()

        if self.eps_high_bound <= self.eps_low_bound:
            ## No improvements can be made
            return None
        
        ## Decrease utility by decreasing epsilon
        new_generator = self.set_new_eps(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_low_bound) * amount))

        print(new_generator.get_dp_epsilon())

        new_generator.fit(self.private_data)

        new_syn = new_generator.generate(count=self.size)

        ## Check if the utility decreased
        new_utility = self.utility_calc.calculateUtility(self.private_data, new_syn)
        if new_utility < self.current_utility:
            self.current_utility = new_utility
            self.generator = new_generator
            return new_syn
        else:
            ## Decreasing the utility did not work
            return None
    
        
    def set_new_eps(self, eps: float) -> Plugin:
        new_generator = self.plugin_class(protection_level=self.protection_level, epsilon=eps)
        return new_generator

    def find_initial_generator(self, generators, protection_level: ProtectionLevel) -> Plugin:
        closest_generator = None
        difference = 0
        for level, generator in generators.items():
            if level == protection_level.name and level == protection_level.name:
                if closest_generator is None:
                    closest_generator = generator
                    privacy_diff = level.privacy_value
                    utility_diff = level.utility_value
                    difference = (privacy_diff + utility_diff) / 2
                else:
                    priv_diff = abs(level.privacy_value - protection_level.privacy_value)
                    util_diff = abs(level.utility_value - protection_level.utility_value)
                    diff = (priv_diff + util_diff) / 2
                    if diff < difference:
                        closest_generator = generator
                        difference = difference
        return closest_generator
    

    # def create_generator(self, generators, data_requirement: DataRequirement) -> Plugin:
    #     ## Find the generator with the most similar requirements to use as a starting point
    #     self.generator = self.find_initial_generator(generators, data_requirement)
    #     if self.generator is None:
    #         ## Create new starting point if no other generators exist
    #         self.generator = DPGANPlugin(epsilon=1)

    #     ## Generate synthetic data as the starting point to check the requirements
    #     syn = self.generator.generate(count=self.size)
        
    #     ## Adapt generator untill requirements are met
    #     self.eps_high_priv_bound = 0.1
    #     self.eps_low_priv_bound = 10
    #     self.eps_high_util_bound = 10
    #     self.eps_low_util_bound = 0.1

    #     satisfied = False

    #     while not satisfied:

    #         ## Fine tune privacy
    #         ## Calculate the current privacy to use during fine tuning
    #         self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)

    #         priv_satisfied = False
    #         while not priv_satisfied:
    #             ## Calculate the privacy with given metric for requirement
    #             val = data_requirement.privacy_metric.calculate(self.private_data, syn)
    #             print('calc req: ' + str(val))

    #             if data_requirement.privacy_metric.satisfied(data_requirement.privacy_value, val, data_requirement.range):
    #                 priv_satisfied = True
    #                 break

    #             ## Check how much the privacy has to change
    #             amount = data_requirement.privacy_metric.amount(data_requirement.privacy_value, val)

    #             ## Check if we need to increase or decrease privacy
    #             if data_requirement.privacy_metric.privacy() == 1:
    #                 if val < data_requirement.privacy_value:
    #                     new = self.increase_privacy(syn, amount)
    #                 else:
    #                     new = self.decrease_privacy(syn, amount)
    #             else:
    #                 if val > data_requirement.privacy_value:
    #                     new = self.increase_privacy(syn, amount)
    #                 else:
    #                     new = self.decrease_privacy(syn, amount)

    #             ## If new is None, no improvements are made
    #             if new is not None:
    #                 syn = new
    #             else:
    #                 break

    #         ## Fine tune utility
    #         ## Calculate the current utility to use during fine tuning
    #         self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

    #         util_satisfied = False
    #         while not util_satisfied:
    #             ## Calculate the utility
    #             val = data_requirement.utility_metric.calculate(self.private_data, syn)
    #             print('calc req: ' + str(val))

    #             if data_requirement.utility_metric.satisfied(data_requirement.utility_value, val, data_requirement.range):
    #                 util_satisfied = True
    #                 break
                    
    #             ## Check how much the utility has to change
    #             amount = data_requirement.utility_metric.amount(data_requirement.utility_value, val)
                
    #             ## Check if we need to increase or decrease utility
    #             if data_requirement.utility_metric.utility() == 1:
    #                 if val < data_requirement.utility_value:
    #                     new = self.increase_utility(syn, amount)
    #                 else:
    #                     new = self.decrease_utility(syn, amount)
    #             else:
    #                 if val > data_requirement.utility_value:
    #                     new = self.increase_utility(syn, amount)
    #                 else:
    #                     new = self.decrease_utility(syn, amount)

    #             ## If new is None, no improvements are made
    #             if new is not None:
    #                 syn = new
    #             else:
    #                 break

    #         ## Calculate the privacy and utility, check if both requirements are still met
    #         print('final check')
    #         priv_val = data_requirement.privacy_metric.calculate(self.private_data, syn)
    #         util_val = data_requirement.utility_metric.calculate(self.private_data, syn)
    #         print('calc priv req: ' + str(priv_val))
    #         print('calc util req: ' + str(util_val))
    #         if data_requirement.privacy_metric.satisfied(data_requirement.privacy_value, priv_val, data_requirement.range) and data_requirement.privacy_metric.satisfied(data_requirement.utility_value, util_val, data_requirement.range):
    #             return self.generator

    # def increase_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Increase privacy')
    #     ## set current generator eps as low bound
    #     self.eps_low_priv_bound = self.generator.get_dp_epsilon()

    #     if self.eps_high_priv_bound >= self.eps_low_priv_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Increase privacy by decreasing epsilon
    #     new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_high_priv_bound) * amount), 1))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)
        
    #     new_syn = new_generator.generate(count=self.size)

    #     ## Check if the privacy increased
    #     new_privacy = self.privacy_calc.calculatePrivacy(self.private_data, new_syn)
    #     if new_privacy > self.current_privacy:
    #         self.current_privacy = new_privacy
    #         self.generator = new_generator
    #         return new_syn
    #     else:
    #         ## Further increase the privacy
    #         return self.increase_privacy(syn, amount * 2)
        
    # def decrease_privacy(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Decrease privacy')
    #     ## set current generator eps as high bound
    #     self.eps_high_priv_bound = self.generator.get_dp_epsilon()

    #     if self.eps_low_priv_bound >= self.eps_high_priv_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Decrease privacy by increasing epsilon
    #     new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() + ((self.eps_low_priv_bound - self.generator.get_dp_epsilon()) * amount), 1))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)
        
    #     new_syn = new_generator.generate(count=self.size)

    #     ## Check if the privacy decreased
    #     new_privacy = self.privacy_calc.calculatePrivacy(self.private_data, new_syn)
    #     if new_privacy < self.current_privacy:
    #         self.current_privacy = new_privacy
    #         self.generator = new_generator
    #         return new_syn
    #     else:
    #         ## Further decrease the privacy
    #         return self.decrease_privacy(syn, amount * 2)

    # def increase_utility(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Increase utility')
    #     ## set current generator eps as high bound
    #     self.eps_low_util_bound = self.generator.get_dp_epsilon()

    #     if self.eps_high_util_bound <= self.eps_low_util_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Increase utility by increasing epsilon
    #     new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() + ((self.eps_high_util_bound - self.generator.get_dp_epsilon()) * amount), 1))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)
        
    #     new_syn = new_generator.generate(count=self.size)

    #     ## Check if the utility increased
    #     new_utility = self.utility_calc.calculateUtility(self.private_data, new_syn)
    #     if new_utility > self.current_utility:
    #         self.current_utility = new_utility
    #         self.generator = new_generator
    #         return new_syn
    #     else:
    #         ## Further increase the utility
    #         return self.increase_utility(syn, amount * 2)

    # def decrease_utility(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Decrease utility')
    #     ## set current generator eps as low bound
    #     self.eps_high_util_bound = self.generator.get_dp_epsilon()

    #     if self.eps_high_util_bound <= self.eps_low_util_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Decrease utility by decreasing epsilon
    #     new_generator = self.set_new_eps(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_low_util_bound) * amount))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)

    #     new_syn = new_generator.generate(count=self.size)

    #     ## Check if the utility decreased
    #     new_utility = self.utility_calc.calculateUtility(self.private_data, new_syn)
    #     if new_utility < self.current_utility:
    #         self.current_utility = new_utility
    #         self.generator = new_generator
    #         return new_syn
    #     else:
    #         ## Further decrease the utility
    #         return self.decrease_utility(syn, amount * 2)










    ### Latest update
    # def create_generator(self, generators, protection_level: ProtectionLevel) -> Plugin:
    #     ## if protection level is specified by an epsilon value, we can easily create the right generator
    #     if protection_level.epsilon is not None:
    #         self.generator = self.plugin_class(epsilon=protection_level.epsilon)
    #         self.generator.fit(self.private_data)
    #         return self.generator
    #     ## Find the generator with the most similar protection level to use as a starting point
    #     self.generator = self.find_initial_generator(generators, protection_level)
    #     if self.generator is None:
    #         ## Create new starting point if no other generators exist
    #         self.generator = self.plugin_class(epsilon=1)
    #         self.generator.fit(self.private_data)

    #     ## Generate synthetic data as the starting point to check the privacy/utility
    #     syn = self.generator.generate(count=self.size)
        
    #     ## Adapt generator untill privacy/utility are met
    #     self.eps_high_bound = 10
    #     self.eps_low_bound = 0.1

    #     satisfied = False

    #     while not satisfied:

    #         ## Calculate the current privacy
    #         current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)
    #         ## Calculate the current utility
    #         current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

    #         ## Calculate the privacy with given metric for requirement
    #         priv_val = protection_level.privacy_metric.calculate(self.private_data, syn)
    #         print('calc req: ' + str(priv_val))
    #         ## Calculate the utility with given metric for requirement
    #         util_val = protection_level.utility_metric.calculate(self.private_data, syn)
    #         print('calc req: ' + str(util_val))

    #         if protection_level.privacy_metric.satisfied(protection_level.privacy_value, priv_val, protection_level.range) and protection_level.utility_metric.satisfied(protection_level.utility_value, util_val, protection_level.range):
    #             satisfied = True
    #             break

    #         ## Check how much the epsilon value of DP has to change and in what direction
    #         priv_amount = protection_level.privacy_metric.amount(protection_level.privacy_value, priv_val)
    #         util_amount = protection_level.utility_metric.amount(protection_level.utility_value, util_val)
    #         ## Check if we need to increase or decrease the epsilon value
    #         if priv_amount >= util_amount:
    #             if protection_level.privacy_metric.privacy() == 1:
    #                 if priv_val < protection_level.privacy_value:
    #                     new = self.decrease_epsilon(syn, priv_amount)
    #                     ## Calculate the new privacy
    #                     # new_privacy = self.privacy_calc.calculatePrivacy(self.private_data, new)
    #                     # if new_privacy < current_privacy:

    #                 else:
    #                     new = self.increase_epsilon(syn, priv_amount)
    #             else:
    #                 if priv_val > protection_level.privacy_value:
    #                     new = self.decrease_epsilon(syn, priv_amount)
    #                 else:
    #                     new = self.increase_epsilon(syn, priv_amount)
    #         else:
    #             if protection_level.utility_metric.utility() == 1:
    #                 if util_val < protection_level.utility_value:
    #                     new = self.increase_epsilon(syn, util_amount)
    #                 else:
    #                     new = self.decrease_epsilon(syn, util_amount)
    #             else:
    #                 if util_val > protection_level.utility_value:
    #                     new = self.increase_epsilon(syn, util_amount)
    #                 else:
    #                     new = self.decrease_epsilon(syn, util_amount)

    #             ## If new is None, no improvements are made
    #             if new is not None:
    #                 syn = new
    #             else:
    #                 return None

    #     return self.generator


    # def decrease_epsilon(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Decrease epsilon')
    #     ## set current generator eps as low bound
    #     self.eps_high_bound = self.generator.get_dp_epsilon()

    #     if self.eps_high_bound <= self.eps_low_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Decrease epsilon
    #     new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_low_bound) * amount), 1))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)
        
    #     new_syn = new_generator.generate(count=self.size)

    #     self.generator = new_generator

    #     return new_syn
        
    # def increase_epsilon(self, syn: DataLoader, amount: float) -> DataLoader:
    #     print('Increase epsilon')
    #     ## set current generator eps as low bound
    #     self.eps_low_bound = self.generator.get_dp_epsilon()

    #     if self.eps_low_bound >= self.eps_high_bound:
    #         ## No improvements can be made
    #         return None
        
    #     ## Increase epsilon
    #     new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() + ((self.eps_high_bound - self.generator.get_dp_epsilon()) * amount), 1))

    #     print(new_generator.get_dp_epsilon())

    #     new_generator.fit(self.private_data)
        
    #     new_syn = new_generator.generate(count=self.size)

    #     self.generator = new_generator

    #     return new_syn