from plugin import Plugin
from protectionLevel import ProtectionLevel
from privacyCalculator import PrivacyCalculator
from utilityCalculator import UtilityCalculator
from privacyMetrics import PrivacyMetric
from utilityMetrics import UtilityMetric

from synthcity.plugins.core.dataloader import DataLoader

from typing import Optional
import random 

class GeneratorCreator():
    def __init__(self, private: DataLoader, plugin_class: Plugin, privacy_calc: PrivacyCalculator, utility_calc: UtilityCalculator):
        self.size = 10000
        self.plugin_class = plugin_class
        self.private_data = private
        self.privacy_calc = privacy_calc
        self.utility_calc = utility_calc

    def create_generator(self, generators, protection_level: ProtectionLevel, privacy_metric: Optional[PrivacyMetric] = None, privacy_value: Optional[float] = None, utility_metric: Optional[UtilityMetric] = None, utility_value: Optional[float] = None, range: Optional[float] = None) -> Plugin:
        self.generators = generators
        self.protection_level = protection_level
        self.privacy_metric = privacy_metric
        self.privacy_value = privacy_value
        self.utility_metric = utility_metric
        self.utility_value = utility_value
        self.range = range

        ## if requirement is specified by an epsilon value, we can easily create the right generator
        if protection_level.epsilon is not None:
            self.generator = self.plugin_class(epsilon=protection_level.epsilon, protection_level=protection_level)
            self.generator.fit(self.private_data)
            return self.generator
        
        ## Find the generator with the most similar protection level to use as a starting point
        self.generator = self.find_initial_generator(generators, privacy_metric, privacy_value, utility_metric, utility_value)
        if self.generator is None:
            ## Create new starting point if no other generators exist
            self.generator = self.init_generator()
            self.generator.fit(self.private_data)

        ## Generate synthetic data as the starting point to check the privacy/utility
        syn = self.generator.generate(count=self.size)
        
        ## Adapt generator untill requirements are met
        self.eps_high_bound = 10
        self.eps_low_bound = 0.1

        satisfied = False
        counter = 0

        while not satisfied:
            ## After 5 rounds of reinitialization, stop the process
            if counter >= 5:
                print('Automatically creating a generator for this protection level did not work. try to add it manually.')
                return None 

            ## Both privacy and utility requirment given
            if privacy_metric is not None and utility_metric is not None:
                priv_satisfied = False
                util_satisfied = False
                ## Calculate the privacy with given metric for requirement
                priv_val = privacy_metric.calculate(self.private_data, syn)
                print('calc priv req: ' + str(priv_val))
                if privacy_metric.satisfied(privacy_value, priv_val, range):
                    priv_satisfied = True
                ## Calculate the utility
                util_val = utility_metric.calculate(self.private_data, syn)
                print('calc util req: ' + str(util_val))
                if utility_metric.satisfied(utility_value, util_val, range):
                    util_satisfied = True

                if priv_satisfied and util_satisfied:
                    satisfied = True
                    continue 

                ## Calculate the current privacy to use during creation
                self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)
                ## Calculate the current utility to use during creation
                self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)
                
                ## Check that we need to update the generator in the same direction for privacy and utility
                ## decrease epsilon
                if (privacy_metric.privacy() == 1 and priv_val < privacy_value) or (privacy_metric.privacy() == 0 and priv_val > privacy_value):
                    ## increase privacy = decrease epsilon
                    if (utility_metric.utility() == 1 and util_val > utility_value) or (utility_metric.utility() == 0 and util_val < utility_value):
                        ## decrease utility = decrease epsilon

                        ## Check how much the privacy has to change
                        priv_amount = privacy_metric.amount(privacy_value, priv_val)
                        ## Check how much the utility has to change
                        util_amount = utility_metric.amount(utility_value, util_val)

                        ## update epsilon
                        if priv_amount <= util_amount:
                            new = self.increase_privacy(priv_amount)
                        else:
                            new = self.decrease_utility(util_amount)
                    else:
                        ## increase utility == increase epsilon
                        counter = counter + 1
                        self.generator = self.init_generator()
                        self.generator.fit(self.private_data)
                        syn = self.generator.generate(count=self.size)
                        continue
                ## increase epsilon
                elif (privacy_metric.privacy() == 1 and priv_val > privacy_value) or (privacy_metric.privacy() == 0 and priv_val < privacy_value):
                    ## decrease privacy = increase epsilon
                    if (utility_metric.utility() == 1 and util_val < utility_value) or (utility_metric.utility() == 0 and util_val > utility_value):
                        ## increase utility = increase epsilon

                        ## Check how much the privacy has to change
                        priv_amount = privacy_metric.amount(privacy_value, priv_val)
                        ## Check how much the utility has to change
                        util_amount = utility_metric.amount(utility_value, util_val)

                        ## update epsilon
                        if priv_amount <= util_amount:
                            new = self.decrease_privacy(priv_amount)
                        else:
                            new = self.increase_utility(util_amount)
                    else:
                        ## decrease utility = decrease epsilon
                        counter = counter + 1
                        self.generator = self.init_generator()
                        self.generator.fit(self.private_data)
                        syn = self.generator.generate(count=self.size)
                        continue
                        
                ## If new is None, no improvements are made
                if new is not None:
                    syn = new
                else:
                    counter = counter + 1
                    self.generator = self.init_generator()
                    self.generator.fit(self.private_data)
                    syn = self.generator.generate(count=self.size)
                    continue

            ## only privacy requirement given            
            elif privacy_metric is not None:
                ## update privacy
                ## Calculate the current privacy to use during creation
                self.current_privacy = self.privacy_calc.calculatePrivacy(self.private_data, syn)

                ## Calculate the privacy with given metric for requirement
                val = privacy_metric.calculate(self.private_data, syn)
                print('calc priv req: ' + str(val))

                if privacy_metric.satisfied(privacy_value, val, range):
                    satisfied = True
                    continue

                ## Check how much the privacy has to change
                amount = privacy_metric.amount(privacy_value, val)

                ## Check if we need to increase or decrease privacy
                if privacy_metric.privacy() == 1:
                    if val < privacy_value:
                        new = self.increase_privacy(amount)
                    else:
                        new = self.decrease_privacy(amount)
                else:
                    if val > privacy_value:
                        new = self.increase_privacy(amount)
                    else:
                        new = self.decrease_privacy(amount)

                ## If new is None, no improvements are made
                if new is not None:
                    syn = new
                else:
                    counter = counter + 1
                    self.generator = self.init_generator()
                    self.generator.fit(self.private_data)
                    syn = self.generator.generate(count=self.size)
                    continue

            ## only utility metric given
            elif utility_metric is not None:
                ## update utility
                ## Calculate the current utility to use during creation
                self.current_utility = self.utility_calc.calculateUtility(self.private_data, syn)

                ## Calculate the utility
                val = utility_metric.calculate(self.private_data, syn)
                print('calc util req: ' + str(val))

                if utility_metric.satisfied(utility_value, val, range):
                    satisfied = True
                    continue
                        
                ## Check how much the utility has to change
                amount = utility_metric.amount(utility_value, val)
                    
                ## Check if we need to increase or decrease utility
                if utility_metric.utility() == 1:
                    if val < utility_value:
                        new = self.increase_utility(amount)
                    else:
                        new = self.decrease_utility(amount)
                else:
                    if val > utility_value:
                        new = self.increase_utility(amount)
                    else:
                        new = self.decrease_utility(amount)

                ## If new is None, no improvements are made
                if new is not None:
                    syn = new
                else:
                    counter = counter + 1
                    self.generator = self.init_generator()
                    self.generator.fit(self.private_data)
                    syn = self.generator.generate(count=self.size)
                    continue
                
        return self.generator

    def init_generator(self) -> Plugin:
        print("Init generator")
        n = random.randint(1,10)
        gen = self.plugin_class(epsilon=n, protection_level=self.protection_level)
        return gen

    def increase_privacy(self, amount: float) -> DataLoader:
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
        
    def decrease_privacy(self, amount: float) -> DataLoader:
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

    def increase_utility(self, amount: float) -> DataLoader:
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

    def decrease_utility(self, amount: float) -> DataLoader:
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
        new_generator = self.plugin_class(epsilon=eps, protection_level=self.protection_level)
        return new_generator

    def find_initial_generator(self, generators, privacy_metric: PrivacyMetric, privacy_value: float, utility_metric: UtilityMetric, utility_value: float) -> Plugin:
        closest_generator = None
        difference = 0
        for level, generator in generators.items():
            syn = generator.generate(count=self.size)
            diff = 0
            if privacy_metric is not None and utility_metric is not None:
                priv_diff = abs(privacy_metric.calculate(self.private_data, syn) - privacy_value)
                util_diff = abs(utility_metric.calculate(self.private_data, syn) - utility_value)
                diff = (priv_diff + util_diff) / 2
            elif privacy_metric is not None:
                diff = abs(privacy_metric.calculate(self.private_data, syn) - privacy_value)
            elif util_diff is not None:
                diff = abs(utility_metric.calculate(self.private_data, syn) - utility_value)
            if closest_generator is None:
                closest_generator = generator
                difference = diff
            elif diff < difference:
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