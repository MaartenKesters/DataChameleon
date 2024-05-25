from generative_model_classes.plugin import Plugin
from metrics.privacyCalculator import PrivacyCalculator
from metrics.utilityCalculator import UtilityCalculator
from metrics.privacyMetrics import PrivacyMetric
from metrics.utilityMetrics import UtilityMetric
from protectionLevel import ProtectionLevel

from synthcity.plugins.core.dataloader import DataLoader

from typing import Optional, Tuple
import random 

class GeneratorCreator():
    def __init__(self, private: DataLoader, plugin_class: Plugin, privacy_calc: PrivacyCalculator, utility_calc: UtilityCalculator):
        self.size = 1000
        self.plugin_class = plugin_class
        self.private_data = private
        self.privacy_calc = privacy_calc
        self.utility_calc = utility_calc

    def create_generator(self, generators, protection_name: str, privacy: Optional[Tuple[PrivacyMetric, float]] = None, utility: Optional[Tuple[UtilityMetric, float]] = None, range: float = None) -> Plugin:
        self.generators = generators
        self.protection_name = protection_name
        if privacy is not None:
            privacy_metric = privacy[0]
            privacy_value = privacy[1]
        else: 
            privacy_metric = None
            privacy_value = None
        if utility is not None:
            utility_metric = utility[0]
            utility_value = utility[1]
        else:
            utility_metric = None
            utility_value = None
        range = range
        
        ## Find the generator with the most similar requirements to use as a starting point
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
            ## After 3 rounds of reinitialization, stop the process
            if counter >= 3:
                return None 

            ## Both privacy and utility requirment given
            if privacy_metric is not None and utility_metric is not None:
                priv_satisfied = False
                util_satisfied = False
                ## Calculate the privacy with given metric for requirement
                priv_val = privacy_metric.calculate(self.private_data, syn)
                # print('calc priv req: ' + str(priv_val))
                if privacy_metric.satisfied(privacy_value, priv_val, range):
                    priv_satisfied = True
                ## Calculate the utility
                util_val = utility_metric.calculate(self.private_data, syn)
                # print('calc util req: ' + str(util_val))
                if utility_metric.satisfied(utility_value, util_val, range):
                    util_satisfied = True

                if priv_satisfied and util_satisfied:
                    print('Requirements satisfied for automatically creating a generator.')
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
                        ## reinit generator
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
                # print('calc priv req: ' + str(val))

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
                # print('calc util req: ' + str(val))

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

    def increase_privacy(self, amount: float) -> DataLoader:
        print('Increase privacy')
        ## set current generator eps as high bound
        self.eps_high_bound = self.generator.get_dp_epsilon()

        if self.eps_low_bound >= self.eps_high_bound:
            ## No improvements can be made
            return None
        
        ## Increase privacy by decreasing epsilon
        new_generator = self.set_new_eps(round(self.generator.get_dp_epsilon() - ((self.generator.get_dp_epsilon() - self.eps_low_bound) * amount), 1))

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
    
    def init_generator(self) -> Plugin:
        n = random.randint(1,10)
        gen = self.plugin_class(epsilon=n, protection_level=ProtectionLevel(self.protection_name))
        return gen
        
    def set_new_eps(self, eps: float) -> Plugin:
        new_generator = self.plugin_class(epsilon=eps)
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