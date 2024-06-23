import pandas as pd

class BiasMeasure:
    """
        'Abstract' class for a bias measure.
    """
    def compute_bias(self, group_error_rate, reference_error_rate):
        """Compute the bias of a given speaker group, given the reference group.
        """     
        pass

    @property
    def metric_name(self):
        """The name of the bias measure.

        Returns
        -------
        "Bias"
        """        
        return "Bias"

class DiffBias(BiasMeasure):
    """
        Class representing a Difference bias measure.
    """
    
    def compute_bias(self, group_error_rate, reference_error_rate):
        """Compute the Difference Bias for a speaker group as the difference 
        between its error rate and that of the reference group.

        Parameters
        ----------
        group_error_rate : float
            The Error Rate of the speaker group 
        reference_error_rate : float
            The Error Rate of the reference group 

        Returns
        -------
        float
            The Difference Bias of the speaker group 
        """       
        return group_error_rate - reference_error_rate
    
    @property
    def metric_name(self):
        """The name of the bias measure.

        Returns
        -------
        "Difference Bias"
        """        
        return "Difference Bias"
    
class RelDiffBias(BiasMeasure):
    """
        Class representing a Relative difference bias measure
    """
    
    def compute_bias(self, group_error_rate, reference_error_rate):
        """Compute the Relative Difference Bias for a speaker group by taking the difference 
        in error rates between itself the reference group and dividing it by the error rate 
        of the reference group.

        Parameters
        ----------
        group_error_rate : float
            The Error Rate of the speaker group 
        reference_error_rate : float
            The Error Rate of the reference group 

        Returns
        -------
        float
            The Relative Difference Bias of the speaker group 
        """    
        return (group_error_rate - reference_error_rate) / reference_error_rate
    
    @property
    def metric_name(self):
        """The name of the bias measure.

        Returns
        -------
        "Relative Difference Bias"
        """      
        return "Relative Difference Bias"

class MetaMeasure:
    """
        'Abstract' class for a meta-measure.
    """
    
    def __init__(self, bias_measure: BiasMeasure):
        """Init MetaMeasure.

        Parameters
        ----------
        bias_measure : BiasMeasure
            The bias measure to use
        """        
        self.bias_measure = bias_measure
    
    def compute(self, error_rates_per_group):
        pass

    @property
    def metric_name(self):
        """The name of the meta-measure.

        Returns
        -------
        "Meta Bias"
        """   
        return "Meta Bias"

class OverallBias(MetaMeasure):
    """
        Overall bias measure
    """
    def __init__(self, bias_measure: BiasMeasure):
        super().__init__(bias_measure=bias_measure)
        
    def compute(self, error_rates_per_group):
        """Compute the Overall Bias. 

        Parameters
        ----------
        error_rates_per_group : DataFrame
            The WER per speaker group per ASR model

        Returns
        -------
        DataFrame
            The overall bias per ASR model
        DataFrame
            The WER and the Bias (as defined by the class `bias_measure` attribute)
            per speaker group per ASR model 
        """     
        group_error_rates_per_model = error_rates_per_group.groupby('Model')

        bias_results = []
        overall_bias_per_model = {}
        
        for name, group in group_error_rates_per_model:
            # Find the reference group's WER (minimum WER in the group)
            reference_error_rate = group['WER'].min()

            misrecognized_groups = group['WER'] != reference_error_rate

            # Compute bias for each group's WER
            group['Bias'] = group['WER'].apply(lambda x: self.bias_measure.compute_bias(x, reference_error_rate))
            # Calculate the overall bias for this model
            overall_bias = group['Bias'].mean()
            overall_bias_per_model[name] = overall_bias

            group_without_reference = group[misrecognized_groups]
            overall_bias = group_without_reference['Bias'].mean()
            overall_bias_per_model[name] = overall_bias
            
            # Append the group with bias results to the list
            bias_results.append(group)
        
        # Concatenate all the results into a single DataFrame
        bias_df = pd.concat(bias_results)
        
        # Return overall bias per model and bias per group
        overall_bias_per_model_df = pd.DataFrame(data=overall_bias_per_model.values(), index=overall_bias_per_model.keys(), columns=[self.metric_name])
        
        return overall_bias_per_model_df, bias_df

    @property
    def metric_name(self):
        """The name of the meta-measure.

        Returns
        -------
        "Overall Bias"
        """   
        return "Overall Bias"
   