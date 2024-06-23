import pandas as pd 

from python.evaluation.asr_performance_data import AsrPerformanceData
from python.clustering.grouping import Grouping
from python.evaluation.metrics import MetaMeasure
from python.utils import word_error_rate
    
class GroupingPerformance:
    """Class for merging a Grouping with an AsrPerformanceData to create the full dataframe
    of extracted speech features, assigned group labels, and ASR performance.
    """    
    def __init__(self, grouping: Grouping, asr_performance: AsrPerformanceData):
        """Init GroupingPerformance.

        Parameters
        ----------
        grouping : Grouping
            The Grouping containing the extracted feature vectors with the assigned group labels
        asr_performance : AsrPerformanceData
            The AsrPerformanceData containing recognition output and metadata of each speaker
        """        
        self.grouping = grouping
        self.asr_performance = asr_performance
        self.merge_data()

    def merge_data(self):
        """Merge the extracted features and their assigned group labels with the ASR performance data,
        with the assumption that both DataFrames contain a column called 'Filename' with (partially)
        matching IDs. Any leftover entries are excluded from the experiment.
        """        
        grouping_data = self.grouping.get_data_with_group_labels()
        performance_data = self.asr_performance.get_data()

        merged_data = performance_data.merge(grouping_data, on='Filename')  

        self.merged_data = merged_data

    def get_group_WERs(self, label_fieldname):
        """Calculate the WERs per group per ASR model.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker group labels

        Returns
        -------
        DataFrame
            WERs per group per ASR model
        """        
        grouped = self.get_data().groupby(['Model', label_fieldname]).agg('sum', numeric_only = True)
        grouped['WER'] = grouped.apply(lambda spkr: word_error_rate(spkr['Sub'], spkr['Del'], spkr['Ins'], spkr['# Wrd']), axis=1)
        wers_per_model_per_group = grouped[['WER']]

        return wers_per_model_per_group

    def calculate_bias(self, label_fieldname, meta_measure: MetaMeasure):
        """Calculate the performance of the ASR models for each speaker group given the 
        column where speaker group labels can be found and the meta-measure to apply.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker group labels
        meta_measure : MetaMeasure
            The meta-measure to apply

        Returns
        -------            
        DataFrame
            The performance per ASR model
        """        
        return meta_measure.compute(self.get_group_WERs(label_fieldname))[0]
    
    # TODO call the compute function once and store the result to avoid extra computational costs
    def calculate_bias_per_group(self, label_fieldname, meta_measure: MetaMeasure):
        """Calculate the WER and bias per speaker group for each ASR model, given 
        the column where speaker group labels can be found and the meta-measure to apply.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker group labels
        meta_measure : MetaMeasure
            The meta-measure to apply

        Returns
        -------
        DataFrame
            The WER and the Bias per speaker group per ASR model 
        """        
        return meta_measure.compute(self.get_group_WERs(label_fieldname))[1]

    def get_wer_per_speaker(self):
        """Calculate the WER for each speaker for each ASR model.

        Returns
        -------
        DataFrame
            WERs per speaker per ASR model
        """        
        wer_per_speaker = self.get_data().copy()

        # Calculate the WER for each speaker
        wer_per_speaker['Speaker Bias'] = self.get_data().apply(lambda spkr: word_error_rate(spkr['Sub'], spkr['Del'], spkr['Ins'], spkr['# Wrd']), axis=1)
        
        return wer_per_speaker

    def get_error_description_per_group(self, label_fieldname):
        """Generate descriptive statistics of the per-speaker WERs of each speaker group.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker group labels

        Returns
        -------
        DataFrame
            Summary statistics of the per-speaker WERs of each speaker group
        """        
        wer_per_speaker = self.get_wer_per_speaker()

        # Group by the 'Group' or 'Cluster' column
        grouped_performance_data = wer_per_speaker.groupby([label_fieldname, 'Model'])

        # Get statistics
        error_rates_descriptions = grouped_performance_data['Speaker Bias'].describe()

        return error_rates_descriptions

    def get_group_distribution(self, x_label_fieldname, y_label_fieldname):
        """Given two group assignment columns, calculate the percentage of each `x_label_fieldname`
        that is assigned to each `y_label_fieldname`.

        Parameters
        ----------
        x_label_fieldname : str
            Column to get the distributions of
        y_label_fieldname : str
            Column to get `x_label_fieldname`'s distributions for

        Returns
        -------
        DataFrame
            Cross tabulation of the two group assignments
        """        
        # Create dataframe with the two assigned groups per speaker  
        groups_by_speaker = self.get_data()[['Filename', x_label_fieldname, y_label_fieldname]].groupby('Filename').first()

        # Create the crosstab table of counts 
        counts_crosstab = pd.crosstab(groups_by_speaker[x_label_fieldname], 
                                    groups_by_speaker[y_label_fieldname])

        # Normalize the crosstab table to get the percentages
        percentage_crosstab = counts_crosstab.div(counts_crosstab.sum(axis=1), axis=0) * 100

        return percentage_crosstab
    
    def get_group_counts_comparison(self, x_label_fieldname, y_label_fieldname):
        """Given two group assignment columns, calculate the number of speakers in each `x_label_fieldname`
        that is assigned to each `y_label_fieldname`.

        Parameters
        ----------
        x_label_fieldname : str
            Column to get the counts of
        y_label_fieldname : str
            Column to get `x_label_fieldname`'s counts for

        Returns
        -------
        DataFrame
            Cross tabulation of the two group assignments
        """            
        # Create dataframe with the two assigned groups per speaker  
        groups_by_speaker = self.get_data()[['Filename', x_label_fieldname, y_label_fieldname]].groupby('Filename').first()
        
        # Create the crosstab table of counts 
        counts_crosstab = pd.crosstab(groups_by_speaker[x_label_fieldname], groups_by_speaker[y_label_fieldname])

        # Add column on totals per group
        counts_crosstab['Total'] = counts_crosstab.sum(axis=1)
        return counts_crosstab

    def get_data(self):
        """Get the data containing all extracted features, assigned group labels, and ASR performance data.

        Returns
        -------
        DataFrame
            DataFrame containing all extracted features, assigned group labels, and ASR performance data
        """      
        return self.merged_data
    
    def get_group_labels(self, label_fieldname):
        """Get the speaker group labels for a column that holds group assignments.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker group labels

        Returns
        -------
        list
            List of the speaker group labels
        """        
        return self.merged_data[label_fieldname].unique()
