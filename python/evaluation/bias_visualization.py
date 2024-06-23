import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from python.evaluation.grouping_performance import GroupingPerformance
from python.evaluation.metrics import MetaMeasure

class BiasVisualization:
    """Class for the visualization of the ASR performance.
    """    
    def __init__(self, data_processor: GroupingPerformance,
                 markers=['o', 'v', 's', 'D', 'p'], 
                 palette=sns.color_palette("colorblind", 5)):
        """Inits BiasVisualization.

        Parameters
        ----------
        data_processor : DataProcessor
            the DataProcessor 
        markers : list, optional
            (still unused), by default ['o', 'v', 's', 'D', 'p']
        palette : _RGBColorPalette, optional
            (still unused), by default sns.color_palette("colorblind", 5)
        """        
        self.data_processor = data_processor
        self.markers = markers
        self.palette = palette

    def correlation_features_errors(self):
        """Plot the Pearson correlation coefficients of every pair of a speech feature and an ASR model's performance.
        """        
        model_columns = self.data_processor.asr_performance.filepath_manager.get_asr_models()
        features = self.data_processor.grouping.data_prep.get_custom_features()
        feature_columns = features.keys()
        custom_feature_names = [features.get(fc)[0] for fc in feature_columns]

        speaking_style = self.data_processor.asr_performance.get_speaking_style()

        features_wer = self.data_processor.get_wer_per_speaker()
        pivot_df = features_wer.pivot_table(index=feature_columns, columns='Model', values='Speaker Bias').reset_index()
        corr = pivot_df.corr(numeric_only=True)[model_columns].loc[feature_columns]

        # Plot the heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", 
                    annot=True, yticklabels=custom_feature_names)
        plt.title(f'Correlation between Features and ASR Performance ({speaking_style})')
        plt.xlabel('ASR models')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

    def plot_group_biases(self, label_fieldname, meta_measure: MetaMeasure):
        """Plot the bias of each speaker group.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker groups
        meta_measure : MetaMeasure
            The meta-measure defining how bias is calculated
        """        
        speaking_style = self.data_processor.asr_performance.get_speaking_style()

        model_bias_df = self.data_processor.calculate_bias_per_group(label_fieldname=label_fieldname, meta_measure=meta_measure)
        groups = self.data_processor.get_group_labels(label_fieldname=label_fieldname)

        model_bias_pivot_df = model_bias_df.pivot_table(values='Bias', index=label_fieldname, columns='Model')

        num_groups = len(groups)
        fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 5), sharey=True)
        
        for ax, group in zip(axes, groups):
            sns.barplot(data=model_bias_pivot_df.loc[group], ax=ax)
            ax.set_title(f'{label_fieldname} {group}')
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_xlabel('ASR Models')
            ax.set_ylabel(meta_measure.metric_name)
        
        plt.suptitle(f'{meta_measure.bias_measure.metric_name} per {label_fieldname} ({speaking_style})')
        plt.tight_layout()
        plt.show()

    def plot_error_rates(self, label_fieldname):
        """Plot the Min, Max, Mean ± SD, and Median Word Error Rate of the speakers within a group.

        Parameters
        ----------
        label_fieldname : str
            The column that holds the assigned speaker groups
        """        
        performance_per_group_description = self.data_processor.get_error_description_per_group(label_fieldname=label_fieldname)
        performance_per_group_description.reset_index(level=['Model'], inplace=True)

        # Plotting
        groups = performance_per_group_description.index.unique()
        models = performance_per_group_description['Model'].unique()

        fig, axes = plt.subplots(1, len(groups), figsize=(2 * len(groups), 5), sharey=True)
        
        # TODO set color per model
        for ax, group in zip(axes, groups):
            df = performance_per_group_description.loc[group]
            means = df['mean']
            stds = df['std']
            mins = df['min']
            maxs = df['max']
            medians = df['50%']
            xticks = np.arange(len(models))
            
            ax.errorbar(xticks, means, yerr=stds, fmt='o', label='Mean ± Std', capsize=6)
            ax.scatter(xticks, mins, marker='*', label='Min')
            ax.scatter(xticks, maxs, marker='s', label='Max')
            ax.scatter(xticks, medians, marker='^', label='Md')

            ax.set_xticks(xticks)
            ax.set_xticklabels(models)
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_title(f'{label_fieldname}: {group}')
            ax.set_xlabel('Model')

        axes[0].set_ylabel('Word Error Rate')
        plt.suptitle(f'Statistics per ASR Model per {label_fieldname} ({self.data_processor.asr_performance.get_speaking_style()})')
        plt.tight_layout()
        plt.show()
