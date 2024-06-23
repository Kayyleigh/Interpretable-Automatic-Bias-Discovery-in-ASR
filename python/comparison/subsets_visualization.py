import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SubsetsVisualization:
    """
    Class for visual comparison of the ASR performance when clustering on different feature sets.

    Attributes
    ----------
    palette : _RGBColorPalette
        (still unused) Colors used for the different ASR models
    """    
    def __init__(self, palette=sns.color_palette("colorblind", 5)):
        """Init SubsetsVisualization.

        Parameters
        ----------
        palette : _RGBColorPalette, optional
            Colors to use for the different ASR models, by default sns.color_palette("colorblind", 5)
        """        
        self.palette = palette # TODO implement colors per ASR model
    
    def plot_error_description_per_feature_subset(self, results):
        """Plot the Min, Max, Mean ± SD, and Median speaker group Word Error Rate 
        for each predefined grouping and/or clustering on some feature set. 

        Parameters
        ----------
        results : dict[tuple[str, str], DataFrame]
            Dictionary where each key is a tuple of the subset's name and the speaking style, and its corresponding value 
            a DataFrame of the error rates per group per ASR model.
        """        
        num_speaking_styles = len(np.unique([style for _, style in results.keys()]))
        num_solutions = len(results) // num_speaking_styles

        fig, axes = plt.subplots(num_speaking_styles, num_solutions, figsize=(2 * num_solutions, 3 * num_speaking_styles), sharey="row")
    
        # Robust against single row/column
        if num_speaking_styles == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_solutions == 1:
            axes = np.expand_dims(axes, axis=1)

        # Ensure solutions are in a consistent order
        sorted_results = sorted(results, key=lambda x: (-ord(x[1][0]), x[0]))
        
        # TODO set color per model
        for i, (solution, speaking_style) in enumerate(sorted_results):
            row = i // num_solutions
            col = i % num_solutions
            ax = axes[row, col]

            label_fieldname = results[(solution, speaking_style)].index.names[1]
            model_performance_per_group = results[(solution, speaking_style)].pivot_table(values='WER', index=label_fieldname, columns='Model')

            models = model_performance_per_group.columns
            description = model_performance_per_group.describe().loc[["mean", "std", "min", "max", "50%"]]

            means = description.loc['mean']
            stds = description.loc['std']
            mins = description.loc['min']
            maxs = description.loc['max']
            medians = description.loc['50%']
            xticks = np.arange(len(model_performance_per_group))

            ax.errorbar(xticks, means, yerr=stds, fmt='o', label='Mean ± Std', capsize=6)
            ax.scatter(xticks, mins, marker='*', label='Min')
            ax.scatter(xticks, maxs, marker='s', label='Max')
            ax.scatter(xticks, medians, marker='^', label='Median')

            ax.set_xticks(xticks)
            ax.set_xticklabels(models)
            ax.tick_params(axis='x', labelrotation=90)

            # To control the ordering of the plots, the names start with an extra number 
            # which is used for ordering, and this number is cut off when printing the title
            ax.set_title(f'{solution[1:]} ({speaking_style})')
        
        for i in range(num_speaking_styles):
            axes[i, 0].set_ylabel('Word Error Rate')
        plt.suptitle(f'Statistics per ASR Model per Feature Subset')
        plt.tight_layout()
        plt.legend()
        plt.show()
