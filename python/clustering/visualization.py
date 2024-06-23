import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from python.clustering.grouping import Grouping
from python.clustering.data_prep import DataPrep
from python.utils import format_label_unit

class DataVisualization:
    """
    Class for the visualization of a DataPrep object.

    Attributes
    ----------
    data_prep : DataPrep
        The DataPrep to visualize
    """    
    def __init__(self, data_prep: DataPrep):
        """Inits DataVisualization.

        Parameters
        ----------
        data_prep : DataPrep
            The DataPrep to visualize
        """        
        self.data_prep = data_prep

    def correlation_matrix(self):
        """Plot a triangular matrix of each Pearson correlation coefficient between pairs of speech features. 
        """        
        corr = self.data_prep.data.corr(numeric_only=True)

        # Don't show it if it's going to be empty anyway
        if corr.shape == (1, 1):
            return

        ticklabels = [custom_label for (field_label, (custom_label, unit)) in self.data_prep.get_custom_features().items()]

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", 
                    annot=True, xticklabels=ticklabels, yticklabels=ticklabels, mask=mask)
        plt.title(f"Correlation between Features ({self.data_prep.get_speaking_style()})")
        plt.tight_layout()
        plt.subplots_adjust(top=0.922, bottom=0.224, left=0.168, right=0.977, hspace=0.2, wspace=0.2)
        plt.xlim(0, corr.shape[0]-1)
        plt.ylim(corr.shape[1], 1)
        plt.show()

class GroupingVisualization:
    """
    Class for the visualization of a Grouping object.

    Attributes
    ----------
    grouping : Grouping
        The DataPrep to visualize
    markers : list
            The markers for the speaker groups to consistently use accross plots
    palette : _RGBColorPalette
            The colors for the speaker groups to consistently use accross plots
    """    
    def __init__(self, grouping: Grouping, 
                 markers=['o', 'v', 's', 'D', 'p'], 
                 palette=sns.color_palette("colorblind", 5)):
        """Inits GroupingVisualization.
 
        Parameters
        ----------
        grouping : Grouping
            The Grouping to visualize
        markers : list, optional
            A list of markers for the speaker groups to consistently use accross plots in this GroupingVisualization. 
            By default ['o', 'v', 's', 'D', 'p']
        palette : _RGBColorPalette, optional
            A list of colors for the speaker groups to consistently use accross plots in this GroupingVisualization. 
            By default sns.color_palette("colorblind", 5)
        """        
        self.grouping = grouping
        self.markers = markers
        self.palette = palette

    def show_grouping_2d(self, x_feature, y_feature):
        """Plot 2d scatterplot of the Grouping.

        Parameters
        ----------
        x_feature : str
            Name of the feature to plot on the x-axis
        y_feature : str
            Name of the feature to plot on the y-axis
        """        
        label_fieldname = self.grouping.get_label_fieldname()

        custom_features = self.grouping.data_prep.get_custom_features()

        # Plot the grouped data
        plt.figure(figsize=(6, 6))

        if x_feature not in self.grouping.data_prep.get_custom_features().keys():
            print(f"x-axis cannot be {x_feature} because {x_feature} is not part of the feature space")
            x_feature=np.zeros(len(self.grouping.data_prep.data))
            x_label = tuple(["Nothing","Cricket Noises per Second"])
        else:
            x_label = custom_features[x_feature]

        if y_feature not in self.grouping.data_prep.get_custom_features().keys():
            print(f"y-axis cannot be {y_feature} because {y_feature} is not part of the feature space")
            y_feature=np.zeros(len(self.grouping.data_prep.data))
            y_label = tuple(["Nothing","Cricket Noises per Second"])
        else:
            y_label = custom_features[y_feature]

        # Scatter plot of the data points with chosen colors and markers
        sns.scatterplot(data=self.grouping.get_data_with_group_labels(), x=x_feature, y=y_feature, 
                        hue=label_fieldname, style=label_fieldname, palette=self.palette, markers=self.markers, edgecolor="k")

        plt.title(f"{self.grouping.label_fieldname}s ({self.grouping.data_prep.get_speaking_style()})")
        plt.xlabel(format_label_unit(x_label))
        plt.ylabel(format_label_unit(y_label))
        plt.legend()
        plt.show()

    def show_grouping_boxplots(self):
        """Plot boxplots to visualize key characteristics of speaker groups. 
        A subplot containing a boxplot for each speaker group is created for every feature of speech.
        """        
        label_fieldname = self.grouping.get_label_fieldname()

        # Melt the DataFrame to long format
        data_melted = pd.melt(self.grouping.get_data_with_group_labels(), id_vars=['Filename', label_fieldname], 
                            var_name='Feature', value_name='Value')

        # Get original and custom feature names
        custom_features = self.grouping.data_prep.get_custom_features()

        # Set up the matplotlib figure
        num_features = len(custom_features)

        fig, axes = plt.subplots(1, num_features, figsize=(8 * num_features, 3), sharey=False)

        # Robust against single row/column
        if num_features == 1:
            axes = np.expand_dims(axes, axis=0)
            
        # Create a boxplot for each feature
        for i, feature in enumerate(custom_features):
            sns.boxplot(data=data_melted[data_melted['Feature'] == feature], x=label_fieldname, y='Value',
                        hue=label_fieldname, palette=self.palette, legend=False, ax=axes[i])
            axes[i].set_title(custom_features[feature][0])
            axes[i].set_xlabel(label_fieldname)
            ymin, ymax = axes[i].get_ylim()
            axes[i].set_yticks = range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1)
            axes[i].set_ylabel(custom_features[feature][1])
            #axes[i].set_ylabel(None)
            #axes[i].tick_params(axis='x', labelrotation=90)
        #axes[0].set_ylabel(custom_features[feature][1])
        plt.suptitle(f"Feature Variation between {label_fieldname}s ({self.grouping.data_prep.get_speaking_style()})")
        #plt.subplots_adjust(top=0.88, bottom=0.197, left=0.046, right=0.99, hspace=0.2, wspace=0.605)
        #plt.subplots_adjust(top=0.88, bottom=0.197, left=0.136, right=0.99, hspace=0.2, wspace=0.605)
        plt.show()
