import numpy as np
from python.clustering.data_prep import DataPrep

class Grouping:
    """
    Class representing a grouping, i.e., a DataPrep object for which each sample is assigned to some group.

    Attributes
    ----------
    data_prep : DataPrep
        DataPrep object containing the data to group
    label_fieldname : str
        The name of the column that contains the assigned group values
    group_labels : list of str
        List containing the speaker group for every speaker in the DataPrep
    data_with_group_labels : DataFrame
        The data with the group column added to it
    """    
    def __init__(self, data_prep: DataPrep, label_fieldname):
        """Init Grouping.

        Parameters
        ----------
        data_prep : DataPrep
            The extracted audio features as a DataPrep object
        label_fieldname : str
            The name to give the column that shows the assigned speaker group 
        """
        self.data_prep = data_prep
        self.label_fieldname = label_fieldname
        self.group_labels = None
        self.data_with_group_labels = None

    def _add_labels(self):
        """
        Add the column with the assinged speaker group name for each speaker.
        """
        self.data_with_group_labels = self.data_prep.data.assign(**{self.label_fieldname: self.group_labels})

    def get_data_with_group_labels(self):
        """Get the data with the column that assigns a speaker group to each speaker.

        Returns
        -------
        DataFrame
            The data with the group column added to it
        """        
        return self.data_with_group_labels
    
    def get_group_labels(self):
        """Get the assigned group labels.

        Returns
        -------
        list
            List containing the speaker group for every speaker in the DataPrep
        """        
        return self.group_labels

    def get_label_fieldname(self):
        """Get the name of the column that holds the group assignments.

        Returns
        -------
        str
            The name of the column that holds the group assignments
        """        
        return self.label_fieldname

    def get_group_sizes(self):
        """Get the number of samples that were assigned to each speaker group.

        Returns
        -------
        ndarray
            Array of length 2 arrays holding the speaker group (which can be a cluster) name and size
        """        
        unique, counts = np.unique(self.group_labels, return_counts=True)
        return np.asarray((unique, counts)).T
    
class Clustering(Grouping):
    """
        Class representing a Grouping of a DataPrep by performing clustering on its data.

        Inherits methods from Grouping.
    """
    def __init__(self, data_prep: DataPrep, clusteringAlgorithm):
        """Inits Clustering by applying the clustering algorithm to the extracted features and adding the resulting cluster assignments using `_add_labels()`.

        Parameters
        ----------
        data_prep : DataPrep
            The extracted audio features as a DataPrep object
        clusteringAlgorithm : sklearn.cluster object
            The clustering algorithm to apply
        """        
        super().__init__(data_prep=data_prep, label_fieldname='Cluster')
        model = clusteringAlgorithm
        self.group_labels = model.fit_predict(self.data_prep.get_normalized_data())
        self._add_labels()

    def _add_labels(self):
        """Add the column containing assigned group names.
        """        
        super()._add_labels()

class PredefinedGrouping(Grouping):
    """
        Class representing a Grouping of a DataPrep by predefined group assignments.

        Inherits methods from Grouping.
    """

    def __init__(self, data_prep: DataPrep, group_labels):
        """Inits PredefinedGrouping and adds the predefined labels by calling `_add_labels()`.

        Parameters
        ----------
        data_prep : DataPrep
            The extracted audio features as a DataPrep object
        group_labels : list of str
            The predefined group assignment. Values will be added to 
            speakers in the same order as they appear in the .csv they were read from
        """        
        super().__init__(data_prep=data_prep, label_fieldname='Group')
        self.group_labels = group_labels
        self._add_labels()

    def _add_labels(self):
        """Add the column containing assigned group names.
        """        
        super()._add_labels()
    