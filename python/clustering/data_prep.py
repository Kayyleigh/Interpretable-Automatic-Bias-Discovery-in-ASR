import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler

class DataPrep:
    """
    A class used to read speech features from .csv into a DataFrame.

    Attributes
    ----------
    speaking_style : str
        Name of the speaking style used in the speech that was extracted from
    data : DataFrame
        Extracted acoustic and prosodic features for every audiofile   
    audiofile_ids : list
        Identifiers of the audiofiles
    custom_features : dict[str, tuple[str, str]]
        Dictionary where each key is a string that matches the field names of the .csv file found at file_path,
        and each value is a tuple containing the custom fieldname for plots and the unit of measurement
    normalized_data : DataFrame
        Version of the data after a scaler has been applied. None if `normalize_data` has not yet been called.
    """
    def __init__(self, file_path, speaking_style, feature_dict=None, dropcolumns=[]):
        """Inits DataPrep.

        Parameters
        ----------
        file_path : str
            Path to the .csv file of extracted speech features
        speaking_style : str
            Name of the speaking style used in the speech that was extracted from. 
        feature_dict : dict[str, tuple[str, str]], optional
            Dictionary where each key is a string that matches the field names of the .csv file found at file_path,
            and each value is a tuple containing the custom fieldname for plots and the unit of measurement. For fields
            that exist in the .csv file but not in feature_dict, the field name from the .csv is used and the unit 
            becomes "Value". By default None
        dropcolumns : list, optional
            Which fields from the extracted speech features to exclude from the dataframe, by default []
        """  
        self.speaking_style = speaking_style

        self.data = pd.read_csv(file_path, encoding='ISO-8859-1')

        # Engineer new temporal features. Instead of the duration of a phoneme, the duration relative to that speaker's average 
        # phoneme duration is computed. This is hardcoded here due to time constraints.
        columnlist = self.data.columns.tolist()
        if 'Mean Duration O (s)' in columnlist:
            self.data['Mean Duration O (s)'] = self.data['Mean Duration O (s)'] * self.data['Mean Articulation Rate (phpm)'] / 60
        if 'Mean Duration E (s)' in columnlist:
            self.data['Mean Duration E (s)'] = self.data['Mean Duration E (s)'] * self.data['Mean Articulation Rate (phpm)'] / 60
        if 'Mean Duration A (s)' in columnlist:
            self.data['Mean Duration A (s)'] = self.data['Mean Duration A (s)'] * self.data['Mean Articulation Rate (phpm)'] / 60
        if 'Mean Duration u (s)' in columnlist:
            self.data['Mean Duration u (s)'] = self.data['Mean Duration u (s)'] * self.data['Mean Articulation Rate (phpm)'] / 60
        if 'Mean Duration @ (s)' in columnlist:
            self.data['Mean Duration @ (s)'] = self.data['Mean Duration @ (s)'] * self.data['Mean Articulation Rate (phpm)'] / 60

        # Drop columns to allow subsets of the extracted features
        self.data.drop(columns=dropcolumns, inplace=True, errors='ignore') 

        self.audiofile_ids = self.data['Filename'] # actually recording IDs, but I treat it as speaker
        
        csv_feature_fields = self.data.columns[1:]

        self.custom_features = {csvfield: (csvfield, "Value") for csvfield in csv_feature_fields}
        
        for csvfield in feature_dict:
            if csvfield in self.custom_features:
                self.custom_features[csvfield] = feature_dict[csvfield]
            # else:
                # print(f"{csvfield} is not one of the extracted features.")

        self.normalized_data = None
    
    def check_normality(self, feature_columns, threshold=0.05):
        """Conduct Shapiro-Wilk test for normality on extracted speech features.

        Parameters
        ----------
        feature_columns : array-like, Series, or list of arrays/Series
            Column names corresponding to the features for which to check for normality
        threshold : float
            The threshold for the p-value to determine normality, by default 0.05

        Returns
        -------
        DataFrame
            The Shapiro-Wilk Statistic and p-value of each feature
        """        
        results = []

        for column in self.data[feature_columns]:
            stat, p_value = shapiro(self.data[column])
            is_normal = p_value >= threshold
            results.append({
                'Feature': column,
                'Shapiro-Wilk Statistic': stat,
                'p-value': p_value,
                'Normality': is_normal
            })

        return pd.DataFrame(results).set_index('Feature')
      
    def normalize_data(self, scaler, check_normality=False):
        """Normalize the data and save the result in the class attribute `normalized_data`.

        Parameters
        ----------
        scaler : sklearn.preprocessing Scaler object
            The scaler to apply to the data
        check_normality : bool, optional
            When True, checks and prints whether all features are normally distributed. By default False
        """        
        feature_columns = self.get_custom_features().keys()

        if check_normality:
            shapiroWilk = self.check_normality(feature_columns)
            isNotNormal = shapiroWilk['Normality'].any(False)
            print(f"{'Not all' if isNotNormal else 'All'} features are normally distributed.")
            print(shapiroWilk)

        self.normalized_data = scaler.fit_transform(self.data[feature_columns])

    def get_audiofile_ids(self):
        """Get the identifiers of the audiofiles found in the .csv at `filepath`.

        Returns
        -------
        list
            Identifiers of the audiofiles
        """        
        return self.audiofile_ids

    def get_normalized_data(self, scaler=MinMaxScaler()):
        """Get the normalized data. If the data was not yet normalized, do that first.

        Parameters
        ----------
        scaler : sklearn.preprocessing Scaler object
            The scaler to apply to the data, by default MinMaxScaler()

        Returns
        -------
        DataFrame
            The normalized data
        """
        if self.normalized_data is None:
            self.normalize_data(scaler=scaler)
        return self.normalized_data
    
    def get_custom_features(self):
        """Get the custom feature names and units.

        Returns
        -------
        dict[str, tuple[str, str]]
            Dictionary where every key is the label used in the .csv at `file_path` and 
            every value is a tuple containing the custom label to use in plots and 
            the corresponding unit of measurement
        """        
        return self.custom_features
    
    def get_speaking_style(self):
        """Get the speaking style.

        Returns
        -------
        str
            Name of the speaking style used in the speech that was extracted from
        """
        return self.speaking_style
