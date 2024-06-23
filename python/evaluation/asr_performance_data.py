import pandas as pd
from python.evaluation.filepaths import FilepathManager

class AsrPerformanceData:
    """
    Class representing ASR performance data.

    Attributes
    ----------
    filepath_manager : FilepathManager
        The FilepathManager that fetches ASR recognition output
    speaking_style_id : str
        The speaking style ID behind the recognition output, by default "Unknown Style"
    speaking_style_info : dict
        The speaking style name and abbreviation. If speaking_style_id does not exist, these
        are set to the first existing speaking style
    """    
    def __init__(self, filepath_manager: FilepathManager, speaking_style_id="Unknown Style"):
        """Init AsrPerformanceData.

        Parameters
        ----------
        filepath_manager : FilepathManager
            The FilepathManager to fetch the ASR recognition output from
        speaking_style_id : str, optional
            The ID of the speaking style to fetch the recognition output of
        """        
        self.filepath_manager = filepath_manager

        self.speaking_style_id = speaking_style_id
        self.speaking_style_info = filepath_manager.get_speaking_style_data(self.speaking_style_id)
        self.data = self.build_dataframe()
    
    def build_dataframe(self, base_metric_idx=0):
        """Build the dataframe of ASR recognition output. First, it fetches the metadata using 
        the filepath manager. Then it loops over all speaker groups and ASR models and for each
        pair it reads the corresponding .csv file and stores its data with the corresponding ASR
        model name. The resulting data is merged with metadata and returned as a dataframe. 

        The (demographic, or otherwise predefined) groups are assigned using the metadata, which
        contains for each speaker the speaker ID, their corresponding filename for that speaking
        style (these filenames should correspond to the 'Filename' column from feature extraction).
        This is because in practice, speaker groups would be unknown.  
        
        The resulting dataframe contains the following columns:
            - 'SPKR' : The ID of the speaker
            - '# Wrd' : The total word count of the speaker
            - 'Sub' : The number of substitutions of the speaker
            - 'Del' : The number of deletions of the speaker
            - 'Ins' : The number of insertions of the speaker
            - 'Model' : The name of the ASR model 

        Parameters
        ----------
        base_metric_idx : int, optional
            (still unused), by default 0

        Returns
        -------
        DataFrame
            The DataFrame containing recognition output and metadata
        """        
        # Find the metadata file for the speaking style that is defined as class attribute
        meta_filepath = self.filepath_manager.get_meta_path(speaking_style_id=self.speaking_style_id)

        # Create DataFrame from metadata .csv file
        meta_data = pd.read_csv(meta_filepath)
        
        # Get names of speaker groups and ASR models
        speaker_groups = self.filepath_manager.get_speaker_groups()
        asr_models = self.filepath_manager.get_asr_models()

        # Initialize a list to hold all model_error_data dataframes
        all_model_error_data = []    

        for group in speaker_groups:
            for model in asr_models:
                model_error_filepath = self.filepath_manager.get_error_rate_path(speaking_style_id=self.speaking_style_id,
                                                                speaker_group=group, asr_model=model)
            
                model_error_data = pd.read_csv(model_error_filepath)[['SPKR', '# Wrd', 'Sub', 'Del', 'Ins']]
                model_error_data['Model'] = model  # Add model identifier
                all_model_error_data.append(model_error_data)

        combined_model_error_data = pd.concat(all_model_error_data, ignore_index=True)
        df = pd.merge(meta_data, combined_model_error_data, on='SPKR', how='inner')
        return df
    
    def get_data(self):
        """Get the DataFrame containing recognition output and metadata.

        Returns
        -------
        DataFrame
            The DataFrame containing recognition output and metadata
        """        
        return self.data
    
    def get_speaking_style(self):
        """Get the ID of the speaking style.

        Returns
        -------
        str
            The ID of the speaking style
        """        
        return self.speaking_style_id
    