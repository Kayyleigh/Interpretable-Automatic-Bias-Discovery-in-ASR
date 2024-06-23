import json

class FilepathManager:
    """
    Class for fetching all the files related to ASR performance according to a given configuration file.

    For the purpose of my research, metadata of predefined speaker group labels is assumed to be present. 
	To truly allow the use of this speaker representation clustering as an alternative to demographic groups, I still
	need to tweak the codebase.
    
    Attributes
    ----------
    base_path : str
        The base path to the ASR output directory
    asr_models : list of str
        List of ASR model names
    speaker_groups : list of str
        List of (demographic, or otherwise predefined) speaker groups
    base_metrics : list of str
        List of base metrics (e.g., "WER")
    path_templates : dict
        Path templates for performance output files and metadata files
    speaking_styles_data : dict
        Dictionary mapping speaking style IDs to their full name and abbreviation

    Methods
    -------
    get_error_rate_path(speaking_style_id, speaker_group, asr_model)
        Get the path for an error rate file. If the speaking style does not exist, the first 
        existing speaking style is used
    get_meta_path(speaking_style_id)
        Get the path for a metadata file. If the speaking style does not exist, the first 
        existing speaking style is used
    get_speaker_groups()
        Get the list of speaker groups specified in the config file
    get_asr_models()
        Get the list of ASR model names
    get_base_metrics()
        Get the list of base metrics
    get_speaking_styles_data()
        Get the dictionary mapping speaking style IDs to their full name and abbreviation
    get_speaking_styles_data(speaking_style_id)
        Get the full name and abbreviation of a specific speaking style
    """    
    def __init__(self, config_path):
        """Init FilePathManager.

    	Parameters
        ----------
        config_path : str
    		Path to the configuration file. The file is expected to be a JSON file 
            containing the following structure:
        
        asr_performance : dict
        	Dictionary containing ASR performance settings:

			base_path : str
				The base path to the ASR output directory
			asr_models : list of str
				List of ASR model names
			speaker_groups : list of str
				List of (demographic, or otherwise predefined) speaker group abbreviations
			base_metrics : list of str
				List of base metrics (e.g., "WER")
			path_templates : dict
				Dictionary containing path templates for files:

				error_rate_file : str
					Template for the error rate file path
				meta_file : str
					Template for the meta file path
                
        speaking_styles : list of dict
          	List of dictionaries, each containing information about a speaking style:
            
			id: str
				The identifier of a speaking style
			name: str
				The full name of the speaking style
			abbreviation: str
				The abbreviation that is used for directory and file names
        """        
        with open(config_path, 'r') as file:
            config = json.load(file)

        performance_config = config['asr_performance']

        self.base_path = performance_config['base_path']
        self.asr_models = performance_config['asr_models']
        self.speaker_groups = performance_config['speaker_groups']
        self.base_metrics = performance_config['base_metrics']
        self.path_templates = performance_config['path_templates']

        speaking_styles_config = config['speaking_styles']

        self.speaking_styles_data = {
            entry['id']: {
                'name': entry['name'],
                'abbreviation': entry['abbreviation']
            }
            for entry in speaking_styles_config
        }

    def _generate_path(self, template, **kwargs):
        """Generate a file path based on a template and keyword arguments.

        Parameters
        ----------
        template : str
            The path template
        **kwargs
        	Keyword arguments to fill the template

        Returns
        -------
        str
            The generated file path
        """
        return template.format(base_path=self.base_path, **kwargs)

    def get_error_rate_path(self, speaking_style_id, speaker_group, asr_model):
        """Get the file path for an error rate file. With the assumption that an error rate directory can be generated 
        knowing the base directory of recognition outputs, the name of the ASR model in question, the ID of the speaking 
        style in question and the name of the speaker group in question, this method fetches the `asr_model`'s recognition 
        output for `speaking_style_id` speech of the `speaker_group` group.

        Parameters
        ----------
        speaking_style_id : str
            The speaking style ID
        speaker_group : str
        	The speaker group
        asr_model : str
			The ASR model 
            
        Returns
        -------
        str
            The file path to the error rate file
        """
        # Get available speaking styles
        speaking_styles_dict = self.get_speaking_styles_data()
        
        # If speaking style is not available, assume the first one as default 
        speaking_style_abbreviation = speaking_styles_dict.get(speaking_style_id, speaking_styles_dict[next(iter(speaking_styles_dict))]).get('abbreviation')

        template = self.path_templates['error_rate_file']
        return self._generate_path(template, speaking_style=speaking_style_abbreviation, speaker_group=speaker_group, 
                                   asr_model=asr_model)
    
    def get_meta_path(self, speaking_style_id):
        """Get the file path for a metadata file. With the assumption that the metadata directory can be generated 
        knowing just the base directory of recognition outputs and the ID of the speaking style in question, this 
        method fetches the metadata for `speaking_style_id` speech of all speaker groups.

        Parameters
        ----------
        speaking_style_id : str
            The speaking style ID
            
        Returns
        -------
        str
            The file path to the metadata file
        """
        # Get available speaking styles
        speaking_styles_dict = self.get_speaking_styles_data()
        
        # If speaking style is not available, assume the first one as default 
        speaking_style_abbreviation = speaking_styles_dict.get(speaking_style_id, speaking_styles_dict[next(iter(speaking_styles_dict))]).get('abbreviation')
        
        template = self.path_templates['meta_file']
        return self._generate_path(template, speaking_style=speaking_style_abbreviation)
    
    def get_speaker_groups(self):
        """Get the list of (demographic, or otherwise predefined) speaker groups.

        Returns
        -------
        list of str
            List of speaker group names
        """
        return self.speaker_groups
    
    def get_asr_models(self):
        """Get the names of the ASR models for which the recognition output is available.

        Returns
        -------
        list of str
            List of ASR model names
        """
        return self.asr_models

    def get_base_metrics(self):
        """Get the names of the base metrics (e.g. Word Error Rate (WER), Phoneme Error Rate (PER), 
        Character Error Rate (CER))for which the recognition output is available.
        
        Returns
        -------
        list of str
            List of base metrics
        """
        return self.base_metrics

    def get_speaking_styles_data(self):
        """Get the speaking style IDs, names and abbreviations.

        Returns
        -------
        dict
            Dictionary where keys are speaking style IDs and values are dictionaries of their 'name' and 'abbreviation'
        """
        return self.speaking_styles_data
    
    def get_speaking_style_data(self, speaking_style_id):
        """Get the full name and abbreviation of a speaking style.

        Returns
        -------
        dict
            Dictionary with the 'name' and 'abbreviation' of the speaking style
        """
        # Get available speaking styles
        speaking_styles_dict = self.get_speaking_styles_data()
        
        # If speaking style is not available, assume the first one as default 
        speaking_style_data = speaking_styles_dict.get(speaking_style_id, speaking_styles_dict[next(iter(speaking_styles_dict))])

        return speaking_style_data
