# Discovering Bias in Dutch Automatic Speech Recognition by Clustering Interpretable Acoustic and Prosodic Features

## Description

This repository contains the codebase for my BSc thesis titled "Discovering Bias in Dutch Automatic Speech Recognition by Clustering Interpretable Acoustic and Prosodic Features" for the [Research Project](https://github.com/TU-Delft-CSE/Research-Project) 2024 at the [TU Delft](https://github.com/TU-Delft-CSE). The project proposes an interpretable approach to bias discovery by clustering speakers based on acoustic and prosodic features.

## Usage
The repository consists of multiple components:
1. **Feature Extraction using Praat:** Scripts for extracting acoustic and prosodic features from speech data using Praat.
2. **Bias Discovery in ASR using Python:** Code to cluster the extracted features into speaker groups and quantify biases in ASR.
3. **Config file that connects the parts:** [config.json](python\config\config.json) file where the location of feature extraction results, as well as the ASR recognition files, should be listed.

## Feature Extraction using Praat

This part of the project focuses on extracting acoustic and prosodic features from speech data using Praat. The [feature extraction script](praat\feature_extraction.praat) enables extraction of the following features:
- Mean Pitch (Hz)
- Mean Speech Rate (Phonemes per Minute): the number of phonemes per minute of audio
- Mean Articulation Rate (Phonemes per Minute): the number of phonemes per minute of speech
- Mean durations (in seconds) of specified phonemes	
- Mean Formant Frequencies (F1, F2, ...) at midpoints (50%) of chosen phoneme(s)
- Mean difference between Formant Frequencies (F1, F2, ...) at 20% vs 80% of diphthong(s)

Formant frequencies can be extracted in hertz or bark. Up to 5 formants can be chosen.

The [extraction setup](praat\extraction_setup.praat) file calls the extraction script on the experimental setup from the BSc thesis. However, the [feature extraction script](praat\feature_extraction.praat) can be used to measure any phonemes, so the script can even be used on other languages than Dutch!

## Bias Discovery in ASR using Python
This part of the project focuses on clustering the extracted features into speaker groups. It expects ASR recognition output files with insertions, deletions, substitutions and word counts for each speaker, and output files from the feature extraction using Praat.

The Python component is split into three parts:
1. **Clustering:** functionality for [data preparation](python\clustering\data_prep.py) and [clustering](python\clustering\grouping.py) speakers into speaker groups. The scaling method and clustering algorithm are softcoded. Predefined groups can be defined as well. The folder also enables [visualization](python\clustering\visualization.py) of clusters, speech characteristics per cluster (boxplots) and feature correlation matrices. 
2. **Evaluation:** functionality for [evaluating ASR models](python\evaluation\grouping_performance.py) based on some grouping, using different [metrics](python\evaluation\metrics.py). 
3. **Comparison:** functionality for comparing the biases of ASR models when using different feature sets.

In [main.py](main.py), the different parts are put together to form a "pipeline" that, given a valid config file, extracted features and ASR recognition output, can cluster speakers into speaker groups, then quantify bias and ASR performance using these groups vs demographic groups when these are present as well. 

In the BSc project, demographic groups were present despite the goal being to find bias without demographic metadata. Due to time constraints, the current codebase still assumes demographic metadata in several places for plotting and comparisons. 

## The [config.json](python\config\config.json) File
This is where the information should be stored on what data the code should expect and where. The following things can be controlled:
- ASR models: names of the ASR models under evaluation
- Speaker groups: names of the predefined demographic groups 
- Speaking styles, each containing an `id`, `name` and `abbreviation`
- Filepaths to the extracted features. Expects one file per speaking style. The value of the `speaking_style` field should be equal to the corresponding speaking style's `id`.
- Filepaths to the ASR recognition output. A filepath template can be given. The one that is there at the moment expects the names of each necessary file to be derived from the ASR model name(s) and speaking style abbreviation(s).
- Filepaths to the demographic metadata. A filepath template can be given. The one that is there at the moment expects the names of each necessary file to be derived from the speaking style abbreviation(s).

For more information on the functionality, please check the relevant files. All files in the [python](python) folder and the [praat](praat) folder are documented.

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/)
 License. See the [LICENSE](LICENSE) file for details.