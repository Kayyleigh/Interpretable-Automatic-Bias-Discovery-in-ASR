# By Kayleigh Jones, 13/06/2024
# as used in the BSc paper 
#
#	"Discovering Bias in Dutch Automatic Speech Recognition by
#	Clustering Interpretable Acoustic and Prosodic Features"
# 
# for the Research Project (CSE3000) at the Delft University of Technology
#
# This script calls the feature extraction script (feature_extraction.praat)
# with some pre-filled arguments. It is possible to extract features with the 
# other script directly, but this one makes it easier to re-use the setup. 
# This script therefore doubles as an overview of my hyperparameters.
#
# Arguments still needed when running this script:
#  - Directory of sound files
#  - Directory of transcription files
#  - sound file extension
#  - Transcription file extension
#  - Result file (full path + / + desired name and extension of new file)
#  - Excluded data file (full path + / + desired name and extension of new file)
#  - Tier number for word segmentations 
#  - Tier number for phoneme segmentations
#  - Option to print info to console
#
# Feature extraction setup pre-filled in this script:
#  Configuration:
#  - Min and max pitch: 50 and 800 Hz (default)
#  - Timestep for measuring mean pitch: 0 (default, becomes: 0.75 / (pitch floor))
#  - Max formant frequency: 5500 Hz (default)
#  - Unit of measurement for formants: bark 
#  Features to extract:
#  - Mean pitch in hertz
#  - Mean articulation rate in phonemes per minute of speech
#  - Mean duration of /E A u O @/ in seconds
#  - Mean F1 and F2 of /E A u O @/ at 50% in bark
#
################################################################################


# Pitch analysis
extract_pitch = 1
min_pitch = 50
max_pitch = 800
timestep = 0

# Speaking Rate analysis (1 = True, 0 = False)
extract_speech_rate = 0
extract_articulation_rate = 1

# Duration analysis
duration_phonemes_separated_by_whitespace$ = "E A u O @" 

# Formant analysis
max_formant_freq = 5500 
unit$ = "bark" 
phonemes_separated_by_whitespace$ = "E A u O @"
num_formants_per_phoneme$ = "2 2 2 2 2"
phoneme_formant_indices$ = "1 2 1 2 1 2 1 2 1 2"

# Diphthong analysis
diphthongs_separated_by_whitespace$ = ""
num_formants_per_diphthong$ = ""
diphthong_formant_indices$ = ""

# Script form
form Feature extraction
	comment Directory information
	text textGrid_directory /scratch/kmjones/JASMIN/Data/data/annot/text/awd/comp-q/nl
	sentence TextGrid_extension .awd
	text Sound_directory /scratch/kmjones/JASMIN/Data/data/audio/wav/comp-q/nl
	sentence Sound_extension .wav
	text Resultfile /scratch/kmjones/feature_extraction/results/extraction_comp_q.csv
	text Excluded_data_file /scratch/kmjones/feature_extraction/results/excluded_comp_q.csv

	comment Tier data
	natural word_tier 1
	natural phoneme_tier 3

	comment Debug
	boolean write_info_to_console 0
endform

runScript: "feature_extraction.praat", 
	... textGrid_directory$, textGrid_extension$, sound_directory$, sound_extension$, 
	... resultfile$, excluded_data_file$, write_info_to_console, 
	... word_tier, phoneme_tier, 
	... extract_pitch, min_pitch, max_pitch, timestep, 
	... extract_speech_rate, extract_articulation_rate, 
	... duration_phonemes_separated_by_whitespace$, 
	... max_formant_freq, unit$,
	... phonemes_separated_by_whitespace$, num_formants_per_phoneme$, phoneme_formant_indices$,
	... diphthongs_separated_by_whitespace$, num_formants_per_diphthong$, diphthong_formant_indices$
