# By Kayleigh Jones, 13/06/2024 as used in the BSc paper 
# "Discovering Bias in Dutch Automatic Speech Recognition by
#  Clustering Interpretable Acoustic and Prosodic Features"
# 
# Script for extracting features from the JASMIN-CGN NL data
# read (Rd) speech and Human Machine Interaction (HMI) speech


### GENERAL 
# Directory information
textGrid_extension$ = ".awd"
sound_extension$ = ".wav"

# Tier data
word_tier = 1
phoneme_tier = 3

# Debug
write_info_to_console = 0

### READ SPEECH
rd_textGrid_directory$ = "/scratch/kmjones/JASMIN/Data/data/annot/text/awd/comp-q/nl"
rd_sound_directory$ = "/scratch/kmjones/JASMIN/Data/data/audio/wav/comp-q/nl"
rd_resultfile$ = "/scratch/kmjones/feature_extraction/results/extraction_read_15062024.csv"
rd_excluded_data_file$ = "/scratch/kmjones/feature_extraction/results/excluded_read_15062024.csv"

runScript: "extraction_setup.praat", rd_textGrid_directory$, textGrid_extension$, 
    ... rd_sound_directory$, sound_extension$, rd_resultfile$, rd_excluded_data_file$, 
    ... word_tier, phoneme_tier, write_info_to_console

### HMI SPEECH
hmi_textGrid_directory$ = "/scratch/kmjones/JASMIN/Data/data/annot/text/awd/comp-p/nl"
hmi_sound_directory$ = "/scratch/kmjones/JASMIN/Data/data/audio/wav/comp-p/nl"
hmi_resultfile$ = "/scratch/kmjones/feature_extraction/results/extraction_hmi_15062024.csv"
hmi_excluded_data_file$ = "/scratch/kmjones/feature_extraction/results/excluded_hmi_15062024.csv"

runScript: "extraction_setup.praat", hmi_textGrid_directory$, textGrid_extension$, 
    ... hmi_sound_directory$, sound_extension$, hmi_resultfile$, hmi_excluded_data_file$, 
    ... word_tier, phoneme_tier, write_info_to_console
    