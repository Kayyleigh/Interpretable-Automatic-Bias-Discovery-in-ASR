#########################################################################################################
# Feature extraction script for Praat by Kayleigh Jones, on 13/06/24, for CSE3000 Research Project	#
# as used in the BSc paper "Discovering Bias in Dutch Automatic Speech Recognition by Clustering	#
# Interpretable Acoustic and Prosodic Features"								#
#													#
#	 REQUIREMENTS											#
#													#
# This Praat script extracts features from a directory of files. Two directories are needed: 		#
# 	- Directory of audio files									#
# 	- Directory of annotations. Must have at least 2 tiers: one with word segmentations, and 	#
#	  one with phoneme segmentations. Note: since changing the speech rates to phonemes instead 	#
#	  of words per minute, the word tier is no longer used. However, it is kept so potential 	#
#	  future additions to the script can use it. 							#
# Directories need to be paired such that each transcription file has a corresponding audio file.	#
#	 												#
#	 OUTPUT												#
#													#
# Two output files are made:								 		#
# 	- Result file of extracted features								#
# 	- Exclusion file of excluded data, identified by filename, and the reason for exclusion.	#
# 	  Reasons may include lack of occurrences of phonemes, or no valid measurements of formants.	#
#													#
#	 FEATURES											#
#													#
# Features that can be extracted using this script:							#
#	- Mean Pitch (Hz)										#
#	- Mean Speech Rate (Phonemes per Minute): the number of phonemes per minute of audio		#
#	- Mean Articulation Rate (Phonemes per Minute): the number of phonemes per minute of speech	#
#	- Mean durations (in seconds) of specified phonemes						#
#	- Mean Formant Frequencies (F1, F2, ...) at midpoints (50%) of chosen phoneme(s)		#
#	- Mean difference between Formant Frequencies (F1, F2, ...) at 20% vs 80% of diphthong(s)	#
#													#
# This script has the following limitations:								#
#	- It assumes that every file contains only 1 speaker. If more speakers are present on diffe-	#
#	  rent tiers, only the chosen tiers are used, therefore ignoring the other speakers. If one 	#
#	  tier contains multiple speakers, the results will be invalid but the script will not know 	#
#	  and therefore not exclude it from the result file.						#
#	- Since this script has to work with diverse speakers without metadata, all frequency ex-	#
#	  traction processes use the same maximum frequency. This can impact the precision of results 	#
#	  since normally a different max is picked depending on the age and gender of the speaker.	#
#	- Due to the complexity of counting syllables, this script counts the number of phonemes per 	#
#	  minute for Speaking Rates. Comparison of results can be less accurate when the files are 	#
#	  expected to contain nonidentical utterances. Note: this was first implemented using word 	#
#	  counts instead. The implementation for this has been commented out so it can be recovered, 	#
#	  should future research require it.								#
#	- When a speaker is found to be invalid, its other features are still extracted, meaning some	#
#	  work is done for no reason. This is because I have not found a way to skip parts of the	#
#	  script without negatively impacting the maintainability.				   	#
#	- The script does not support measuring formants at the start (20%) of diphthongs because it	#
#	  is not worth implementing for my use case due to time constraints.				#
#													#
#	 USAGE												#
#													#
# Example of how to call this script:									#
# 	/path/to/praat_executable --run /path/to/this/praat_script.praat "/path/to/annot/dir" 		#
#	".awd" "/path/to/sound/dir" ".wav" "/path/to/result/file.csv" "/path/to/exclusion/file.csv"	#
#	0 1 3 1 50 800 0 0 1 "y" 5500 "E A Y+" "1 2 1" "1 3 4 2" "Y+" "1" "2"				#
#													#
# The 3rd line should contain the following arguments (different values may be picked):			#
# 	0 (Bool): whether to log progress to console (not recommended for large sample sizes!)		#
#	1 (Nat): number of the annotation tier with 1 word per interval on it. If no word tier is 	#
#	  present in the annotations, any number can be given because it isn't used anyway. 		#
#	3 (Nat): number of the annotation tier with 1 phoneme per interval on it			#
#	1 (Bool): whether to extract the mean pitch							#
#	50 (Nat): minimum pitch, 50 is Praat's default							#
#	800 (Nat): maximum pitch, 800 is Praat's default						#
#	0 (Nat): timestep for pitch measuring, 0 is Praat's default					#
#	0 (Bool): whether to extract the speech rate							#
#	1 (Bool): whether to extract the articulation rate						#
#	"y" (String): phonemes to extract the mean duration of. WHITESPACE-SEPARATED, e.g. "a e u" 	#
#	5500 (Nat): max formant frequency to measure. 5500 is Praat's default (for women)		#
#	"E A Y+" (String): phonemes to extract the formants at 50% of, similar to for durations		#
#	"1 2 1" (String): for each phoneme (see prev), how many formants to extract. Must be 		#
#		whitespace-separated natural numbers							#
#	"1 3 4 2" (String): for each phoneme x formant, the formant number. Must be 1 to 5. The		#
#		past 3 example args together mean: take F1 of /E/, F3 and F4 of /A/, and F2 of /Y+/ 	#
#	"Y+" (String), "1" (String), "2" (String): same as above, but for diphthongs to extract the	#
#		difference between the formants at 80% vs 20%						#
#													#
# Wherever configurations are hardcoded, Praat's standard values are used.				#
#########################################################################################################

# Script form
form Feature extraction
	comment Directory information
	text TextGrid_directory /scratch/kmjones/JASMIN/Data/data/annot/text/awd/comp-q/nl
	sentence TextGrid_extension .awd
	text Sound_directory /scratch/kmjones/JASMIN/Data/data/audio/wav/comp-q/nl
	sentence Sound_extension .wav
	text Resultfile /scratch/kmjones/feature_extraction/results/extraction_comp_q.csv
	text Excluded_data_file /scratch/kmjones/feature_extraction/results/excluded_comp_q.csv

	comment General
	boolean Write_info_to_console 0
	
	comment Tier data
	natural Word_tier 1
	natural Phoneme_tier 3
	
	comment Pitch analysis
	boolean Extract_pitch 1
	natural Min_pitch 50
	natural Max_pitch 800
	positive Timestep 0

	comment Speaking rate analysis
	boolean Extract_speech_rate 0
	boolean Extract_articulation_rate 1

	comment Vowel duration analysis
	sentence Duration_phonemes_separated_by_whitespace y 

	comment Vowel Formant analysis at 50%
	natural Max_formant_freq 5500 
	optionmenu Unit 2
		option hertz
		option bark
	sentence Phonemes_separated_by_whitespace E A Y+ u
	sentence Num_formants_per_phoneme 1 1 1 1
	sentence Phoneme_formant_indices 1 2 2 2

	comment Diphthongization analysis at 20% vs 80%
	sentence Diphthongs_separated_by_whitespace Y+
	sentence Num_formants_per_diphthong 1
	sentence Diphthong_formant_indices 2
endform

# Check if the result file already exists
if fileReadable (resultfile$)
	pause The resultfile already exists! Overwrite?
	filedelete 'resultfile$'
endif

# Check if the excluded data file already exists
if fileReadable (excluded_data_file$)
	pause The excluded data file already exists! Overwrite?
	filedelete 'excluded_data_file$'
endif

# Make phoneme & diphthong vector (already needed for title line of result file) 
duration_phonemes$# = splitByWhitespace$# (duration_phonemes_separated_by_whitespace$)
phonemes$# = splitByWhitespace$# (phonemes_separated_by_whitespace$)
diphthongs$# = splitByWhitespace$# (diphthongs_separated_by_whitespace$)

# Do ugly stuff because the intuitive way does not work at all for some reason
num_formants_per_phoneme$# = splitByWhitespace$# (num_formants_per_phoneme$)
phoneme_formant_indices$# = splitByWhitespace$# (phoneme_formant_indices$)
num_formants_per_diphthong$# = splitByWhitespace$# (num_formants_per_diphthong$)
diphthong_formant_indices$# = splitByWhitespace$# (diphthong_formant_indices$)

for i to size (num_formants_per_phoneme$#)
	num_formants_per_phoneme[i] = number (num_formants_per_phoneme$# [i])
endfor
for i to size (phoneme_formant_indices$#)
	phoneme_formant_indices[i] = number (phoneme_formant_indices$# [i])
endfor
for i to size (num_formants_per_diphthong$#)
	num_formants_per_diphthong[i] = number (num_formants_per_diphthong$# [i])
endfor
for i to size (diphthong_formant_indices$#)
	diphthong_formant_indices[i] = number (diphthong_formant_indices$# [i])
endfor

### Create row with column titles to the result file

titlesline$ = "Filename,"

if extract_pitch
	titlesline$ += "Mean Pitch (Hz),"
endif

if extract_speech_rate
	#titlesline$ += "Mean Speech Rate (wpm),"
	titlesline$ += "Mean Speech Rate (phpm),"
endif
if extract_articulation_rate
	#titlesline$ += "Mean Articulation Rate (wpm),"
	titlesline$ += "Mean Articulation Rate (phpm),"
endif

for d_p from 1 to size (duration_phonemes$#)
	averageDuration$ = "Mean Duration " + duration_phonemes$# [d_p] + " (s)"
	titlesline$ += "'averageDuration$',"
endfor

phoneme_formant_index = 1
for p from 1 to size (phonemes$#)
	num_formants = num_formants_per_phoneme[p]
	for f from 1 to num_formants
		formantName$ = "Mean F" +  string$ (phoneme_formant_indices[phoneme_formant_index]) + " " + phonemes$# [p] + " (" + unit$ + ")"
		titlesline$ += "'formantName$',"
		phoneme_formant_index += 1
	endfor
endfor

diphthong_formant_index = 1
for d from 1 to size (diphthongs$#)
	num_formants_diph = num_formants_per_diphthong[d]
	for f from 1 to num_formants_diph
		diphthongName$ = "Mean Delta F" + string$ (diphthong_formant_indices[diphthong_formant_index]) + " " + diphthongs$# [d] + " (" + unit$ + ")"
		titlesline$ += "'diphthongName$',"
		diphthong_formant_index += 1
	endfor
endfor

# Remove last comma (prevent empty extra column from being added)
titlesline$ = left$ (titlesline$, length (titlesline$) - 1)
titlesline$ += "'newline$'"

# Add fieldnames line to result file
fileappend "'resultfile$'" 'titlesline$'

# Add fieldnames to excluded data file
exclusion_titlesline$ = "Filename," + "Reason," + "'newline$'"
fileappend "'excluded_data_file$'" 'exclusion_titlesline$'

### Start of actual feature extraction 

textGridList = Create Strings as file list: "textGridList", textGrid_directory$ + "/*" + textGrid_extension$
numFiles = Get number of strings

# PRINTING TO CONSOLE
if write_info_to_console
	writeInfoLine: "Starting feature extraction..."
endif

for file from 1 to numFiles
	selectObject: textGridList
	filename$ = Get string: file

	thisTextGrid = Read from file: textGrid_directory$ + "/" + filename$
	thisTextGrid$ = selected$("TextGrid")

	# Assuming 1 speaker per file 
	thisSpeaker$ = thisTextGrid$

	# PRINTING TO CONSOLE
	if write_info_to_console
		writeInfoLine: "Extracting features for ", thisSpeaker$, "..."	
	endif

	# Fetch and select sound and file
	thisSound = Read from file: sound_directory$ + "/" + thisSpeaker$ + sound_extension$
	thisSound$ = selected$("Sound")

	# Initialize validity of speaker. 1 for valid, 0 for invalid. Invalid if e.g. the vowel in question never occurs.
	isValid = 1
	# If invalid, reason(s) will be stored in order to list excluded datapoints in a file
	# The array of reasons is not initialized because Praat does not like that. Due to the way veriables are treated 
	# (= globally), it is initialised when the first reason is found 
	reasonsInvalidIndex = 1

	resultline$ = "'thisSpeaker$',"

	### Pitch extraction
	if extract_pitch
		select Sound 'thisSound$'
		plus TextGrid 'thisTextGrid$'

		# Extract all intervals where the speaker's phoneme tier is not empty
		Extract intervals where: phoneme_tier, "no", "is not equal to", ""

		# Save the ID numbers of the currently selected intervals
		num_voiced_intervals = numberOfSelected()
		for i from 1 to num_voiced_intervals
  			voiced_sounds[i] = selected(i)
		endfor

		# Create sound chain by concatinating all nonempty intervals, to create a cut of the audio where only the speaker speaks
		# This is necessary to avoid measuring the pitch of Text-to-Speech (or even background noise) as well, when applicable. 
		voiced_sound_chain = Concatenate

		# Iteratively remove all voiced intervals sound objects that are left over after concatination   
		for i from 1 to num_voiced_intervals
			selectObject: voiced_sounds[i]
			Remove
		endfor

		# Select sound chain of concatinated voiced intervals
		selectObject: voiced_sound_chain

		# Create Pitch object for the sound chain
		To Pitch... timestep min_pitch max_pitch

		# Calculate the mean pitch of the entire sound chain
		meanPitch = Get mean... 0 0 Hertz

		# Remove the new Pitch object from the object list
		Remove

		# Remove the sound chain from the object list
		selectObject: voiced_sound_chain
		Remove

		# PRINTING TO CONSOLE
		if write_info_to_console
			appendInfoLine: "mean pitch: ", meanPitch
		endif

		# Write mean pitch to result
		resultline$ += "'meanPitch:10',"
	endif

	### Speaking rate(s) extraction

	select TextGrid 'thisTextGrid$'
	#numWords = Count intervals where: word_tier, "is not equal to", ""
	numPhonemes = Count intervals where: phoneme_tier, "is not equal to", ""	

	# Extract Speech rate (Words per Minute, including pauses)
	if extract_speech_rate
		audio_duration = Get total duration
		#speech_rate_words = numWords/(audio_duration/60)
		speech_rate_phonemes = numPhonemes/(audio_duration/60)

		# Write mean speech rate to result
		#resultline$ += "'speech_rate_words:10',"
		resultline$ += "'speech_rate_phonemes:10',"

		# PRINTING TO CONSOLE
		if write_info_to_console
			#appendInfoLine: "words per minute (incl. pause intervals): ", speech_rate_words, " (", numWords, " words, ", audio_duration,"s)"
			appendInfoLine: "phonemes per minute (incl. pause intervals): ", speech_rate_phonemes, " (", numPhonemes, " phonemes, ", audio_duration,"s)"
		endif
	endif
	
	# Extract Articulation rate (Words per Minute, excluding pauses)
	if extract_articulation_rate
		#speaking_duration_words = Get total duration of intervals where: word_tier, "is not equal to", ""
		#articulation_rate_words = numWords/(speaking_duration_words/60)

		speaking_duration_phonemes = Get total duration of intervals where: phoneme_tier, "is not equal to", ""
		articulation_rate_phonemes = numPhonemes/(speaking_duration_phonemes/60)

		# Write mean articulation rate to result
		#resultline$ += "'articulation_rate_words:10',"
		resultline$ += "'articulation_rate_phonemes:10',"

		# PRINTING TO CONSOLE
		if write_info_to_console
			#appendInfoLine: "words per minute (excl. pause intervals): ", articulation_rate_words, " (", numWords, " words, ", speaking_duration_words,"s)"
			appendInfoLine: "phonemes per minute (excl. pause intervals): ", articulation_rate_phonemes, " (", numPhonemes, " phonemes, ", speaking_duration_phonemes,"s)"
		endif
	endif

	### Vowel duration analysis

	for d_p from 1 to size (duration_phonemes$#)
		select TextGrid 'thisTextGrid$'
		plus Sound 'thisSound$'

		curPhoneme$ = duration_phonemes$# [d_p]
	
		# Extract formant frequencies of current phoneme
		Extract intervals where... phoneme_tier no "is equal to" 'curPhoneme$'

		# If selected sound is thisSound, the extraction failed
		if selected ("Sound") == thisSound
			# Print to console even if printing is off
			appendInfoLine: "Could not extract duration of ", curPhoneme$, " for ", thisSpeaker$, ": ", curPhoneme$, " was never said."
			isValid = 0
			reasonsInvalid$ [reasonsInvalidIndex] = "No 'curPhoneme$' in phoneme tier" 
			reasonsInvalidIndex += 1
		else
			n = numberOfSelected ("Sound")
			totDuration = 0

			for i from 1 to n
				sound[i] = selected ("Sound", i)
			endfor

			for i from 1 to n
				selectObject: sound[i]

				# Create the Formant Object (default settings)
				To Formant (burg)... 0 5 max_formant_freq 0.025 50

				duration = Get total duration
				totDuration += duration

				Remove
				selectObject: sound[i]
				Remove
			endfor

			averageDuration = totDuration / n

			# PRINTING TO CONSOLE
			if write_info_to_console
				appendInfoLine: "mean duration of ", curPhoneme$, ": ", averageDuration, " seconds"
			endif

			# Add average duration of current phoneme to result
			resultline$ += "'averageDuration:10',"
		endif
	endfor

	### Vowel Formant analysis

	# keep track of the current index of the list of formant indices of all phonemes
	phoneme_formant_index = 0

	# Call formant frequency extraction for each phoneme
	for p from 1 to size (phonemes$#)
		select TextGrid 'thisTextGrid$'
		plus Sound 'thisSound$'

		curPhoneme$ = phonemes$# [p]
		
		num_formants = num_formants_per_phoneme[p]

		# PRINTING TO CONSOLE 
		if write_info_to_console
			appendInfoLine: newline$, "Extracting formants of ", curPhoneme$
		endif
	
		# Extract formant frequencies of current phoneme
		Extract intervals where... phoneme_tier no "is equal to" 'curPhoneme$'

		# if selected sound is thisSound, it means the extraction failed
		# Should only happen if the required phoneme was never pronounced
		if selected ("Sound") == thisSound
			appendInfoLine: thisSpeaker$, " is invalid: did not say any ", curPhoneme$
			isValid = 0
			reasonsInvalid$ [reasonsInvalidIndex] = "No 'curPhoneme$' in phoneme tier" 
			reasonsInvalidIndex += 1
		else
			n = numberOfSelected ("Sound")

			for i from 1 to n
				sound[i] = selected ("Sound", i)
			endfor

			formantMatrix## = zero## (n, num_formants)

			for i from 1 to n
				selectObject: sound[i]

				# Create the Formant Object (default settings)
				To Formant (burg)... 0 5 max_formant_freq 0.025 50

				dur = Get total duration
				midpoint = dur/2

				for f from 1 to num_formants
					f_idx = phoneme_formant_indices[phoneme_formant_index + f]
					#formant = Get value at time... f_idx midpoint Hertz Linear
					formant = Get value at time: f_idx, midpoint, unit$, "linear"

					if formant = undefined
						formant = -1
					endif

					formantMatrix [i, f] = formant 
				endfor 	
				Remove
				selectObject: sound[i]
				Remove
			endfor

			for f from 1 to num_formants
				f_idx = phoneme_formant_indices[phoneme_formant_index + f]

				sumOfFormant = 0
				numValidFormants = 0

				for i from 1 to n
					if formantMatrix [i, f] != -1
						
						# PRINTING TO CONSOLE
						if write_info_to_console
							appendInfoLine: "(", i, ") F", f_idx, ": ", formantMatrix [i, f], unit$
						endif

						numValidFormants += 1
						sumOfFormant += formantMatrix [i, f]
					endif
				endfor

				if numValidFormants != 0
					averageFormant = sumOfFormant / numValidFormants

					# Add current formant of current phoneme to result
					resultline$ += "'averageFormant:10',"

					# PRINTING TO CONSOLE
					if write_info_to_console
						appendInfoLine: "Average F", f_idx, ": ", averageFormant
					endif
				else 
					appendInfoLine: thisSpeaker$, " is invalid: PRAAT could not measure any formant frequencies for formant F", f_idx, " of ", curPhoneme$, "."
					isValid = 0
					reasonsInvalid$ [reasonsInvalidIndex] = "F'f_idx' is always undefined for 'curPhoneme$'"
					reasonsInvalidIndex += 1
				endif
			endfor
		endif	
		phoneme_formant_index += num_formants
	endfor

	### Diphthongization analysis

	# keep track of the current index of the list of formant indices of all diphthongs
	diphthong_formant_index = 0

	# Call diphthongization extraction for each diphthong
	for d from 1 to size (diphthongs$#)
		select TextGrid 'thisTextGrid$'
		plus Sound 'thisSound$'

		curDiphthong$ = diphthongs$# [d]

		num_formants_diph = num_formants_per_diphthong[d]

		# PRINTING TO CONSOLE
		if write_info_to_console
			appendInfoLine: newline$, "Extracting diphthongization of ", curDiphthong$
		endif
	
		# Extract formant frequencies of current diphthong
		Extract intervals where... phoneme_tier no "is equal to" 'curDiphthong$'

		# If selected sound is thisSound, it means the extraction failed 
		# Should only happen if the required diphthong was never pronounced
		if selected ("Sound") == thisSound
			appendInfoLine: thisSpeaker$, " is invalid: did not say any ", curDiphthong$
			isValid = 0
			reasonsInvalid$ [reasonsInvalidIndex] = "No 'curDiphthong$' in phoneme tier"
			reasonsInvalidIndex += 1
		else
			n = numberOfSelected ("Sound")

			for i from 1 to n
				sound[i] = selected ("Sound", i)
			endfor

			diphthongizationMatrix## = zero## (n, num_formants_diph)

			for i from 1 to n
				selectObject: sound[i]

				# Create the Formant Object (default settings)
				To Formant (burg)... 0 5 max_formant_freq 0.025 50

				dur = Get total duration
				start = dur/5
				end = dur - start

				for f from 1 to num_formants_diph
					f_idx = diphthong_formant_indices[diphthong_formant_index + f]

					#formant_start = Get value at time... f_idx start Hertz Linear
					formant_start = Get value at time: f_idx, start, unit$, "linear"
					#formant_end = Get value at time... f_idx end Hertz Linear
					formant_end = Get value at time: f_idx, end, unit$, "linear"

					if formant_start = undefined
						diphthongizationMatrix [i, f] = -1
					elsif formant_end = undefined
						diphthongizationMatrix [i, f] = -1
					else
						diphthongizationMatrix [i, f] = formant_end - formant_start
					endif
				endfor 	
				Remove
				selectObject: sound[i]
				Remove

			# Next occurrence
			endfor

			for f from 1 to num_formants_diph
				f_idx = diphthong_formant_indices[diphthong_formant_index + f]

				sumOfDeltas = 0
				numValidDeltas = 0

				for i from 1 to n
					if diphthongizationMatrix [i, f] != -1
						
						# PRINTING TO CONSOLE
						if write_info_to_console
							appendInfoLine: "(", i, ") ΔF", f_idx, ": ", diphthongizationMatrix [i, f], unit$
						endif

						numValidDeltas += 1
						sumOfDeltas += diphthongizationMatrix [i, f]
					endif
				endfor

				if numValidDeltas != 0
					averageDelta = sumOfDeltas / numValidDeltas

					# Add current delta of current diphthong to result
					resultline$ += "'averageDelta:10',"

					# PRINTING TO CONSOLE
					if write_info_to_console
						appendInfoLine: "Average ΔF", f_idx, ": ", averageDelta
					endif
				else 
					appendInfoLine: thisSpeaker$, " is invalid: PRAAT could not measure any frequencies for formant F", f_idx, " of ", curDiphthong$, "."
					isValid = 0
					reasonsInvalid$ [reasonsInvalidIndex] = "Delta F'f_idx' is always undefined for 'curDiphthong$'"
					reasonsInvalidIndex += 1	
				endif
				
			# Next formant
			endfor
		endif
		diphthong_formant_index += num_formants_diph	

		# Next diphthong
	endfor

	# Remove last comma (prevent empty extra column from being added)
	resultline$ = left$ (resultline$, length (resultline$) - 1)
	resultline$ += "'newline$'"

	# Save result to file
	if isValid == 1
		fileappend "'resultfile$'" 'resultline$'
	else 
		# Format reasons by putting "AND" between each entry 
		reasonInvalid$ = reasonsInvalid$ [1]
		for reason from 2 to (reasonsInvalidIndex - 1)
			reasonInvalid$ += " AND " + reasonsInvalid$ [reason]
		endfor

		# Save invalid filename and reason to excluded data file
		appendInfoLine: "Speaker ", thisSpeaker$, " is invalid and has been excluded from the result." 
		exclusion_line$ = "'thisSpeaker$'," + "'reasonInvalid$'" + "'newline$'"
		fileappend "'excluded_data_file$'" 'exclusion_line$'
	endif 

	# Remove initial sound and textGrid from object list
	select TextGrid 'thisTextGrid$'
	plus Sound 'thisSound$'
	Remove

	# (next file)

endfor

# PRINTING TO CONSOLE
if write_info_to_console
	appendInfoLine: newline$, newline$, "Finished."
endif
