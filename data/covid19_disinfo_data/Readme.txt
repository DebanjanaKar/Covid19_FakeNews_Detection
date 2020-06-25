Description of the dataset
==========================
The COVID-19 Infodemic Twitter dataset consists of manually annotated tweets collected during the COVID-19 pandemic from all over the world. The dataset consists of tweets for English and Arabic and includes seven types of annotations (for more details, please refer to our paper [1]): 


** Question/Task 1: Does the tweet contain a verifiable factual claim?
Labels:
• YES: if it contains a verifiable factual claim;
• NO: if it does not contain a verifiable factual claim;
• Don't know or can't judge: the content of the tweet does not have enough information to make a judgment. It is recommended to categorize the tweet using this label when the content of the tweet is not understandable at all. For example, it uses a language (i.e., non-English) or references that it is difficult to understand;

** Question/Task 2: To what extent does the tweet appear to contain false information?
Labels:
1. NO, definitely contains no false information
2. NO, probably contains no false information
3. Not sure
4. YES, probably contains false information
5. YES, definitely contains false information

** Question/Task 3: Will the tweet's claim have an effect on or be of interest to the general public?
Labels:
1. NO, definitely not of interest
2. NO, probably not of interest
3. Not sure
4. YES, probably of interest
5. YES, definitely of interest

** Question/Task 4: To what extent does the tweet appear to be harmful to the society, person(s), company(s) or product(s)?
Labels:
1. NO, definitely not harmful
2. NO, probably not harmful
3. Not sure
4. YES, probably harmful
5. YES, definitely harmful

** Question/Task 5: Do you think that a professional fact-checker should verify the claim in the tweet?
Labels:
1. NO, no need to check
2. NO, too trivial to check
3. YES, not urgent
4. YES, very urgent
5. Not sure

** Question/Task 6: Is the tweet harmful to society and why?
Labels:
A. NO, not harmful
B. NO, joke or sarcasm
C. Not sure
D. YES, panic
E. YES, xenophobic, racist, prejudices, or hate-speech
F. YES, bad cure
G. YES, rumor, or conspiracy
H. YES, other

** Question/Task 7: Do you think that this tweet should get the attention of a government entity?
Labels:
A. NO, not interesting
B. Not sure
C. YES, categorized as in question 6
D. YES, other
E. YES, blame authorities
F. YES, contains advice
G. YES, calls for action
H. YES, discusses action taken
I. YES, discusses cure
J. YES, asks question

LIST OF VERSIONS
================
v1.0 [2020/05/01]: initial distribution of the annotated dataset
* English data: 504 tweets
* Arabic data: 218 tweets



CONTENTS OF THE DISTRIBUTION AND DATA FORMAT
===========================
The directory contains the following two sub-directories:
* Readme.txt this file

1. "English": This directory contains tab-separated values (i.e., TSV) file, and one JSON file. The TSV file stores ground-truth annotations for the aforementioned tasks. The data format of these files is described in detail below. Each line in the JSON file corresponds to data from a single tweet stored in JSON format (as downloaded from Twitter).  

2. "Arabic": Similarly to English, this directory contains one TSV file and one JSON file using the same format. 

Format of the TSV files under the "annotations" directory
---------------------------------------------------------
Each TSV file in this directory contains the following columns, separated by a tab:

* tweet_id: corresponds to the actual tweet id from Twitter.
* tweet_text: corresponds to the original text of a given tweet as downloaded from Twitter.
* q*_label (column 3-9): corresponds to the label for question 1 to 7.


Note that there are NA (i.e., null) entries in the TSV files that simply indicate "not applicable" cases. We label NA for question 2 to 5 when question 1 is labeled as NO. 

** Examples ** 
Please don't take hydroxychloroquine (Plaquenil) plus Azithromycin for #COVID19 UNLESS your doctor prescribes it. Both drugs affect the QT interval of your heart and can lead to arrhythmias and sudden death, especially if you are taking other meds or have a heart condition.
Labels:
Q1: Yes; 
Q2: NO: probably contains no false info
Q3: YES: definitely of interest
Q4: NO: probably not harmful
Q5: YES:very-urgent
Q6: NO:not-harmful
	Q7: YES:discusses_cure

BREAKING: @MBuhari’s Chief Of Staff, Abba Kyari, Reportedly Sick, Suspected Of Contracting #Coronavirus | Sahara Reporters A top government source told SR on Monday that Kyari has been seriously “down” since returning from a trip abroad. READ MORE: https://t.co/Acy5NcbMzQ https://t.co/kStp4cmFlr.
Labels:
Q1: Yes; 
Q2: NO: probably contains no false info
Q3: YES: definitely of interest
Q4: NO: definitely not harmful
Q5: YES:not-urgent
Q6: YES:rumor
	Q7: YES:classified_as_in_question_6


STATISTICS
=========
Some statistics about the dataset

** Class label distribution for the English tweets:
Q1	504
no	209
yes	295
	
Q2	295
1_no_definitely_contains_no_false_info	47
2_no_probably_contains_no_false_info	171
3_not_sure	40
4_yes_probably_contains_false_info	25
5_yes_definitely_contains_false_info	12

	
Q3	295
1_no_definitely_not_of_interest	9
2_no_probably_not_of_interest	44
3_not_sure	7
4_yes_probably_of_interest	177
5_yes_definitely_of_interest	58

	
Q4	295
1_no_definitely_not_harmful	106
2_no_probably_not_harmful	66
3_not_sure	2
4_yes_probably_harmful	67
5_yes_definitely_harmful	54

	
Q5	295
no_no_need_to_check	77
no_too_trivial_to_check	57
yes_not_urgent	112
yes_very_urgent	49
	
Q6	504
no_joke_or_sarcasm	62
no_not_harmful	333
not_sure	2
yes_bad_cure	3
yes_other	25
yes_panic	23
yes_rumor_conspiracy	42
yes_xenophobic_racist_prejudices_or_hate_speech	14
		
Q7	504
no_not_interesting	319
not_sure	6
yes_asks_question	2
yes_blame_authorities	81
yes_calls_for_action	8
yes_classified_as_in_question_6	34
yes_contains_advice	9
yes_discusses_action_taken	12
yes_discusses_cure	5
yes_other	28


** Class label distribution of Arabic tweets:
Q1	218
no	78
yes	140
	
Q2	140
1_no_definitely_contains_no_false_info	31
2_no_probably_contains_no_false_info	62
3_not_sure	5
4_yes_probably_contains_false_info	40
5_yes_definitely_contains_false_info	2

	
Q3	140
1_no_definitely_not_of_interest	1
2_no_probably_not_of_interest	5
3_not_sure	9
4_yes_probably_of_interest	76
5_yes_definitely_of_interest	49

	
Q4	140
1_no_definitely_not_harmful	68
2_no_probably_not_harmful	21
3_not_sure	3
4_yes_probably_harmful	46
5_yes_definitely_harmful	2

	
Q5	140
no_no_need_to_check	22
no_too_trivial_to_check	55
yes_not_urgent	48
yes_very_urgent	15
	
Q6	218
no_joke_or_sarcasm	2
no_not_harmful	159
yes_bad_cure	1
yes_other	5
yes_panic	12
yes_rumor_conspiracy	33
yes_xenophobic_racist_prejudices_or_hate_speech	6
	
	
Q7	218
no_not_interesting	163
yes_blame_authorities	13
yes_calls_for_action	1
yes_classified_as_in_question_6	30
yes_contains_advice	1
yes_discusses_cure	6
yes_other	4


LICENSE
=======
This dataset is freely available for general research use.

CITATION
=========
If you use this data in your research, please consider citing the following paper:

[1] Firoj Alam, Shaden Shaar, Alex Nikolov, Hamdy Mubarak, Giovanni Da San Martino, Ahmed Abdelali, Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Kareem Darwish, Preslav Nakov, "Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society", arxiv, 2020, https://arxiv.org/pdf/2005.00033.pdf

@misc{alam2020fighting,
    title={Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society},
    author={Firoj Alam and Shaden Shaar and Alex Nikolov and Hamdy Mubarak and Giovanni Da San Martino and Ahmed Abdelali and Fahim Dalvi and Nadir Durrani and Hassan Sajjad and Kareem Darwish and Preslav Nakov},
    year={2020},
    eprint={2005.00033},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

CREDITS
=======
Firoj Alam, Qatar Computing Research Institute, HBKU
Shaden Shaar, Qatar Computing Research Institute, HBKU
Alex Nikolov, Sofia University
Hamdy Mubarak, Qatar Computing Research Institute, HBKU
Giovanni Da San Martino, Qatar Computing Research Institute, HBKU
Ahmed Abdelali, Qatar Computing Research Institute, HBKU
Fahim Dalvi, Qatar Computing Research Institute, HBKU
Nadir Durrani, Qatar Computing Research Institute, HBKU
Hassan Sajjad, Qatar Computing Research Institute, HBKU
Kareem Darwish, Qatar Computing Research Institute, HBKU
Preslav Nakov, Qatar Computing Research Institute, HBKU


CONTACT
======= 
Please contact tanbih@qcri.org


ACKNOWLEDGMENTS
===============

Thanks to the QCRI's [Crisis Computing](https://crisiscomputing.qcri.org/) team for facilitating us with [Micromappers](https://micromappers.qcri.org/).
