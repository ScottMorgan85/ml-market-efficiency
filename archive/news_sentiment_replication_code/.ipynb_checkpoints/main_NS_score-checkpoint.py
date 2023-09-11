'''
Program: Calculates the news sentiment score of text files
Last Modified: 09/17/2020

This code was developed in the study "Measuring News Sentiment" by Adam Shapiro, 
Moritz Sudhof, and Daniel Wilson. Journal of Econometrics. Please cite accordingly. 

###################### - COPYRIGHT - ##########################################
[Shapiro, Adam Hale, Moritz Sudhof, and Daniel J. Wilson. 2020. “Measuring News Sentiment.” 
FRB San Francisco Working Paper 2017-01. Available at https://doi.org/10.24148/wp2017-01.]

Copyright (c) 2020, Adam Shapiro, Moritz Sudhof, and Daniel Wilson and the Federal Reserve Bank of San Francisco. 
All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the 
following conditions are met: 

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the 
      following disclaimer. 

    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
      following disclaimer in the documentation and/or other materials provided with the distribution. 

    * Redistributions [and use] in source and binary forms must be for Noncommercial purposes only. “Noncommercial” 
      means not primarily intended for or directed towards commercial advantage or monetary compensation.

    * Neither the name “Federal Reserve Bank of San Francisco” nor the names of its contributors may be used to 
      endorse or promote products derived from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###############################################################################
'''

###################### - MANUAL SETTINGS - ####################################
function_dir = ''
data_dir = ''
output_dir = ''
doc_types = []

###################### - LIBRARIES - ##########################################
import os
import numpy as np
import pandas as pd
from collections import Counter

# Set directory to load in aux functions
os.chdir(function_dir)
import NS_functions
import importlib
importlib.reload(NS_functions)


##################### - START OF CODE - #######################################

# Function that loads in the texts and applies the scoring function
def score_all_articles(scoring_func):
    scores = []    
    
    for doc_type in doc_types:
        doc_dir = data_dir + doc_type + '//'
        os.chdir(doc_dir)
        fns = [fn for fn in os.listdir(doc_dir) if ".txt" in fn]

        for fn in fns:
            fn = os.path.join(doc_dir, fn)
            text = NS_functions.parse_news_text_file(fn)
            score = scoring_func(text)
            scores.append({
                "file": os.path.basename(fn),
                "sentiment": score,
                "type": doc_type,
                "text": text
            })
    return scores



# Set up our scoring function -- Vader augmented with the LM and HL lexicons
lm_lexicon = NS_functions.load_lm_lexicon()
hl_lexicon = NS_functions.load_hl_lexicon()


# News PMI + LM + HL Lexicon
news_lexicon_fn = "ns.vader.sentences.20k.csv"
news_lexicon = NS_functions.load_news_pmi_lexicon(news_lexicon_fn)
news_lm_hl_lexicon = NS_functions.combine_lexicons([lm_lexicon, hl_lexicon, news_lexicon])
news_lm_hl_scoring_func = NS_functions.get_lexicon_scoring_func(news_lm_hl_lexicon)
news_lm_hl_negated_scoring_func = NS_functions.get_negated_lexicon_scoring_func(news_lm_hl_lexicon)


# Get the new-win scores
scored_articles = score_all_articles(news_lm_hl_negated_scoring_func)

# Create df of scores
scores = [d['sentiment'] for d in scored_articles]
types = [d['type'] for d in scored_articles]
files = [d['file'] for d in scored_articles]
df = pd.DataFrame({'file':files, 'type':types, 'score':scores})


# Output .csv of news sentiment to data folder
os.chdir(output_dir)
df.to_csv('ns_scores.csv')


