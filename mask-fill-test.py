import python_utils
import json
import csv
import re
import pandas
import nameutils
from nameutils import MaskedNER, find_word_indices, string_match
import argparse
from argparse import ArgumentParser

def catch_key_error(dic, index):
  try:
     return dic[index]
  except KeyError:
    return False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--twts", 
                        help="choose which tweet file to use, include suffix. JSONL only.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    with open("word_changes_sample_1k.tsv") as fd: #random sample of 1k words
    #with open("word_changes.tsv") as fd:
        word_changes = list(csv.reader(fd, delimiter=";", quotechar='"'))

    word_changes = word_changes[1:] #rid first row

    wordchange_start = {}
    wordchange_end = {}

    for row in word_changes:
        wordchange_start.update({row[0]: row[2]}) #period 1
        wordchange_end.update({row[0]: row[3]}) #period 2 

    mner = MaskedNER(census = {})
    info = []
    i = 1

    print("Opening file")
    with open(f"{args.twts}") as reader:
        print("File opened")
        for entry in reader:
            if i%100 == 0:
                print(i)
            entry = json.loads(entry) #parse string as dictionary

            if entry['is_retweet']: #only unique tweets!
                i = i+1
                continue

            else:
                tokens = list(set(entry['normalized_tweet'].lower().split(' '))) #unique words in the current tweet

                for tk in tokens: #for every word in the tweet (tokens)
                    if entry['fasttext_langid'] == 'en' and entry['langid_langid'] == 'en' and entry['temp_bin'] == catch_key_error(wordchange_start, tk) and bool(string_match(entry['normalized_tweet'].lower(), {f'{tk}'})): #if the time of the tweet is the same as the time in the dictionary
                        #tweets_at_start[tk].append(entry['normalized_tweet'].lower()) #append to list with purpose unknown for now
                        top1k_start = mner.prob_name(mner.preprocess_data(entries=[entry['normalized_tweet'].lower()], span_sets=[string_match(entry['normalized_tweet'].lower(), {f'{tk}'})])[0][0], f'{tk}', True, 1000)
                        top10_start = mner.prob_name(mner.preprocess_data(entries=[entry['normalized_tweet'].lower()], span_sets=[string_match(entry['normalized_tweet'].lower(), {f'{tk}'})])[0][0], f'{tk}', True, 10) #unsure how to get index from this output alone
                        
                        try:
                            idx = list(top1k_start[1].keys()).index(tk) + 1
                        except:
                            idx = -1

                        info.append({"word": tk,
                            "bin": entry['temp_bin'],
                            "tweet": entry['normalized_tweet'],
                            "top_n": top10_start[1],                            
                            "prob_word": top1k_start[0],
                            "word_rank": idx}) #index + 1
                    
                    elif entry['fasttext_langid'] == 'en' and entry['langid_langid'] == 'en' and entry['temp_bin'] == catch_key_error(wordchange_end, tk) and bool(string_match(entry['normalized_tweet'].lower(), {f'{tk}'})): #if the time bins are the same
                        #tweets_at_end[tk].append(entry['normalized_tweet'].lower())    
                        top1k_end = mner.prob_name(mner.preprocess_data(entries=[entry['normalized_tweet'].lower()], span_sets=[string_match(entry['normalized_tweet'].lower(), {f'{tk}'})])[0][0], f'{tk}', True, 1000)
                        top10_end = mner.prob_name(mner.preprocess_data(entries=[entry['normalized_tweet'].lower()], span_sets=[string_match(entry['normalized_tweet'].lower(), {f'{tk}'})])[0][0], f'{tk}', True, 10) #unsure how to get index from this output alone
                        
                        try:
                            idx = list(top1k_end[1].keys()).index(tk) + 1
                        except:
                            idx = -1

                        info.append({"word": tk,
                            "bin": entry['temp_bin'],
                            "tweet": entry['normalized_tweet'],
                            "top_n": top10_end[1],                            
                            "prob_word": top1k_end[0],
                            "word_rank": idx}) #index + 1

                    else:
                        continue

                i = i+1
        
    
    ##----SAVE FILE----##
    with open(f"output-{args.twts}", 'w') as fin:
        fin.write('\n'.join(map(json.dumps, info)))

    
if __name__ == "__main__":
    main()