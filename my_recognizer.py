import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word in range(0,len(test_set.get_all_Xlengths())):

        wordScore = {}
        current_word_sequences, current_sequ_length = test_set.get_item_Xlengths(word)
        for word,model in models.items():
            try:
                score = model.score(current_word_sequences,current_sequ_length)
                #print(score,"",word)
                wordScore[word] = score
                #print(current_word_feature_lists_sequences)
                #print(current_sequences_length)
                #print(models.items())
            except Exception as e:
                #print(e)
                pass
        probabilities.append(wordScore)
        guesses.append(calc_best_score(wordScore))
    return probabilities, guesses




    # return probabilities, guesses
def calc_best_score(word_log_likelihoods):

        # Max of dictionary of values by comparing each item by value at index

    return max(word_log_likelihoods, key=word_log_likelihoods.get)
