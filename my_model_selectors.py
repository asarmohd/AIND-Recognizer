import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import array


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def calculateBIC(self, compNumber):
        """
        TODO
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        BICModel = self.base_model(compNumber)
        logL = BICModel.score(self.X, self.lengths)
        p = compNumber * (compNumber - 1) + (compNumber - 1) + 2 * self.X.shape[1] * compNumber
        BICscore = (-2 * logL) + (p * np.log(self.X.shape[0]))
        return BICscore

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        bScore = -1
        BICModel = None
        bestBICModel = None
        for compNumber in range(self.min_n_components, self.max_n_components + 1):
            try:
                bicScore = self.calculateBIC(compNumber)
                if bicScore > bScore:
                    bScore = bicScore
                    bestBICModel = self.base_model(compNumber)
            except Exception as e:
                continue
        return bestBICModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calculateDIC(self, compNumber):
        """
        TODO
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        DICModel = self.base_model(compNumber)
        logL = DICModel.score(self.X, self.lengths)
        total_logL = 0
        for word in self.words:
            word_x, word_x_lengths = self.hwords[word]
            total_logL += DICModel.score(word_x, word_x_lengths)
        avg_logL = total_logL / (len(self.words) - 1)
        dic_score = logL - avg_logL

        return dic_score

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        bScore = float("Inf")
        bestDICModel = None
        for compNumber in range(self.min_n_components, self.max_n_components + 1):
            try:
                dic_score = self.calculateDIC(compNumber)
                print(dic_score)
                if dic_score < bScore:
                    bScore = dic_score
                    bestDICModel = self.base_model(compNumber)
            except Exception as e:
                print(e)
                continue
        return bestDICModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def findScore(self, compNumber):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        scores = []
        kf= KFold(n_splits=2)

        for cvTrainIndex, cvTestIndex in kf.split(self.sequences):
            self.X, self.lengths = combine_sequences(cvTrainIndex, self.sequences)
            cvTestX, cvTestLen = combine_sequences(cvTestIndex, self.sequences)
            hmmModel = self.base_model(num_states=compNumber).fit(self.X, self.lengths)

            scores.append(hmmModel.score(cvTestX, cvTestLen))

        return np.mean(scores), hmmModel

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        bScore = float("Inf")
        bHmmModel = None
        for compNumber in range(self.min_n_components, self.max_n_components + 1):
            try:
                crrScore, model = self.findScore(compNumber)
                if crrScore < bScore:
                    best_score = crrScore
                    bHmmModel = model
            except Exception as e:
                continue
        return bHmmModel


