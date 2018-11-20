import numpy as np

from nltk import pos_tag
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin
from typing import Dict, List, Optional, Set


class TextPreprocessor(TransformerMixin):
    """
    Basic text pre-processing for nlp tasks.
    """

    # The language this TextPreprocessor supports.
    LANGUAGE: str = 'english'
    # Maps tree bank parts of speech tags to word net parts of speech tags.
    # I know very little about parts of speech.. Taken with good faith from:
    # https://stackoverflow.com/questions/5364493/lemmatizing-pos-tagged-words-with-nltk
    TAG_MAP: Dict = {
        'CC': None,
        'CD': wordnet.NOUN,
        'DT': None,
        'EX': wordnet.ADV,
        'FW': None,
        'IN': wordnet.ADV,
        'JJ': wordnet.ADJ,
        'JJR': wordnet.ADJ,
        'JJS': wordnet.ADJ,
        'LS': None,
        'MD': None,
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'NNP': wordnet.NOUN,
        'NNPS': wordnet.NOUN,
        'PDT': wordnet.ADJ,
        'POS': None,
        'PRP': None,
        'PRP$': None,
        'RB': wordnet.ADV,
        'RBR': wordnet.ADV,
        'RBS': wordnet.ADV,
        'RP': wordnet.ADJ,
        'SYM': None,
        'TO': None,
        'UH': None,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB,
        'WDT': None,
        'WP': None,
        'WP$': None,
        'WRB': None,
        '$': None,
        '#': None,
        '“': None,
        '”': None,
        '(': None,
        ')': None,
        ',': None,
        '.': None,
        ':': None
    }

    def __init__(self):
        """
        Creates a TextPreprocessor.
        """
        self._stop_words: Set = set(stopwords.words('english'))
        self._lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

    def fit(self, X: np.array, y=None) -> 'TextPreprocessor':
        """
        Fits the transformer to X and y.
        :param X: The numpy array of the training set.
        :param y: The numpy array of the target values.
        :return: self
        """
        # Not applicable to TextPreprocessor
        return self

    def transform(self, X: np.array) -> np.array:
        """
        Transform the text training set to pre-processed text.
        :param X: The np array of the training set to transform.
        :return: The transformed training set.
        """
        for index in range(X.shape[0]):  # type: int
            X[index] = self.process(X[index])
        return X

    def process(self, sentence: str) -> np.array:
        """
        Removes stop words and punctuation from a sentence, then lowercases and
        lemmatizes all of the remaining words.
        :param sentence: The sentence to process.
        :return: The processed sentence.
        """
        words: List = []
        lower_case_sentence: str = sentence.lower()
        for word, tag in pos_tag(wordpunct_tokenize(lower_case_sentence)):  # type: str, str
            if not word.isalpha():
                continue
            # Get rid of stop words
            if word in self._stop_words:
                continue
            wordnet_tag: str = TextPreprocessor.__get_wordnet_pos(tag)
            lemmatized_word: str = self._lemmatizer.lemmatize(
                word, wordnet_tag) if wordnet_tag else word
            words.append(lemmatized_word)
        return np.array(words)

    @staticmethod
    def __get_wordnet_pos(treebank_tag) -> Optional[str]:
        """
        Get the corresponding word net parts of speech tag from a tree bank
        parts of speech tag.
        :param treebank_tag: The tree bank parts of speech tag.
        :return: The word net parts of speech tag.
        """
        return TextPreprocessor.TAG_MAP.get(treebank_tag, None)
