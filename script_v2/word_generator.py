''' Extracts lists of words from a given input to be used for later vocabulary
    generation or for creating tokenized datasets.
    Supports functionality for handling different file types and
    filtering/processing of this input.
'''

from __future__ import division, print_function

import MeCab
import os
import unicodedata
import neologdn
import numpy as np
import re
import stanfordnlp

from multiprocessing import Pool
from text_unidecode import unidecode
from tokens import RE_MENTION
from filter_utils import (
    convert_linebreaks,
    convert_nonbreaking_space,
    correct_length,
    extract_emojis,
    mostly_english,
    non_english_user,
    process_word,
    punct_word,
    remove_control_chars,
    remove_variation_selectors,
    separate_emojis_and_text)
from functools import partial
from nltk.tokenize.casual import TweetTokenizer
from operator import itemgetter
from tqdm import tqdm

SPACE_RE = re.compile(r'[\s\u3000]+')
# Only catch retweets in the beginning of the tweet as those are the
# automatically added ones.
# We do not want to remove tweets like "Omg.. please RT this!!"
RETWEET_RE = re.compile(r'^[rR][tT]')

# Use fast and less precise regex for removing tweets with URLs
# It doesn't matter too much if a few tweets with URL's make it through
URLS_RE = re.compile(r'https?://|www\.')

MENTION_RE = re.compile(RE_MENTION)
ALLOWED_CONVERTED_UNICODE_PUNCTUATION = """!"#$'()+,-.:;<=>?@`~"""


class StanfordTokenizer():
    def __init__(self, lang, model_dir, processors='tokenize,mwt,pos,lemma'):
        self.nlp = stanfordnlp.Pipeline(processors=processors,
                                        models_dir=model_dir,
                                        lang=lang)

    def tokenize(self, text, lemma=False, lower=True):

        def word_text(word):
            text = word.lemma if lemma else word.text
            if lower:
                text = text.lower()
            return text

        return [word_text(word) for sent in self.nlp(text).sentences for word in sent.words]


class MeCabTokenizer():
    def __init__(self, stem=False, neologd=False, neologdn=False):
        option = ""
        if stem:
            option += "-F\s%f[6] -U\s%m -E\\n"
        else:
            option += "-Owakati"

        if neologd:
            option += " -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"

        self.tagger = MeCab.Tagger(option)

        if neologdn:
            self.neologdn = True
        else:
            self.neologdn = False

    def tokenize(self, text):
        if self.neologdn:
            text = neologdn.normalize(text)
        return self.tagger.parse(text).split()


def get_default_tokenizer(lang, model_dir='/data/stanfordnlp_resources'):
    if lang == 'ja':
        return MeCabTokenizer()
    elif lang in ('ar', 'zh'):
        return StanfordTokenizer(lang, model_dir, processors="tokenize,mwt")
    else:
        return TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)


class WordGenerator():
    ''' Cleanses input and converts into words. Needs all sentences to be in
        Unicode format. Has subclasses that read sentences differently based on
        file type.

    Takes a generator as input. This can be from e.g. a file.
    unicode_handling in ['ignore_sentence', 'convert_punctuation', 'allow']
    unicode_handling in ['ignore_emoji', 'ignore_sentence', 'allow']
    '''
    def __init__(self,
                 file_path,
                 lang,
                 norm_unicode_text=False,
                 allow_unicode_text=False,
                 ignore_emojis=True,
                 remove_variation_selectors=True,
                 break_replacement=True,
                 processes=1,
                 chunksize=100):
        self.file_path = file_path
        self.lang = lang
        self.tokenizer = None
        self.norm_unicode_text = norm_unicode_text
        self.allow_unicode_text = allow_unicode_text
        self.remove_variation_selectors = remove_variation_selectors
        self.ignore_emojis = ignore_emojis
        self.break_replacement = break_replacement
        self.processes = processes
        self.chunksize = chunksize
        self.reset_stats()

    def get_words(self, sentence):
        """ Tokenizes a sentence into individual words.
            Converts Unicode punctuation into ASCII if that option is set.
            Ignores sentences with Unicode if that option is set.
            Returns an empty list of words if the sentence has Unicode and
            that is not allowed.
        """

        sentence = sentence.strip().lower()

        if self.break_replacement:
            sentence = convert_linebreaks(sentence)

        if self.remove_variation_selectors:
            sentence = remove_variation_selectors(sentence)

        if self.tokenizer is None:
            self.tokenizer = get_default_tokenizer(self.lang)

        words = self.tokenizer.tokenize(sentence)
        if self.norm_unicode_text:
            converted_words = []
            for w in words:
                accept_sentence, c_w = self.convert_unicode_word(w)
                # Unicode word detected and not allowed
                if not accept_sentence:
                    return []
                else:
                    converted_words.append(c_w)
            words = converted_words

        words = [process_word(w) for w in words]
        return words

    def check_ascii(self, word):
        """ Returns whether a word is ASCII """

        try:
            word.encode('ascii')
            return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False

    def convert_unicode_punctuation(self, word):
        word_converted_punct = []
        for c in word:
            decoded_c = unidecode(c).lower()
            if len(decoded_c) == 0:
                # Cannot decode to anything reasonable
                word_converted_punct.append(c)
            else:
                # Check if all punctuation and therefore fine
                # to include unidecoded version
                allowed_punct = punct_word(
                        decoded_c,
                        punctuation=ALLOWED_CONVERTED_UNICODE_PUNCTUATION)

                if allowed_punct:
                    word_converted_punct.append(decoded_c)
                else:
                    word_converted_punct.append(c)
        return ''.join(word_converted_punct)

    def convert_unicode_word(self, word):
        """ Converts Unicode words to ASCII using unidecode. If Unicode is not
            allowed (set as a variable during initialization), then only
            punctuation that can be converted to ASCII will be allowed.
        """
        if self.check_ascii(word):
            return True, word

        # First we ensure that the Unicode is normalized so it's
        # always a single character.
        word = unicodedata.normalize("NFKC", word)

        # Convert Unicode punctuation to ASCII equivalent. We want
        # e.g. u"\u203c" (double exclamation mark) to be treated the same
        # as u"!!" no matter if we allow other Unicode characters or not.
        word = self.convert_unicode_punctuation(word)

        if self.ignore_emojis:
            _, word = separate_emojis_and_text(word)

        # If conversion of punctuation and removal of emojis took care
        # of all the Unicode or if we allow Unicode then everything is fine
        if self.check_ascii(word) or self.allow_unicode_text:
            return True, word
        else:
            # Sometimes we might want to simply ignore Unicode sentences
            # (e.g. for vocabulary creation). This is another way to prevent
            # "polution" of strange Unicode tokens from low quality datasets
            return False, ''

    def data_preprocess_filtering(self, line, iter_i):
        """ To be overridden with specific preprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            preprocessed text.

            Runs prior to tokenization.
        """
        return True, line, {}

    def data_postprocess_filtering(self, words, iter_i):
        """ To be overridden with specific postprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            postprocessed text.

            Runs after tokenization.
        """
        return True, words, {}

    def extract_valid_sentence_words(self, line):
        """ Line may either a string of a list of strings depending on how
            the stream is being parsed.
            Domain-specific processing and filtering can be done both prior to
            and after tokenization.
            Custom information about the line can be extracted during the
            processing phases and returned as a dict.
        """

        info = {}

        pre_valid, pre_line, pre_info = \
            self.data_preprocess_filtering(line, self.stats['total'])

        info.update(pre_info)
        if not pre_valid:
            self.stats['pretokenization_filtered'] += 1
            return False, [], info

        words = self.get_words(pre_line)
        if len(words) == 0:
            self.stats['unicode_filtered'] += 1
            return False, [], info

        post_valid, post_words, post_info = \
            self.data_postprocess_filtering(words, self.stats['total'])

        info.update(post_info)
        if not post_valid:
            self.stats['posttokenization_filtered'] += 1

        return post_valid, post_words, info

    def generate_array_from_input(self):
        sentences = []
        for words in self:
            sentences.append(words)
        return sentences

    def reset_stats(self):
        self.stats = {'pretokenization_filtered': 0,
                      'unicode_filtered': 0,
                      'posttokenization_filtered': 0,
                      'total': 0,
                      'valid': 0}

    def __iter__(self):
        if self.file_path is None:
            raise ValueError("Stream should be set before iterating over it!")

        if self.processes > 1:
            pool = Pool(self.processes)
            map_func = partial(pool.imap_unordered, chunksize=self.chunksize)
        else:
            pool = None
            map_func = map

        try:
            with open(self.file_path) as stream:
                for (valid, words, info) in tqdm(map_func(self.extract_valid_sentence_words, stream)):
                    # print("rnmb", words)
                    # Words may be filtered away due to unidecode etc.
                    # In that case the words should not be passed on.
                    if valid and len(words):
                        self.stats['valid'] += 1
                        yield words, info

                    self.stats['total'] += 1
                    # print("cnmb", words, "\n")
        finally:
            if pool is not None:
                pool.close()


class TweetWordGenerator(WordGenerator):
    ''' Returns np array or generator of ASCII sentences for given tweet input.
        Any file opening/closing should be handled outside of this class.
    '''
    def __init__(self,
                 file_path,
                 lang,
                 emojis=None,
                 english_words=None,
                 norm_unicode_text=True,
                 allow_unicode_text=True,
                 ignore_retweets=True,
                 ignore_url_tweets=True,
                 ignore_mention_tweets=False,
                 processes=1,
                 chunksize=100):

        emojis_len = [(e, len(e)) for e in emojis]
        emojis_len_decending = [e[0] for e in sorted(emojis_len, key=itemgetter(1), reverse=True)]
        self.emojis = emojis_len_decending
        self.english_words = english_words
        self.ignore_retweets = ignore_retweets
        self.ignore_url_tweets = ignore_url_tweets
        self.ignore_mention_tweets = ignore_mention_tweets

        WordGenerator.__init__(self, file_path, lang, ignore_emojis=False,
                               norm_unicode_text=norm_unicode_text,
                               allow_unicode_text=allow_unicode_text,
                               processes=processes, chunksize=chunksize)

    def validated_tweet(self, text):
        ''' A bunch of checks to determine whether the tweet is valid.
            Also returns emojis contained by the tweet.
        '''

        # Ordering of validations is important for speed
        # If it passes all checks, then the tweet is validated for usage

        if self.ignore_retweets and RETWEET_RE.search(text):
            return False

        if self.ignore_url_tweets and URLS_RE.search(text):
            return False

        if self.ignore_mention_tweets and MENTION_RE.search(text):
            return False

        return True

    def data_preprocess_filtering(self, line, iter_i):
        text = line.strip()
        valid = self.validated_tweet(text)
        if valid:
            text, emojis = extract_emojis(text, self.emojis)

            text = MENTION_RE.sub('', text)
            text = RETWEET_RE.sub('', text)
            text = URLS_RE.sub('', text)
            text = SPACE_RE.sub(' ', text)
            text = text.replace('&amp', '&')
            text = text.strip()
        else:
            text = ''
            emojis = []
        return valid, text, {'emojis': emojis}

    def data_postprocess_filtering(self, words, iter_i):
        valid_length = correct_length(words, 1, None)
        valid_english, n_words, n_english = mostly_english(words,
                                                           self.english_words)
        return True, words, {'length': len(words),
                             'n_normal_words': n_words,
                             'n_english': n_english}

        if valid_length and valid_english:
            return True, words, {'length': len(words),
                                 'n_normal_words': n_words,
                                 'n_english': n_english}
        else:
            return False, [], {'length': len(words),
                               'n_normal_words': n_words,
                               'n_english': n_english}
