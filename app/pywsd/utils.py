# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python Word Sense Disambiguation (pyWSD): Misc utility functions
# Copyright (C) 2014-2018 alvations
# For license information, see LICENSE.md

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag


import re
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
_treebank_word_tokenizer = TreebankWordTokenizer()

improved_open_quote_regex = re.compile(u'([«“‘„]|[`]+)', re.U)
improved_close_quote_regex = re.compile(u'([»”’]+)', re.U)
improved_punct_regex = re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U)
_treebank_word_tokenizer.STARTING_QUOTES.insert(0, (improved_open_quote_regex, r' \1 '))
_treebank_word_tokenizer.ENDING_QUOTES.insert(0, (improved_close_quote_regex, r' \1 '))
_treebank_word_tokenizer.PUNCTUATION.insert(0, (improved_punct_regex, r'\1 \2 \3 '))

def word_tokenize(text, language='english', preserve_line=False):
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [token for sent in sentences
            for token in _treebank_word_tokenizer.tokenize(sent)]


SS_PARAMETERS_TYPE_MAP = {'definition':str,
                          'lemma_names':list, # list(str)
                          'examples':list,
                          'hypernyms':list,
                          'hyponyms': list,
                          'member_holonyms':list,
                          'part_holonyms':list,
                          'substance_holonyms':list,
                          'member_meronyms':list,
                          'substance_meronyms': list,
                          'part_meronyms':list,
                          'similar_tos':list
                          }

def remove_tags(text):
    """ Removes <tags> in angled brackets from text. """
    import re
    tags = {i:" " for i in re.findall("(<[^>\n]*>)",text.strip())}
    no_tag_text = reduce(lambda x, kv:x.replace(*kv), tags.iteritems(), text)
    return " ".join(no_tag_text.split())

def offset_to_synset(offset):
    """
    Look up a synset given offset-pos
    (Thanks for @FBond, see http://moin.delph-in.net/SemCor)
    >>> synset = offset_to_synset('02614387-v')
    >>> print '%08d-%s' % (synset.offset, synset.pos)
    >>> print synset, synset.definition
    02614387-v
    Synset('live.v.02') lead a certain kind of life; live in a certain style
    """
    return wn._synset_from_pos_and_offset(str(offset[-1:]), int(offset[:8]))

def semcor_to_synset(sensekey):
    """
    Look up a synset given the information from SemCor sensekey format.
    (Thanks for @FBond, see http://moin.delph-in.net/SemCor)
    >>> ss = semcor_to_offset('live%2:42:06::')
    >>> print '%08d-%s' % (ss.offset, ss.pos)
    >>> print ss, ss.definition
    02614387-v
    Synset('live.v.02') lead a certain kind of life; live in a certain style
    """
    return wn.lemma_from_key(sensekey).synset

def semcor_to_offset(sensekey):
    """
    Converts SemCor sensekey IDs to synset offset.
    >>> print semcor_to_offset('live%2:42:06::')
    02614387-v
    """
    synset = wn.lemma_from_key(sensekey).synset
    offset = '%08d-%s' % (synset.offset, synset.pos)
    return offset



porter = PorterStemmer()
wnl = WordNetLemmatizer()

def lemmatize(ambiguous_word, pos=None, neverstem=False,
              lemmatizer=wnl, stemmer=porter):
    """
    Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.

    This is to handle the case where users input a surface word as an ambiguous
    word and the surface word is a not a lemma.
    """
    # Try to be a little smarter and use most frequent POS.
    pos = pos if pos else penn2morphy(pos_tag([ambiguous_word])[0][1],
                                     default_to_noun=True)
    lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    stem = stemmer.stem(ambiguous_word)
    # Ensure that ambiguous word is a lemma.
    if not wn.synsets(lemma):
        if neverstem:
            return ambiguous_word
        if not wn.synsets(stem):
            return ambiguous_word
        else:
            return stem
    else:
        return lemma


def penn2morphy(penntag, returnNone=False, default_to_noun=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        if returnNone:
            return None
        elif default_to_noun:
            return 'n'
        else:
            return ''

def lemmatize_sentence(sentence, neverstem=False, keepWordPOS=False,
                       tokenizer=word_tokenize, postagger=pos_tag,
                       lemmatizer=wnl, stemmer=porter):
    words, lemmas, poss = [], [], []
    for word, pos in postagger(tokenizer(sentence)):
        pos = penn2morphy(pos)
        lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer, stemmer))
        poss.append(pos)
        words.append(word)
    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]
    return lemmas

def synset_properties(synset, parameter):
    """
    Making from NLTK's WordNet Synset's properties to function.
    Note: This is for compatibility with NLTK 2.x
    """
    return_type = SS_PARAMETERS_TYPE_MAP[parameter]
    func = 'synset.' + parameter
    return eval(func) if isinstance(eval(func), return_type) else eval(func)()

def has_synset(word):
    """" Returns a list of synsets a word after lemmatization. """
    return wn.synsets(lemmatize(word, neverstem=True))

# To check default parameters of simple_lesk()
## a = inspect.getargspec(simple_lesk)
## print zip(a.args[-len(a.defaults):],a.defaults)
