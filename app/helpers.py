from __future__ import unicode_literals, division, print_function

from semantic import SemanticSentence

#from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
#from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


from collections import Counter
from itertools import chain
import unicodedata
import pandas as pd
import numpy as np
import string
import pickle
import spacy
import re
import pandas
import nltk
import copy
import random

def get_matching_ents(spacy_answer, spacy_text):
    """

    :param spacy_answer:
    :param spacy_text:
    :return:

    """
    return [ent for ent in spacy_text.ents if ent[-1].ent_type_ == spacy_answer[-1].ent_type_]

def check_question_quality(question, spacy_text):
    if question['spacy_answer'][-1].ent_type_ != '':
        print(question['spacy_answer'][-1].ent_type_)
        matching_ents = get_matching_ents(question['spacy_answer'], spacy_text)
        if len(matching_ents) >= 4:
            print("OK question")
            question.update({'quality': True})
            question.update({'matching_ents': matching_ents})
            print("QUESTION:", question['question'])
            print("ANSWER:", question['answer'])
            print("ENT TYPE:", question['spacy_answer'][-1].ent_type_)
            print("MATCHING ENTS:", question['matching_ents'])
        else:
            print("NOK question")
            question.update({'quality': False})
            question.update({'matching_ents': matching_ents})
            print("QUESTION:", question['question'])
            print("ANSWER:", question['answer'])
            print("ENT TYPE:", question['spacy_answer'][-1].ent_type_)
            print("MATCHING ENTS:", question['matching_ents'])
    else:
        print("OK question")
        question.update({'quality': True})
        question.update({'matching_ents': []})
        print("QUESTION:", question['question'])
        print("ANSWER:", question['answer'])
        print("ENT TYPE:", question['spacy_answer'][-1].ent_type_)
        print("MATCHING ENTS:", question['matching_ents'])
        
    return question


def predict_best_question(questions, model, nlp, top_n=1):
    max_pred = 0
    predictions = []
    for i, q in enumerate(questions):
        question_vector = SemanticSentence(q['sentence'], q['question'], q['answer'], nlp, srl=q['srl']).vector()
        pred = model.predict_proba(question_vector.reshape(1, -1))[0][1]
        predictions.append((q, pred))
    top_questions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    #remove duplicate sentences
    non_dup_top_questions = []
    for tq in top_questions:
        if tq[0]['sentence'] not in [q[0]['sentence'] for q in non_dup_top_questions]:
            non_dup_top_questions.append(tq)
    return [q[0] for q in non_dup_top_questions[:top_n]]

def get_best_sentences(text, num=1):
    sentence_count = num
    parser = PlaintextParser(text, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')
    return [str(s) for s in summarizer(parser.document, sentence_count)]

def unpunkt(text):
    return "".join([c if unicodedata.category(c)[0] != 'P' else ' ' for c in text])


def get_article_url(topic):
    topic_dict = {
        'art': ['https://en.wikipedia.org/wiki/Othello', 'https://en.wikipedia.org/wiki/Industrial_design', 'https://en.wikipedia.org/wiki/Outsider_art', 'https://en.wikipedia.org/wiki/To_the_Lighthouse', 'https://en.wikipedia.org/wiki/Architecture', 'https://en.wikipedia.org/wiki/Opera', 'https://en.wikipedia.org/wiki/Romeo_and_Juliet', 'https://en.wikipedia.org/wiki/Guernica_(Picasso)', 'https://en.wikipedia.org/wiki/Jewish_literature', 'https://en.wikipedia.org/wiki/The_Rules_of_the_Game', 'https://en.wikipedia.org/wiki/Conceptual_art', 'https://en.wikipedia.org/wiki/Duino_Elegies', 'https://en.wikipedia.org/wiki/Gregorian_chant', 'https://en.wikipedia.org/wiki/Cinema_of_France', 'https://en.wikipedia.org/wiki/Dance', 'https://en.wikipedia.org/wiki/Persian_art', 'https://en.wikipedia.org/wiki/Musical_instrument'],
        'people': ['https://en.wikipedia.org/wiki/William_Henry_Bragg', 'https://en.wikipedia.org/wiki/Georges_Clemenceau', 'https://en.wikipedia.org/wiki/Chanakya', 'https://en.wikipedia.org/wiki/Muhammad', 'https://en.wikipedia.org/wiki/Hayreddin_Barbarossa', 'https://en.wikipedia.org/wiki/Robert_Johnson', 'https://en.wikipedia.org/wiki/Robert_Johnson', 'https://en.wikipedia.org/wiki/Robert_Guiscard', 'https://en.wikipedia.org/wiki/Robert_Guiscard', 'https://en.wikipedia.org/wiki/Thomas_Edison', 'https://en.wikipedia.org/wiki/Lionel_Messi', 'https://en.wikipedia.org/wiki/Led_Zeppelin', 'https://en.wikipedia.org/wiki/William_Harvey', 'https://en.wikipedia.org/wiki/Jomo_Kenyatta', 'https://en.wikipedia.org/wiki/Abraham', 'https://en.wikipedia.org/wiki/Hokusai', 'https://en.wikipedia.org/wiki/Claude_L%C3%A9vi-Strauss', 'https://en.wikipedia.org/wiki/Ignacy_%C5%81ukasiewicz', 'https://en.wikipedia.org/wiki/Li_Ning', 'https://en.wikipedia.org/wiki/Pierre_Boulez', 'https://en.wikipedia.org/wiki/Laurence_Olivier', 'https://en.wikipedia.org/wiki/Eduard_Shevardnadze', 'https://en.wikipedia.org/wiki/Isaiah_Berlin', 'https://en.wikipedia.org/wiki/William_Pitt_the_Younger', 'https://en.wikipedia.org/wiki/Tertullian', 'https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart', 'https://en.wikipedia.org/wiki/Antoine_Lavoisier', 'https://en.wikipedia.org/wiki/Emperor_Taizu_of_Song'],
        'society': ['https://en.wikipedia.org/wiki/Afroasiatic_languages', 'https://en.wikipedia.org/wiki/Multilingualism', 'https://en.wikipedia.org/wiki/Yoruba_people', 'https://en.wikipedia.org/wiki/Wu_Chinese', 'https://en.wikipedia.org/wiki/Javanese_language', 'https://en.wikipedia.org/wiki/Looney_Tunes', 'https://en.wikipedia.org/wiki/Rights', 'https://en.wikipedia.org/wiki/University_of_London', 'https://en.wikipedia.org/wiki/Oedipus_complex', 'https://en.wikipedia.org/wiki/Orphanage', 'https://en.wikipedia.org/wiki/Commonwealth_of_Independent_States', 'https://en.wikipedia.org/wiki/Morse_code', 'https://en.wikipedia.org/wiki/Peasant', 'https://en.wikipedia.org/wiki/Honour', 'https://en.wikipedia.org/wiki/Ethnography', 'https://en.wikipedia.org/wiki/Middle_English'],
        'history': ['https://en.wikipedia.org/wiki/Archaeology', 'https://en.wikipedia.org/wiki/History_of_aviation', 'https://en.wikipedia.org/wiki/History_of_geography', 'https://en.wikipedia.org/wiki/Legal_history', 'https://en.wikipedia.org/wiki/History_of_transport', 'https://en.wikipedia.org/wiki/Archaeological_culture', 'https://en.wikipedia.org/wiki/Geological_history_of_Earth', 'https://en.wikipedia.org/wiki/History_of_atheism', 'https://en.wikipedia.org/wiki/Maritime_history'],
        'science': ['https://en.wikipedia.org/wiki/Substitution_reaction', 'https://en.wikipedia.org/wiki/Cloud', 'https://en.wikipedia.org/wiki/Heath', 'https://en.wikipedia.org/wiki/Europium', 'https://en.wikipedia.org/wiki/Sulfuric_acid', 'https://en.wikipedia.org/wiki/Caesium', 'https://en.wikipedia.org/wiki/Helium', 'https://en.wikipedia.org/wiki/Formaldehyde', 'https://en.wikipedia.org/wiki/Tundra', 'https://en.wikipedia.org/wiki/Propene', 'https://en.wikipedia.org/wiki/Hydrosphere', 'https://en.wikipedia.org/wiki/Jet_stream', 'https://en.wikipedia.org/wiki/Boron', 'https://en.wikipedia.org/wiki/Thunderstorm', 'https://en.wikipedia.org/wiki/Plateau', 'https://en.wikipedia.org/wiki/Uranium', 'https://en.wikipedia.org/wiki/Methane', 'https://en.wikipedia.org/wiki/Season', 'https://en.wikipedia.org/wiki/Chemical_substance', 'https://en.wikipedia.org/wiki/Potassium_nitrate', 'https://en.wikipedia.org/wiki/Antimony', 'https://en.wikipedia.org/wiki/El_Ni%C3%B1o%E2%80%93Southern_Oscillation'],
        'math': ['https://en.wikipedia.org/wiki/Algorithm', 'https://en.wikipedia.org/wiki/Tessellation', 'https://en.wikipedia.org/wiki/Plane_(geometry)', 'https://en.wikipedia.org/wiki/0', 'https://en.wikipedia.org/wiki/Three-dimensional_space', 'https://en.wikipedia.org/wiki/Axiom', 'https://en.wikipedia.org/wiki/Square', 'https://en.wikipedia.org/wiki/Topological_space', 'https://en.wikipedia.org/wiki/Pythagorean_theorem', 'https://en.wikipedia.org/wiki/Data_structure', 'https://en.wikipedia.org/wiki/Homology_(mathematics)', 'https://en.wikipedia.org/wiki/Trigonometry', 'https://en.wikipedia.org/wiki/Convex_set', 'https://en.wikipedia.org/wiki/Metric_space', 'https://en.wikipedia.org/wiki/Mathematics', 'https://en.wikipedia.org/wiki/Pi', 'https://en.wikipedia.org/wiki/Data_compression', 'https://en.wikipedia.org/wiki/Constant_(mathematics)', 'https://en.wikipedia.org/wiki/Naive_set_theory', 'https://en.wikipedia.org/wiki/Shape', 'https://en.wikipedia.org/wiki/Rational_number']
    }

    return random.choice(topic_dict[topic])

