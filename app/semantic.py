from __future__ import unicode_literals, division, print_function


from collections import Counter
import pandas as pd
import numpy as np
import unicodedata
import pickle
import pywsd
import spacy
import nltk
import re
from spacy.matcher import Matcher
from spacy.attrs import TAG
import allen_srl


senna_dir='senna'

class DistractorSet(object):
    def __init__(self, question, text, spacy_text, nlp):
        self.nlp = nlp
        self.question = question
        self.raw_text = text
        self.spacy = spacy_text
        self.spacy_answer = self.question['spacy_answer']
        self.spacy_sentence = self.question['spacy_sentence']
        self.matching_ents = self.question['matching_ents']
        self.candidate_distractors = self.collect_distractors()
        self.distractors = []
        self.distractors = self.make_distractors()

    def collect_distractors(self):
        self.noun_chunks = list(self.spacy.noun_chunks)
        self.pos_pattern_matches = self.get_pos_pattern_matches()
        self.root_pos_matches = self.get_root_pos_matches()
        return self.matching_ents + self.noun_chunks + self.pos_pattern_matches + self.root_pos_matches

    def get_root_pos_matches(self):
        if self.spacy_answer.root.pos_ != 'NOUN':
            matcher = Matcher(self.nlp.vocab)
            matcher.add("root_pos", None, [{TAG: self.spacy_answer.root.tag_}])
            matches = matcher(self.spacy)
            return [self.spacy[m[1]:m[2]] for m in matches]
        else:
            return []

    def get_pos_pattern_matches(self):
        pos_tag_pattern = [w.tag_ for w in self.spacy_answer]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("answer_pattern", None, [{TAG: tag} for tag in pos_tag_pattern])
        matches = matcher(self.spacy)
        return [self.spacy[m[1]:m[2]] for m in matches]

    def filter_duplicates(self, distractors):
        non_duplicates = set()
        non_dup_set = set()
        
        for d in distractors:
            if d.text not in non_dup_set:
                non_duplicates.add(d)
                non_dup_set.add(d.text)
        return non_duplicates

    def remove_distractor_in_answer(self, distractors):
        return [d for d in distractors if d.text not in self.spacy_answer.text]

    def remove_answer_in_distractor(self, distractors):
        return [d for d in distractors if self.spacy_answer.text not in d.text]

    def remove_common_root_with_answer(self, distractors):
        return [d for d in distractors if d.root.text != self.spacy_answer.root.text]

    def get_similarities_to_answer(self, distractors):
        return [(d, self.spacy_answer.similarity(d)) for d in distractors]

    def sort_distractors(self, distractors):
        return sorted(distractors, key=lambda x: x[1], reverse=True)

    def filter_subsets(self, sorted_distractors):
        non_subsets = set()
        for sd in sorted_distractors:
            if sd[0].text not in " ".join({ns[0].text for ns in non_subsets}):
                non_subsets.add(sd)
        return non_subsets

    def filter_root_duplicates(self, sorted_distractors):
        non_root_duplicates = set()
        for sd in sorted_distractors:
            if sd[0].root.text not in {nrd[0].root.text for nrd in non_root_duplicates}:
                non_root_duplicates.add(sd)
        return non_root_duplicates

    def filter_unmatching_roots(self, sorted_distractors):
        matching_roots = set()
        for sd in sorted_distractors:
            if sd[0].root.pos_ == self.spacy_answer.root.pos_:
                matching_roots.add(sd)
        return matching_roots

    def filter_non_temporal(self, sorted_distractors):
        if self.spacy_answer[-1].text in ' days months years hours minutes seconds weeks ':
            temporal = set()
            for sd in sorted_distractors:
                if sd[0][-1].text in ' days months years hours minutes seconds weeks ':
                    temporal.add(sd)
            return temporal
        else:
            return sorted_distractors

    def filter_insentence_ners(self, sorted_distractors):
        not_in_sentence = set()
        for sd in sorted_distractors:
            if sd[0].text not in self.spacy_sentence.text:
                not_in_sentence.add(sd)
        return not_in_sentence

    def ner_bubble_up(self, sorted_distractors):
        if self.spacy_answer[-1].ent_type_ != '':
            bubbled_ner = []
            answer_ent_type = self.spacy_answer[-1].ent_type_
            last_ent_index = 0

            for i, sd in enumerate(sorted_distractors):

                if sd[0][-1].ent_type_ == answer_ent_type:

                    if last_ent_index < i or last_ent_index == 0:
                        if last_ent_index == 0:
                            bubbled_ner.insert(last_ent_index, sd)
                            last_ent_index += 1
                        else:
                            bubbled_ner.insert(last_ent_index, sd)
                            last_ent_index += 1
                    else:
                        bubbled_ner.append(sd)

                else:
                    bubbled_ner.append(sd)

            return bubbled_ner
        else:
            return sorted_distractors

    def get_synset(self, spacy_word, sent=None):
        if not sent:
            for sentence in self.spacy.sents:
                if spacy_word[0].i in [w.i for w in sentence]:
                    context_sent = sentence
        else:
            context_sent = sent

        synset = pywsd.similarity.max_similarity(context_sent.text, spacy_word.root.text, pos='n')

        if not synset:
            synsets = nltk.corpus.wordnet.synsets(self.spacy_answer.root.text)
            if len(synsets) > 0:
                synset = synsets[0]
            else:
                return None
        return synset

    def add_similarity_score(self, sorted_distractors):
        if self.spacy_answer.root.pos_ == 'NOUN' and self.spacy_answer[-1].ent_type_ == '' and self.spacy_answer.root.tag_ not in 'NNPS':
            sorted_distractors = list(sorted_distractors)
            answer_synset = self.get_synset(self.spacy_answer, sent=self.spacy_sentence)
            scores = []
            if not answer_synset:
                return sorted_distractors
            else:
                for sd in sorted_distractors:
                    distractor_synset = self.get_synset(sd[0])
                    if not distractor_synset:
                        scores.append(0)
                    else:
                        scores.append(answer_synset.lch_similarity(distractor_synset))

            scores = np.array(scores)
            scores = scores/scores.max()

            plus_sim = []
            for i, sd in enumerate(sorted_distractors):
                plus_sim.append((sd[0], sd[1]+scores[i]))

            return plus_sim

        else:
            return sorted_distractors

    def make_distractors(self):

        distractors = self.filter_duplicates(self.candidate_distractors)
        if len(self.matching_ents) < 4 or self.spacy_answer[-1].ent_type_ == 'GPE':
            distractors = self.remove_common_root_with_answer(distractors) # this is messing everything up!
        distractors = self.remove_distractor_in_answer(distractors)
        distractors = self.remove_answer_in_distractor(distractors)
        distractors = self.get_similarities_to_answer(distractors)
        sorted_distractors = self.sort_distractors(distractors)

        sorted_distractors = sorted_distractors[:50]


        sorted_distractors = self.filter_non_temporal(self.sort_distractors(sorted_distractors))
        sorted_distractors = self.filter_subsets(self.sort_distractors(sorted_distractors))
        if len(self.matching_ents) < 4 or self.spacy_answer[-1].ent_type_ == 'GPE':
            sorted_distractors = self.filter_root_duplicates(self.sort_distractors(sorted_distractors))
        sorted_distractors = self.filter_unmatching_roots(self.sort_distractors(sorted_distractors))
        sorted_distractors = self.filter_insentence_ners(self.sort_distractors(sorted_distractors))
        sorted_distractors = self.add_similarity_score(self.sort_distractors(sorted_distractors))

        #this last filter returns a list which should be sorted...
        sorted_distractors = self.ner_bubble_up(self.sort_distractors(sorted_distractors))


        output = []
        if self.spacy_answer[0].tag_ == 'DT':

            article = self.spacy_answer[0].text

            for sd in sorted_distractors:
                if sd[0][0].tag_ == 'DT' and sd[0][0].text != article:
                    output.append(article+" "+sd[0][1:].text.lower())
                elif sd[0][0].tag_ != 'DT':
                    output.append(article+" "+sd[0].text.lower())
                else:
                    output.append(sd[0].text.lower())

        else:
            for sd in sorted_distractors:
                if sd[0][0].tag_ == 'DT':
                    output.append(sd[0][1:].text.lower())
                else:
                    output.append(sd[0].text.lower())

        return output[:3]


class SemanticSentence(object):
    def __init__(self, sentence, question, answer, nlp, srl=None, srl_predictor=None):
        if srl == None:
            self.ascii_answer = unicodedata.normalize('NFKD', answer).encode('ascii','ignore')
            self.srl = allen_srl.get_srl(sentence, srl_predictor)
            self.answer_srl_label = self.set_answer_srl_label()
        else:
            self.srl = srl

        self.nlp = nlp
        self.raw_sentence = sentence
        self.raw_question = question
        self.raw_answer = answer
        self.spacy_sent = self.nlp(self.raw_sentence)
        self.spacy_ques = self.nlp(self.raw_question)
        self.answer_length = self.set_answer_length()
        self.spacy_answer = self.set_spacy_answer()
        self.answer_pos = self.set_answer_pos()
        self.answer_ner = self.set_answer_ner()
        self.answer_ner_iob  = self.set_answer_ner_iob()
        self.answer_depth = self.set_answer_depth()
        self.answer_word_count = self.set_answer_word_count()
        self.all_pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PUNCT']
        self.all_ner_tags = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        self.all_srl_labels = ['V', 'A0', 'A1', 'A2', 'C-arg', 'R-arg', 'AM-ADV', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-PRP', 'AM-REC', 'AM-TMP']
        
    def set_answer_length(self):
        return len(self.raw_answer.split())
    
    def set_answer_depth(self):
        try:
            return self.raw_question.split().index('_')/len(self.raw_question.split())
        except:
            return self.raw_question.index('_')/len(self.raw_sentence)
        
    def set_answer_pos(self):
        output = [word.tag_ for word in self.spacy_answer][:5]
        return self.take5(output)
        
    def set_spacy_answer(self):
        self.answer_index_start = [i for i, word in enumerate(self.spacy_ques) if word.text == '_'][0]
        self.answer_index_end = (self.answer_index_start + self.answer_length) - 1
        return [word for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]]
        
    def set_answer_word_count(self):
        self.word_count = Counter([word.lemma_ for word in self.spacy_sent])
        output =  [self.word_count[word.lemma_] for word in self.spacy_sent][:5]
        while len(output) < 5:
            output += [0]
        return output
    
    def set_answer_ner(self):
        output = [word.ent_type_ for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]][:5]
        return self.take5(output)
    
    def set_answer_ner_iob(self):
        output = [word.ent_iob_ for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]][:5]
        return self.take5(output)

    def set_answer_srl_label(self):
        if len(self.srl) > 0:
            srl_labels = []
            for rel in self.srl:
                for key in rel:
                    if self.ascii_answer in rel[key]:
                        srl_labels += [key]
            if len(srl_labels) > 0:
                return Counter(srl_labels).most_common(1)[0][0]
            else:
                return 'XX'
        else:
            return 'XX'
    
    def take5(self, output):
        output = output[:5]
        while len(output) < 5:
            output.append('XX')
        return output
    
    def vector(self):
        vector = []
        vector += [self.answer_length, self.answer_depth]
        vector += self.answer_word_count
        for tag in self.answer_pos:
            vector += list(([tag] == np.array(self.all_pos_tags)).astype('int'))
        for tag in self.answer_ner_iob:
            vector += list(([tag] == np.array(['I', 'O', 'B'])).astype('int'))
        for tag in self.answer_ner:
            vector += list(([tag] == np.array(self.all_ner_tags)).astype('int'))
        vector += list(([tag] == np.array(self.all_srl_labels)).astype('int'))
        return np.array(vector)

class Blanker(object):
    def __init__(self, sent_with_srl, nlp):
        self.nlp = nlp
        self.spacy = self.nlp(sent_with_srl['sentence'])
        self.srl = sent_with_srl['srl']
        self.blanks = self.make_blanks()
        
    def make_blanks(self):
        good_tags = [u'CD', u'JJ', u'VB', u'VBG', u'FW']

        #iterate thru some list of patterns
        matcher = Matcher(self.nlp.vocab)

        for i, pattern in enumerate(good_tags):
            matcher.add(f"{i}", None, [{TAG: tag} for tag in pattern.split()])

        matches = matcher(self.spacy) # this has to be a spacy blanked_sentence, so we have to run nlp(sentence) to get this

        noun_ent_matches = []
        for chunk in self.spacy.noun_chunks:
            noun_ent_matches.append((0, chunk.start, chunk.end))
        for ent in self.spacy.ents:
            noun_ent_matches.append((0, ent.start, ent.end))
        for match in matches:
            noun_ent_matches.append(match)

        # now we need to generate the actual question sentences with blanks, this is just for one sentence
        all_blanks = []
        for m in noun_ent_matches:
            spacy_answer = self.spacy[m[1]:m[2]]
            answer = ""
            blanked_sentence = ""
            for i, token in enumerate(self.spacy):
                if i in range(m[1], m[2]):
                    answer += (token.text+token.whitespace_)
                    blanked_sentence += ('_'+token.whitespace_)
                else:
                    blanked_sentence += (token.text+token.whitespace_)

            all_blanks.append({'question': blanked_sentence, 'answer': answer, 'sentence': self.spacy.text, 'spacy_sentence': self.spacy, 'spacy_answer': spacy_answer, 'srl':self.srl})

        return all_blanks