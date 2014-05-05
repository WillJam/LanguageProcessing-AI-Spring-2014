#!/usr/bin/env python
"""
Train and predict using a Hidden Markov Model part-of-speech tagger.

Usage:
  hw5.py training_file test_file
"""

import optparse
import collections
import math
import copy

import hw5_common

# Smoothing methods
NO_SMOOTHING = 'None'  # Return 0 for the probability of unseen events
ADD_ONE_SMOOTHING = 'AddOne'  # Add a count of 1 for every possible event.
# *** Add additional smoothing methods here ***

# Unknown word handling methods
PREDICT_ZERO = 'None'  # Return 0 for the probability of unseen words

# If p is the most common part of speech in the training data,
# Pr(unknown word | p) = 1; Pr(unknown word | <anything else>) = 0
PREDICT_MOST_COMMON_PART_OF_SPEECH = 'MostCommonPos'
# *** Add additional unknown-word-handling methods here ***


class BaselineModel:
  '''A baseline part-of-speech tagger.

  Fields:
    dictionary: map from a word to the most common part-of-speech for that word.
    default: the most common overall part of speech.
  '''
  def __init__(self, training_data):
    '''Train a baseline most-common-part-of-speech classifier.

    Args:
      training_data: a list of pos, word pairs:
    '''

    self.dictionary = dict()
    pos_count = dict()
    dictionary_count = dict()
    for data_index in range(0, len(training_data)):
      #extract some data
      curr_pair = tuple(training_data[data_index])
      curr_pos = str(curr_pair[0])
      curr_word = curr_pair[1]
      curr_pair = tuple((str(curr_pos), str(curr_word)))

      
      if(curr_word != '<s>'):
        if curr_pos not in pos_count:
          pos_count[str(curr_pos)] = 0
        pos_count[str(curr_pos)] += 1

        if curr_pair not in dictionary_count:
          dictionary_count[curr_pair] = 0
        dictionary_count[curr_pair] += 1

        self.dictionary[curr_word] = str(curr_pos)

        #get max
        if (dictionary_count[curr_pair] > dictionary_count[(str(self.dictionary[curr_word]), curr_word)]):
          self.dictionary[curr_word] = dictionary_count[curr_pair]

      
    #get most common
    MAX = 0
    for x, y in pos_count.iteritems():
      if y > MAX:
        MAX = y
        self.default = x
        
    

    
    # FIXME *** IMPLEMENT ME ***

  def predict_sentence(self, sentence):
    return [self.dictionary.get(word, self.default) for word in sentence]


class HiddenMarkovModel:
  def __init__(self, order, emission, transition, parts_of_speech, known):
    # Order 0 -> unigram model, order 1 -> bigram, order 2 -> trigram, etc.
    self.order = order
    # Emission probabilities, a map from (pos, word) to Pr(word|pos)
    self.emission = emission
    # Transition probabilities
    # For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)
    self.transition = transition
    # A set of parts of speech known by the model
    self.parts_of_speech = parts_of_speech
    # A set of words known by the model
    self.known_words = known

  def predict_sentence(self, sentence):
    return self.find_best_path(self.compute_lattice(sentence))

  def compute_lattice(self, sentence):
    """Compute the Viterbi lattice for an example sentence.

    Args:
      sentence: a list of words, not including the <s> tokens on either end.

    Returns:
      FOR ORDER 1 Markov models:
      lattice: [{pos: (score, prev_pos)}]
        That is, lattice[i][pos] = (score, prev_pos) where score is the
        log probability of the most likely pos/word sequence ending in word i
        having part-of-speech pos, and prev_pos is the part-of-speech of word i-1
        in that sequence.

        i=0 is the <s> token before the sentence
        i=1 is the first word of the sentence.
        len(lattice) = len(sentence) + 2.

      FOR ORDER 2 Markov models: ??? (extra credit)
    """
    #fisrt case with sentence tag
    lattice = []
    lattice.append({})
    lattice[0]['<s>'] = (math.log(1), None)

    #middle cases with the actual words
    layernum = 0
    for word in sentence:
      layernum += 1
      lattice.append({})
      for pos in self.parts_of_speech:
        if(pos != '<s>'):
          prev_pos = ""
          MAX = float('-inf')
          for key, value in lattice[layernum - 1].iteritems():
            #test to see if we've seen the word
            emission = float('-inf')
            transition = float('-inf')
            if (pos, word) in self.emission:
              #if we have, assign it the emission value
              emission = math.exp(self.emission[(pos,word)])
            if (key,pos) in self.transition:
              transition = math.exp(self.transition[(key, pos)])
            
            score = emission * transition  * math.exp(value[0])

            if(MAX <= score):
              MAX = score
              prev_pos = key
              
            
            
               
          value = MAX
          if(MAX != float('-inf') and MAX != 0):
            value = math.log(MAX)
          lattice[layernum][pos] = (value, prev_pos)


    #end case with sentence tag
    lattice.append({})
    MAX = float('-inf')
    layernum += 1
    for key, value in lattice[layernum - 1].iteritems():
        transition = float('-inf')

        if (key,'<s>') in self.transition:
          if(key != '<s>'):
           transition = math.exp(self.transition[(key, '<s>')])

        score = math.exp(self.emission[('<s>', '<s>')]) * transition * math.exp(value[0])
        if(MAX <= score):
            MAX = score
            prev_pos = key

    value = MAX
    if(MAX != float('-inf') and MAX != 0):
      value = math.log(MAX)

    lattice[layernum]['<s>'] = (value, prev_pos)
    
      
    return lattice
      
    

  @staticmethod
  def train(training_data,
      smoothing=ADD_ONE_SMOOTHING,
      unknown_handling=PREDICT_MOST_COMMON_PART_OF_SPEECH,
      order=1):
      # You can add additional keyword parameters here if you wish.
    '''Train a hidden-Markov-model part-of-speech tagger.

    Args:
      training_data: A list of pairs of a word and a part-of-speech.
      smoothing: The method to use for smoothing probabilities.
         Must be one of the _SMOOTHING constants above.
      unknown_handling: The method to use for handling unknown words.
         Must be one of the PREDICT_ constants above.
      order: The Markov order; the number of previous parts of speech to
        condition on in the transition probabilities.  A bigram model is order 1.

    Returns:
      A HiddenMarkovModel instance.
    '''


    parts_of_speech = set()
    known_words = set()
    transition = dict()
    emission = dict()

    emission_count_dict = dict()

    emission_pos_count_dict = dict();

    transition_count_dict = dict()
    
    transition_prev_pos_count_dict = dict()

    
    for data_index in range(0, len(training_data)):
      #extract some data
      curr_pair = tuple(training_data[data_index])
      curr_pos = curr_pair[0]
      curr_word = curr_pair[1]
      
      parts_of_speech.add(curr_pos)
      known_words.add(curr_word)

      #add the pos to the count
      if curr_pos in emission_pos_count_dict:
        emission_pos_count_dict[curr_pos] += 1.0
      else:
        emission_pos_count_dict[curr_pos] = 1.0

      #add emission data
      if curr_pair in emission_count_dict:
        emission_count_dict[curr_pair] += 1.0
      else:
        emission_count_dict[curr_pair] = 1.0

      #extract some more data
      if(data_index != 0):
        prev_pair = training_data[data_index - 1]
        prev_pos = prev_pair[0]

        transition_pair = (prev_pos, curr_pos)

        #add the previous pos to transition count
        if prev_pos in transition_prev_pos_count_dict:
          transition_prev_pos_count_dict[prev_pos] += 1.0
        else:
          transition_prev_pos_count_dict[prev_pos] = 1.0

        #add transition stuff
        if transition_pair in transition_count_dict:
            transition_count_dict[transition_pair] += 1.0
        else:
            transition_count_dict[transition_pair] = 1.0


    # do smoothing
    if smoothing is ADD_ONE_SMOOTHING:
      #add one for transition for every part of speech
      for prev in parts_of_speech:
        for curr in parts_of_speech:
          transition_prev_pos_count_dict[prev] += 1.0

          if (prev, curr) not in transition_count_dict:
            transition_count_dict[(prev, curr)]= 0
          transition_count_dict[(prev, curr)] += 1.0
          
      #add one for emission for every part of speech
      for word in known_words:
        for pos in parts_of_speech:
          emission_pos_count_dict[pos] += 1.0
          if (pos, word) not in emission_count_dict:
            emission_count_dict[(pos, word)]= 0
          emission_count_dict[(pos, word)] += 1.0

    """
    for x,y in emission_count_dict.iteritems():
      print x, y

    for x,y in transition_count_dict.iteritems():
      print x, y
    """
      
    #calculate new emission probability
    for emission_pair, count in emission_count_dict.iteritems():
      #print str(emission_count_dict[curr_pair]) + " / " + str(emission_pos_count_dict[curr_pos])
      emission[emission_pair] = math.log(count / emission_pos_count_dict[emission_pair[0]])

      
    #calculate new transition probability
    for transition_pair, count in transition_count_dict.iteritems():
       #print str(transition_pair) + " " + str(transition_count_dict[transition_pair]) + " / " + str(transition_prev_pos_count_dict[prev_pos])
       transition[transition_pair] = math.log(count/transition_prev_pos_count_dict[transition_pair[0]])

    return HiddenMarkovModel(order,emission,transition,parts_of_speech,known_words)
    

  @staticmethod
  def find_best_path(lattice):
    """Return the best path backwards through a complete Viterbi lattice.

    Args:
      FOR ORDER 1 MARKOV MODELS (bigram):
        lattice: [{pos: (score, prev_pos)}].  See compute_lattice for details.

    Returns:
      FOR ORDER 1 MARKOV MODELS (bigram):
        A list of parts of speech.  Does not include the <s> tokens surrounding
        the sentence, so the length of the return value is 2 less than the length
        of the lattice.
    """
    # FIXME *** IMPLEMENT ME ***

    best_path = []
    start = lattice[len(lattice) - 1]['<s>'][1]
    best_path.append(start)
    count = 0
    for num in range(len(lattice) - 2, 1, -1):
      best_path.insert(0,lattice[num][best_path[count]][1])
      count += 1
      
    return best_path
    
def main():
  parser = optparse.OptionParser()
  parser.add_option('-s', '--smoothing', choices=(NO_SMOOTHING,
    ADD_ONE_SMOOTHING), default=NO_SMOOTHING)
  parser.add_option('-o', '--order', default=1, type=int)
  parser.add_option('-u', '--unknown',
      choices=(PREDICT_ZERO, PREDICT_MOST_COMMON_PART_OF_SPEECH,),
      default=PREDICT_ZERO)
  options, args = parser.parse_args()
  train_filename, test_filename = args
  training_data = hw5_common.read_part_of_speech_file(train_filename)
  if options.order == 0:
    model = BaselineModel(training_data)
  else:
    model = HiddenMarkovModel.train(
        training_data, options.smoothing, options.unknown, options.order)
  predictions = hw5_common.get_predictions(
      test_filename, model.predict_sentence)
  for word, prediction, true_pos in predictions:
    print word, prediction, true_pos

if __name__ == '__main__':
  main()
