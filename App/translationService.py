import pandas as pd
import io
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit 
import os
import flask
import json
import time
import datetime
from textblob import TextBlob as tb
f = open('./iHumanQuestions_wOptions_20210419.csv',"rb")
df2 = pd.read_csv(f)
possible_responses = df2["Statement.RESPONSE"].values.tolist()
paired_questions = df2["Statement.QUESTION"].values.tolist()

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

text = ""

from sentence_transformers import SentenceTransformer
lsbert_model = SentenceTransformer('bert-large-nli-mean-tokens')

import pyaudio
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from six.moves import queue


class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,

            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,

            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True

        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:

            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]


            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


rate = 44100
chunk = int(rate/10)


access_token = "02dE4JynMyMV6NtH2NRPiyaURlQiAfpCS3o_TxhSbT_X9JicRl5EaGeZbFBjOTfJxeLz6CFY53_eeT5wJbAB0Ct9h6awA"


example_mc = MediaConfig('audio/x-raw', 'interleaved', 44100, 'S16LE', 1)

streamclient = RevAiStreamingClient(access_token, example_mc)
import json

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



import io
import pickle
with open('lsEmbedded_pairedQuestions_20210419.csv', 'rb') as filehandle:
  ls_embedded_paired_questions = pickle.load(filehandle)

body_parts_list = ["stomach", "belly", "abdominals", "abs", "heart", "head", "shoulders", "knees", "toes", "eye", "ear", "mouth", "nose", "chest", "leg", "lung", "lungs", "kidney", "kidneys"]

def get_bodyPart(text):
 
  import nltk
  tkn_text = nltk.word_tokenize(text)

  for word in tkn_text:
    if word in body_parts_list:
      global bodyPart
      bodyPart = word

def get_response(text):

  add_bodypart = False
  if not any(part in text for part in body_parts_list):
    if 'pain' in text:
      if bodyPart != 'none':  
        text = text + ' ' + bodyPart
        add_bodypart = True  

  text_vec = lsbert_model.encode([text])[0]

  similarities = []
  for i in range(len(possible_responses)):
    similarities.append(cosine(text_vec, ls_embedded_paired_questions[i]))
  
  max_sim_index = max((v, i) for i, v in enumerate(similarities))[1]
   
  response = possible_responses[max_sim_index]

  paired_q = paired_questions[max_sim_index]
  similarity = similarities[max_sim_index]

  if add_bodypart == True:
    global add_bodypart_count
    if add_bodypart_count == 0:
      response = "Did you mean the " + bodyPart + " pain? " + response
      add_bodypart_count += 1
    elif add_bodypart_count == 1:
      response = "The " + bodyPart + " pain? " + response 
      add_bodypart_count += 1
    elif add_bodypart_count == 2:
      response = "Oh, you mean the " + bodyPart + " pain? " + response
      add_bodypart_count == 0

  dict_results = {}
  dict_results['response'] = response
  dict_results['paired_q'] = paired_q
  dict_results['similarity'] = similarity

  return(dict_results)

def seperate_question(text):

  nonQuestionPunctuation = ['.', ';', '!']
  qPunct_i = []
  nonQpunct_i = []
  for i in range(len(text)):
    if text[i] in nonQuestionPunctuation:
      nonQpunct_i.append(i)
    elif text[i] == '?':
      qPunct_i.append(i)

 
  if len(qPunct_i) == 0 or len(nonQpunct_i) == 0:    
    return(text)


  elif max(nonQpunct_i) < min(qPunct_i):    
    return(text[max(nonQpunct_i)+2:])


  elif max(qPunct_i) < min(nonQpunct_i):
    return(text[:max(qPunct_i)+1])


  else:
    return(text)

def check_userName(df):


  user_intro = df['User'][0].lower()
  name_first = user_name_first.lower()
  name_last = user_name_last.lower()
  name_full = name_first + ' ' + name_last

  if name_full in user_intro:
    return("full name")
  elif name_first in user_intro:
    return("first name")
  elif name_last in user_intro:
    return("last name")
  else:
    return('none')

def check_userJobFunction(df):

  user_intro = df['User'][0].lower()
  jobFunction = user_jobFunction.lower()

  if jobFunction in user_intro:
    return 1
  else:
    return 0

def check_userTitle(df):

  user_intro = df['User'][0].lower()
  
  if user_title.lower() in user_intro:
    return 1
  else:
    return 0

def get_greeting(df):

  greetings_list_formal = ("hello", "good evening", "good afternoon", "good morning", "nice to meet you", "a pleasure to meet you", "good to see you", "greetings", "nice to meet you", 
                "pleased to meet you", "good to meet you")
  
  greetings_list_informal = ("hiya", "howdy", "how's it going", "hi", "morning", "evening", "yo", "what's up", "hey there", "hey", "sup")
  
  user_intro = df['User'][0].lower()

  if any(greeting in user_intro for greeting in greetings_list_formal):
    return "formal"
  elif any(greeting in user_intro for greeting in greetings_list_informal):
    return "informal"
  else:
    return "none"

def check_patient_name(df):

  title_fullName = patient_title + ' ' + patient_name_full
  title_firstName = patient_title + ' ' + patient_name_first
  title_lastName = patient_title + ' ' + patient_name_last

  user_inputs = df['User']
  patient_addresses = []

  for i in range(len(user_inputs)):
    
    if title_fullName in user_inputs[i]:
      patient_addresses.appended("full name and title")
    elif title_firstName in user_inputs[i]:
      patient_addresses.append("first name and title")
    elif title_lastName in user_inputs[i]:
      patient_addresses.append("last name and title")
    elif patient_name_full in user_inputs[i]:
      patient_addresses.append("full name and no title")
    elif patient_name_first in user_inputs[i]:
      patient_addresses.append("first name and no title")
    elif patient_name_last in user_inputs[i]:
      patient_addresses.append("last name and no title")
    else:
      patient_addresses.append("none")

  return patient_addresses

def get_repeatedWords(df):

  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize

  counts_list = []
  repeatedWords_list = []

  for i in range(len(df['User'])):
    repeatedWords = []
    question = df['User'][i].lower()
    
    if i < 2:
      patient_words = nltk.word_tokenize(' '.join(df['Patient'][0:i]))
      patient_words = [word.lower() for word in patient_words if word.isalnum()]
      patient_words = [word for word in patient_words if not word in stopwords.words()]
      users_words = ' '.join(df['User'][0:i])

      for word in patient_words:
        if word not in users_words.lower():   
          if word in question:
            repeatedWords.append(word)

      repeatedWords_count = len(set(repeatedWords))
      counts_list.append(repeatedWords_count)
      repeatedWords_list.append(list(set(repeatedWords)))

    else:
      patient_words = nltk.word_tokenize(' '.join(df['Patient'][i-2:i]))
      patient_words = [word.lower() for word in patient_words if word.isalnum()]
      patient_words = [word for word in patient_words if not word in stopwords.words()]
      users_words = ' '.join(df['User'][i-2:i])

      for word in patient_words:
        if word not in users_words.lower():
          if word in question:
            repeatedWords.append(word)

      repeatedWords_count = len(set(repeatedWords))
      counts_list.append(repeatedWords_count)
      repeatedWords_list.append(list(set(repeatedWords)))

  df['RepeatedWord_words'] = repeatedWords_list
  df['RepeatedWord_counts'] = counts_list

def get_polarity(text):

    polarity_list = []
    user_inputs = df['User']
    for i in range(len(user_inputs)):
      polarity = tb(user_inputs[i]).sentiment.polarity
      polarity_list.append(polarity)
    return polarity_list

def get_subjectivity(text):

    subjectivity_list = []
    user_inputs = df['User']
    for i in range(len(user_inputs)):
      subjectivity = tb(user_inputs[i]).sentiment.subjectivity
      subjectivity_list.append(subjectivity)
    return subjectivity_list



def chat(socketio):


  text = ''
  user_dialogue = []
  patient_dialogue = []
  paired_qs = []
  similarity_scores = []

  unclear_questions = ["I'm not sure what you mean.", "Can you rephrase that?", "I don't know what that means."]
  unclear_index = 0       

  global add_bodypart_count
  add_bodypart_count = 0             

  global patient_name_full
  patient_name_full = "Tom Bradford"
  global patient_name_first
  patient_name_first = "Tom"
  global patient_name_last
  patient_name_last = "Bradford"
  global patient_title
  patient_title = "Mr."

  global bodyPart
  bodyPart = 'none'

  global user_name_first
  user_name_first = "Rishav"
  global user_name_last
  user_name_last = "Kumar"
  global user_title
  user_title = "Dr."
  global user_jobFunction
  user_jobFunction = "Nurse"
  
  print("\nThank you, the patient is ready for you. \n\nPatient Information \nName: Tom Bradford \nAge: 71\n\n(Ask a question, or type 'exit' to quit.)\n")

  with MicrophoneStream(rate, chunk) as stream:
  
    try:
      
        response_gen = streamclient.start(stream.generator())

        for response in response_gen:
            y = json.loads(response)
            
            text = ""
            if y["type"] == "final":
                for i in y["elements"]:
                    text+=i["value"]
                
            if  text=="":
              continue
            print("My speech to text response:",text)
            
            if(text=="Exit."):
              break
            user_dialogue.append(text)
            text = seperate_question(text)
            get_bodyPart(text)
            response_values = get_response(text)

            response = response_values['response']
            paired_q = response_values['paired_q']
            similarity = response_values['similarity']

            paired_qs.append(paired_q)
            similarity_scores.append(similarity)
            get_bodyPart(response)

            if similarity < 0.80:
              response = unclear_questions[unclear_index]
              unclear_index += 1
              if unclear_index > 2:
                unclear_index = 0

            patient_dialogue.append(response)
            
            print("Patient:", response)
            
            socketio.emit('data', {
                        'content': text,
                        'type': response,
                    })
            time.sleep(1)
            
            
    except KeyboardInterrupt:
        streamclient.client.send("EOS")
        pass

  
  print("Thank you!")
  scoring = list()

  global df
  df = pd.DataFrame()
  df['User'] = user_dialogue
  df['Patient'] = patient_dialogue
  
  df['Paired_Question'] = paired_qs 
  df['Similarity_Score'] = similarity_scores
  df['Polarity'] = get_polarity(df)
  df['Subjectivity'] = get_subjectivity(df)
  get_repeatedWords(df)
  df['Patient_Named'] = check_patient_name(df)


  global intro_scores
  intro_scores = {}
  intro_scores['userName_Provided'] = check_userName(df)
  intro_scores['userJobFunction_Provided'] = check_userJobFunction(df)
  intro_scores['userTitle_Provided'] = check_userTitle(df)
  intro_scores['Greeting_Provided'] = get_greeting(df)

  Greeting_Provided = intro_scores['Greeting_Provided']
  userJobFunction_Provided = intro_scores['userJobFunction_Provided']
  userName_Provided = intro_scores['userName_Provided']
  userTitle_Provided = intro_scores['userTitle_Provided']

  Intro_scores_list = [Greeting_Provided, userJobFunction_Provided, userName_Provided, userTitle_Provided]
  
  if len(df) > 3:
    while len(Intro_scores_list) < len(df):
      Intro_scores_list.append(0)
    df['Intro_scores'] = Intro_scores_list
  
  df.to_csv("output.csv")



