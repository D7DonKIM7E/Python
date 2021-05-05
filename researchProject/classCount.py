import os
import re
import json
import pprint

# gender
man = 0
woman = 0
unknown_gender = 0

# emotion
angry_emotion = 0
sad_emotion = 0
happy_emotion = 0
surprise_emotion = 0
neutral_emotion = 0
unknown_emotion = 0

# gaze
middle = 0
left = 0
right = 0
down = 0
up = 0
back = 0

# tone
angry_tone = 0
sad_tone = 0
happy_tone = 0
surprise_tone = 0
neutral_tone = 0

# intent
ask_others = 0
ask_location = 0
question_others = 0
question_location = 0
statement = 0
greeting = 0

# multimodal
yes = 0
no = 0

count = 0
file_lst = []

fold_path = '/data/json_done/100_upload_json/'
file_lst = os.listdir(fold_path)
#print(file_lst)

# gender_lst = []
# gaze_lst = []
# emotion_lst = []
# tone_lst = []
# intent_lst = []
# multi_lst = []

neutral_sad = 0
neutral_happy = 0
neutral_surprise = 0
neutral_angry = 0
sad_neutral = 0
happy_neutral = 0
surprise_neutral = 0
angry_neutral = 0

deficit = 0
deficit_lst = []

for fi in file_lst:
    js = fi.endswith('.json')
    if js == True:
        with open(os.path.join(fold_path, fi), 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        if len(json_data) != 7:
            print(fi)
            deficit += 1
            deficit_lst.append(fi)

        for i, c in enumerate(json_data):
            tmp_emotion = ''
            tmp_tone = ''
            if c['from_name'] == 'bbox':
                gender = c['value']['rectanglelabels'][0]
                # gender_lst.append(gender)
                # gender_lst = list(set(gender_lst))
                if gender == 'man':
                    man += 1
                elif gender == 'woman':
                    woman += 1
                elif gender == 'unknown':
                    unknown_gender += 1
            if c['from_name'] == 'gaze':
                gaze = c['value']['choices'][0]
                # gaze_lst.append(gaze)
                # gaze_lst = list(set(gaze_lst))
                if gaze == 'middle':
                    middle += 1
                elif gaze == 'left':
                    left += 1
                elif gaze == 'right':
                    right += 1
                elif gaze == 'down':
                    down += 1
                elif gaze == 'up':
                    up += 1
                elif gaze == 'back':
                    back += 1
            if c['from_name'] == 'emotion':
                emotion = c['value']['choices'][0]
                # emotion_lst.append(emotion)
                # emotion_lst = list(set(emotion_lst))
                if emotion == 'neutral':
                    neutral_emotion += 1
                elif emotion == 'angry':
                    angry_emotion += 1
                elif emotion == 'sad':
                    sad_emotion += 1
                elif emotion == 'happy':
                    happy_emotion += 1
                elif emotion == 'surprise':
                    surprise_emotion += 1
                elif emotion == 'unknown':
                    unknown_emotion += 1
                tmp_emotion = emotion
            if c['from_name'] == 'tone':
                tone = c['value']['choices'][0]
                # tone_lst.append(tone)
                # tone_lst = list(set(tone_lst))
                if tone == 'neutral':
                    neutral_tone += 1
                elif tone == 'angry':
                    angry_tone += 1
                elif tone == 'sad':
                    sad_tone += 1
                elif tone == 'happy':
                    happy_tone += 1
                elif tone == 'surprise':
                    surprise_tone += 1
                tmp_tone = tone
            
            if tmp_emotion == 'neutral' and tmp_tone == 'sad':
                neutral_sad += 1
            if tmp_emotion == 'neutral' and tmp_tone == 'happy':
                neutral_happy += 1
            if tmp_emotion == 'neutral' and tmp_tone == 'surprise':
                neutral_surprise += 1
            if tmp_emotion == 'neutral' and tmp_tone == 'angry':
                neutral_angry += 1

            if tmp_tone == 'neutral' and tmp_emotion == 'sad':
                sad_neutral += 1
            if tmp_tone == 'neutral' and tmp_emotion == 'happy':
                happy_neutral += 1
            if tmp_tone == 'neutral' and tmp_emotion == 'surprise':
                surprise_neutral += 1
            if tmp_tone == 'neutral' and tmp_emotion == 'angry':
                angry_neutral += 1

            if c['from_name'] == 'intent':
                intent = c['value']['choices'][0]
                # intent_lst.append(intent)
                # intent_lst = list(set(intent_lst))
                if intent == 'ask-others':
                    ask_others += 1
                elif intent == 'ask-location':
                    ask_location += 1
                elif intent == 'question-others':
                    question_others += 1
                elif intent == 'question-location':
                    question_location += 1
                elif intent == 'greeting':
                    greeting += 1
                elif intent == 'statement':
                    statement += 1
            if c['from_name'] == 'multimodal':
                multimodal = c['value']['choices'][0]
                # multi_lst.append(multimodal)
                # multi_lst = list(set(multi_lst))
                if multimodal == 'yes':
                    yes += 1
                elif multimodal == 'no':
                    no += 1
    count += 1
    print(count)
print('deficit json file: ', deficit)
print('deficit_lst : ', deficit_lst)

'''
# print('gender : ', gender_lst)
# print('gaze : ', gaze_lst)
# print('emotion : ', emotion_lst)
# print('tone : ', tone_lst)
# print('intent : ', intent_lst)
# print('multimodal : ', multi_lst)
'''

# gender
g_total = man+woman+unknown_gender
print('### gender ###')
print('man : ',man, round(float(man/g_total), 2))
print('woman : ',woman, round(float(woman/g_total), 2))
print('unknown : ', unknown_gender, round(float(unknown_gender/g_total), 2))
print('total : ', man+woman+unknown_gender)

# gaze
ga_total = middle+left+right+down+up+back
print('### gaze ###')
print('middle : ', middle, round(float(middle/ga_total), 2))
print('left : ', left, round(float(left/ga_total), 2))
print('right : ', right, round(float(right/ga_total), 2))
print('down : ', down, round(float(down/ga_total), 2))
print('up : ', up, round(float(up/ga_total), 2))
print('back : ', back, round(float(back/ga_total), 2))
print('total : ', middle+left+right+down+up+back)

# emotion
e_total = angry_emotion+sad_emotion+happy_emotion+surprise_emotion+neutral_emotion+unknown_emotion

print('### emotion ###')
print('angry : ', angry_emotion, round(float(angry_emotion/e_total), 2))
print('sad : ', sad_emotion, round(float(sad_emotion/e_total), 2))
print('happy : ', happy_emotion, round(float(happy_emotion/e_total), 2))
print('surprise : ', surprise_emotion, round(float(surprise_emotion/e_total), 2))
print('neutral : ', neutral_emotion, round(float(neutral_emotion/e_total), 2))
print('unknown : ', unknown_emotion, round(float(unknown_emotion/e_total), 2))
print('total : ', angry_emotion+sad_emotion+happy_emotion+surprise_emotion+neutral_emotion+unknown_emotion)

# tone
t_total = angry_tone + sad_tone + happy_tone + surprise_tone + neutral_tone
print('### tone ###')
print('angry : ', angry_tone, round(float(angry_tone/t_total), 2))
print('sad : ', sad_tone, round(float(sad_emotion/t_total), 2))
print('happy : ', happy_tone, round(float(happy_emotion/t_total), 2))
print('surprise : ', surprise_tone, round(float(surprise_emotion/t_total), 2))
print('neutral : ', neutral_tone, round(float(neutral_emotion/t_total), 2))
print('total : ', angry_tone + sad_tone + happy_tone + surprise_tone + neutral_tone)

# intent
i_total = ask_others + ask_location + question_others + question_location + statement + greeting
print('### intent ###')
print('ask-others : ', ask_others, round(float(ask_others/i_total), 2))
print('ask-location : ', ask_location, round(float(ask_location/i_total), 2))
print('question-others : ', question_others, round(float(question_others/i_total), 2))
print('question-location : ', question_location, round(float(question_location/i_total), 2))
print('statement : ', statement, round(float(statement/i_total), 2))
print('greeting : ', greeting, round(float(greeting/i_total), 2))
print('total : ', ask_others + ask_location + question_others + question_location + statement + greeting)

# multimodal
m_total = yes+no
print('### multimodal ###')
print('yes : ', yes, round(float(yes/m_total), 2))
print('no : ', no, round(float(no/m_total), 2))
print('total : ', yes + no)

# emotion-tone
print('### emotion(neutral)-tone ###')
print('neutral_sad : ', neutral_sad)
print('neutral_happy : ', neutral_happy)
print('neutral_surprise : ', neutral_surprise)
print('neutral_angry : ', neutral_angry)

# tone-emotion
print('### tone(neutral)-emotion ###')
print('sad_neutral : ', sad_neutral)
print('happy_neutral : ', happy_neutral)
print('surprise_neutral : ', surprise_neutral)
print('angry_neutral : ', angry_neutral)

