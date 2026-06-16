import nltk # to process text data
import numpy as np # to represent corpus as arrays
import random 
import string # to process standard python strings
from sklearn.metrics.pairwise import cosine_similarity # We will use this later to decide how similar two sentences are
from sklearn.feature_extraction.text import TfidfVectorizer

filepath='./others.txt'
corpus=open(filepath,'r',errors = 'ignore')
raw_data=corpus.read()
#print (raw_data)

raw_data = raw_data.lower()
# print(raw_data)

sent_tokens = nltk.sent_tokenize(raw_data)
# print(sent_tokens)

# print("printing lists in new line") 
  
# print(*sent_tokens, sep = "\n")

word_tokens = nltk.word_tokenize(raw_data)
# print(word_tokens)

# print(len(sent_tokens), len(word_tokens))

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) #see previous section 1.2.5 lemmatization


test_sentence='Today was a wonderful day. The sun was shining so brightly and the birds were chirping loudly!'
# your code here
test_word_tokens = nltk.word_tokenize(test_sentence)# converts documents to list of words

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# print(LemTokens(test_word_tokens))

GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split(): # Looks at each word in your sentence
        if word.lower() in GREETING_INPUTS: # checks if the word matches a GREETING_INPUT
            return random.choice(GREETING_RESPONSES) # replies with a GREETING_RESPONSE
        
print(greeting("hello"))


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response) #add user response to sent_tokens
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens) #get tfidf value
    vals = cosine_similarity(tfidf[-1], tfidf) #get cosine similarity value
    idx=vals.argsort()[0][-2] 
    flat = vals.flatten() 
    flat.sort() #sort in ascending order
    req_tfidf = flat[-2] 
    
    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        print("ROBO:",robo_response)
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        print("ROBO:",robo_response)
        return robo_response
    
if __name__ == "__main__":
    a = input("Enter your question")
    response(a)
