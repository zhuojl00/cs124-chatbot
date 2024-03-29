# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import collections
import numpy as np
import string
#import pandas as pd
#from nrclex import NRCLex
from porter_stemmer import PorterStemmer
#new version 5/25
# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=True):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'
        self.creative = creative
        self.stemmer = PorterStemmer()
        self.data_points = 0
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.movieTitles = util.load_titles('data/movies.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.user_ratings = np.zeros(ratings.shape[0])


        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################
        
        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def validate_emotions(self, emotion):
        if emotion == "no emotion":
            return
        negatives = ['surprise', 'disgust', 'fear', 'anger', 'negative', 'sadness']
        positives = ["anticip", "trust", "joy"]
        if (emotion in negatives):
            print("Oh! I am sorry that I have caused you ", emotion, ". Please continue.")
        if (emotion in positives):
            print("Great! I am so glad I was able to make you ", emotion, ". Please continue.")


    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    
    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        updatedLine = self.preprocess(line)

        if(self.creative):
            self.validate_emotions(self.find_emotion(line))
        all_candidates = []
        for i in range(len(self.movieTitles)):
            all_candidates.append(i)
        #print("arguments are ", updatedLine)
        exact = self.disambiguate(updatedLine, all_candidates)
        #print("exact is", exact)
        if len(exact) == 1:
            movie_index = exact[0]
            print("Great, you liked", self.movieTitles[movie_index][0])
        else:
            #pattern = re.findall('"([^"]*)"', updatedLine)
            pattern = self.extract_titles(updatedLine)
            if not self.creative and len(pattern) > 1:
                print("Sorry, I can only handle on title at a time! Can you only tell me about one movie?")
                return
            if len(pattern) > 0:
                candidate_index = self.find_movies_by_title(pattern[0])
                candidate_title = []
                for c in candidate_index:
                    candidate_title.append(self.movieTitles[c][0])

                if len(candidate_index) == 1:
                    if self.extract_sentiment(updatedLine) == 0:
                        print("I'm sorry, I'm not sure if you liked " + pattern[0] + ". Can you tell me more about the movie?")
                    elif self.extract_sentiment(updatedLine) == 1:
                        index = candidate_index[0]
                        self.user_ratings[index] = self.extract_sentiment(updatedLine)
                        print("So you liked " + pattern[0] + "? Tell me what you thought about another movie!")
                        self.data_points += 1
                    elif self.extract_sentiment(updatedLine) == -1:
                        index = candidate_index[0]
                        self.user_ratings[index] = self.extract_sentiment(updatedLine)
                        print("So you didn't like " + pattern[0] + "? Tell me what you thought about another movie!")
                        self.data_points += 1
                elif len(candidate_index) == 0:
                    print("I've never heard of " + pattern[0] +",sorry. Tell me about another movie you like?")
                else:
                    print("Which one did you mean?", ', '.join(candidate_title))
            if self.data_points == 5:
                recs = self.recommend(self.user_ratings, self.ratings)
                if len(recs) >= 1:
                    print("Given what you told me, I think you would like " + self.titles[recs[0]][0])
                    self.data_points = 0
            else:
                if self.creative:
                        updatedLine = line.replace('?', ' ?')
                        tokens = updatedLine.split(" ")
                        question = False
                        if tokens[len(tokens) - 1] == "?":
                            question = True
                        words = re.findall('([\w(?:\'\w)?]+)',updatedLine)
                        if (words[len(words)-1] == "?"):
                            words.pop()
                        greetings = ["hi", "sup", "hello", "yo", "hey"]
                        if words[0].lower() in greetings:
                            print("Hello!")
                        elif len(words) > 2 and question:
                            response = ""
                            start = 2
                            if '\'' in words[0]:
                                start = 1
                            for i in range(start,len(words)):
                                if words[i].lower() == "me":
                                    response += "you" + " "
                                elif words[i].lower() == "my":
                                    response += "your" + " "
                                elif words[i].lower() == "mine":
                                    response += "yours" + " "
                                elif words[i].lower() == "yours":
                                    response += "mine" + " "
                                elif words[i].lower() == "your":
                                    response += "my" + " "
                                elif words[i].lower() == "you":
                                    response += "me" + " "
                                else:
                                    response += words[i] + " "
                            if words[0].lower() == "can" and words[1].lower() == 'you':
                                print ("No, I cannot " + response.strip() + ".")
                            if words[0].lower() == "what's":
                                print ("I do not know what " + response.strip() + " is.")
                            if words[0].lower() == "do" and words[1].lower() == 'you':
                                print ("I do not " + response.strip() + ".")
                        elif words[0].lower() == "i'm":
                            response = ""
                            for i in range(1,len(words)):
                                response += words[i] + " "
                            print(response)
                            print("Wow, you really are "+ response.strip() + ".")
                        else:
                            print("Sorry, I don't understand. Tell me about a movie that you have seen.")        

        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

        #print(self.find_emotion("you make me so charitable!!!"))

        '''TESTING FINE GRAINED SENITMENT EXTRACTION!!!'''
        if self.creative:
            sent = "I loved \"Zootopia\" "
            #print("extracting sentiment, expecting 2", self.extract_sentiment(sent))
            sent = "\"Zootopia\" was terrible."
            
            #print("extracting sentiment, expecting -2", self.extract_sentiment(sent))
            sent = "I really reeally liked \"Zootopia\"!!!"
            
            
            #print("extracting sentiment, expecting 2", self.extract_sentiment(sent))

        '''TESTING EDIT DISTANCE!!!!!!!!!'''
        if (self.creative):
            mispell = "the notbook"
            #print("finding possible matches, expecting [5448] got:", self.find_movies_closest_to_title(mispell))
            mispell = "Blargdeblargh"
            #print("finding possible matches, expecting [] got:", self.find_movies_closest_to_title(mispell))

            mispell = "BAT-MAAAN"
            #print("finding possible matches, expecting [524, 5743] got:", self.find_movies_closest_to_title(mispell))

            mispell = "Te"
            #print("finding possible matches, expecting [8082, 4511, 1664] got:", self.find_movies_closest_to_title(mispell))

            mispell  = "Sleeping Beaty"
            #print("finding possible matches, expecting [1656] got:", self.find_movies_closest_to_title(mispell))


        


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        
        return text
    # helper function that checks if title of movie is a substring return -1 if movie is not substring
    # return the index when the first word matches    
    def isSubstring(self, splited_input, splited_movie):
        M = len(splited_movie)
        N = len(splited_input) 
 
        for i in range(N):
            if (splited_input[i] == splited_movie[0]):
                if (splited_input[i:i+M] == splited_movie):
                    return i
        return -1



    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        if self.creative:
            if (len(re.findall('"([^"]*)"', preprocessed_input)) == 0):
                potential_titles = {} # {startIndex: titles}
                input_lower = preprocessed_input.lower().strip(string.punctuation)
                splited_input = re.split(r' ', input_lower)
                
                for movie in self.movieTitles:
                    movietitle = movie[0].lower()
                    articles = ["a", "an", "the"]
                    containsYear = re.findall('\(\d{4}\)', movietitle)
                    if len(containsYear) != 0:
                        movietitle = movietitle[:-7]   
                    for article in articles:
                        size = len(article)
                        if (movietitle[-size-2:] == ', ' + article):
                                movietitle = article + " " + movietitle[:-size-2]     
                    splited_movie = re.split(r' ', movietitle)
                    startIndex = self.isSubstring(splited_input, splited_movie)
                    if startIndex >= 0:
                        if startIndex in potential_titles:
                            old_title = potential_titles[startIndex]
                            if len(old_title) < len(movietitle):
                                potential_titles[startIndex] = movietitle
                        else:
                            potential_titles[startIndex] = movietitle
                # print(list(potential_titles.values()))
                return list(potential_titles.values())
            else:
                return re.findall('"([^"]*)"', preprocessed_input)
        else: 
            return re.findall('"([^"]*)"', preprocessed_input)
    
    def edit_distance (self, string1, string2):
        matrix = np.zeros([len(string1) + 1, len(string2) + 1])
        #matrix[0][0] = 0
        for i in range(len(string2)+1):
            matrix[len(string1)][i] = i 

        for j in range(len(string1)+1):
            matrix[j][0] = len(string1) - j 
        #print("start matrix", matrix)
        for i in range(1, len(string2) + 1):
            for j in range(1, len(string1) + 1):
                sub = matrix[len(string1) - j + 1][i - 1]
                if(string1[j - 1] != string2[i - 1]):
                    sub += 2
                #print(matrix[len(string1) - j + 1][i])

                e1 = matrix[len(string1) - j + 1][i] + 1
                e2 = matrix[len(string1) - j][i - 1]+ 1
                
                poss = np.array([sub, e1, e2])
                matrix[len(string1) - j][i] =  poss.min()
        
        return matrix[0][len(string2)]
        '''if(string1 == string2):
            return 0
        if (len(string1) == 0):
            return len(string2)
        if (len(string2) == 0):
            return len(string1)
        sub = 2 + self.edit_distance(string1[1:], string2[1:])
        e1 = 1 + self.edit_distance(string1[1:], string2)
        e2 = 1 + self.edit_distance(string1, string2[1:])
        '''
        

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        if not self.creative:
            articles = ["a", "an", "the"]
            titles = []
            title = title.lower()
            realTitle = title
            containsYear = re.findall('\(\d{4}\)', title)
            ##Titanic  OR titanic, the (1997) the
            for article in articles:
                size = len(article)
                if (title[0:size] == article):
                    if(len(containsYear) == 0):
                        realTitle = title[size+1:].strip() + ", " + article
                    else:
                        realTitle = title[size+1:-6].strip() + ", " + article + " " + title[-6:]

            for i in range(len(self.movieTitles)):
                movie = self.movieTitles[i]
                #if movie[0][:len(realTitle)].lower() == realTitle:
                if (len(containsYear) != 0 and movie[0].lower() == realTitle):
                    titles.append(i)
                if (len(containsYear) == 0 and movie[0][:-7].lower() == realTitle):
                    titles.append(i)
            return titles

        else:
            titles = []
            title = title.lower()
            check_tokens = title.split()
            # print("input tokens", check_tokens)
            for i in range(len(self.movieTitles)):
                curr_title = self.movieTitles[i][0].lower()
                curr_title = re.sub(r'[^\w\s]', '', curr_title)
                curr_tokens = curr_title.split()
                if set(check_tokens).issubset(set(curr_tokens)):
                    titles.append(i)
            # print(titles)
            return titles

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        preprocessed_input = re.sub('"([^"]*)"', ' ', preprocessed_input)
        stemmed = {}
        for word in self.sentiment:
            stem_word = self.stemmer.stem(word)
            stemmed[stem_word] = self.sentiment[word]

        negations = ["no", "not", "none", "noone", "nobody", "nothing", "neither", "nowhere", "never", "didn't", 
        "hardly", "scarcely", "barely", "doesn’t", "isn’t", "wasn’t", "shouldn’t", "wouldn’t", "couldn’t", "won’t", "can’t", "don’t"]

        multipliers = ["very", "really", "extremely", "reeally", "so", "exceedingly", "vastly", "hugely", "tremendously", "extraordinarily", "especially"]
        highEmotions = ["love", "hate", "terrible", "awful", "miserable", "dreadful", "amazing", "incredible", "shocked"]


        count = 0

        
        negationSeen = False
        shouldSwitch = False
        shouldMultiply = False
        multiplierSeen = False
        
        for i in range(len(highEmotions)):
            highEmotions[i] =  self.stemmer.stem(highEmotions[i])
        for i in range(len(negations)):
            negations[i] =  self.stemmer.stem(negations[i])
        for i in range (len(multipliers)):
            multipliers[i] =  self.stemmer.stem(multipliers[i])

        words = preprocessed_input.split(' ')
        for i in range(len(words)):
            highEmotion = False
            word = self.stemmer.stem(words[i]).lower()
            # word = words[i].lower()
            # Another way to handle punctuation: word[len(word) - 1]
            # if reach negation -> switch
            # if negation has been seen and we reacch punctuation -> switch
            
            if (word in negations):
                negationSeen = True
                shouldSwitch = True
            if (negationSeen and any(p in word and not p == "\'" for p in string.punctuation)): 
                shouldSwitch = not shouldSwitch
                negationSeen = False
            
            for letter in word: 
                if letter in string.punctuation: 
                    word = word.replace(letter, "")
            word = self.stemmer.stem(word)
        
            if word in highEmotions:
                highEmotion = True

            if(self.creative):
                if (word in multipliers):
                    shouldMultiply = True
                    multiplierSeen = True
                if (multiplierSeen and any(p in word for p in string.punctuation)): 
                    shouldMultiply = not shouldMultiply
                    multiplierSeen = False
            if word in stemmed:
                #print("recognized the word!! ", word)
                num = 1
                if(self.creative):
                    if (shouldMultiply or highEmotion):
                        num *=2
                    if (shouldSwitch):
                        num *= -1
                if stemmed[word] == 'pos':
                    if(self.creative):
                        count += num
                    else:
                        if (shouldSwitch):
                            count -= 1
                        else:
                            count += 1
                            
                else:
                    if(self.creative):
                        count -= num
                    else:
                        if (shouldSwitch):
                            count += 1
                        else:
                            count -= 1
            #else:
                #print("did not recognize the word :( ", word)
                    
        
        if (not self.creative):
            if(count >= 1):
                return 1
            elif (count == 0):
                return 0
            return -1
        else: 
            if(count >= 2):
                return 2
            elif (count <= -2):
                return -2
            return count


    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        movies = self.extract_titles(preprocessed_input)
        preprocessed_input = re.sub('"([^"]*)"', ' ', preprocessed_input)
        input = re.findall('([\w(?:\'\w)?]+)',preprocessed_input)
        start = -1
        end = len(input)
        prev = 0
        answer = []
        currdiv = ""
        nextdiv = ""
        dividers = ['and','but','or','yet']
        for movie in movies:
            if (start >= len(input)-1):
                senti = prev
                answer.append((movie, senti))
                break
            for i in range(start+1, end):
                if input[i] in dividers:
                    end = i
                    nextdiv = input[i]
                    break
            sentence = ""
            for i in range(start+1,end):
                sentence += input[i] + " "
            local = self.extract_sentiment(sentence)
            if input[end-1] == "not":
                senti = -1
            elif local == 0 and (currdiv == "and" or currdiv == "or"):
                senti = prev + local
            else: senti = local
            prev = senti
            answer.append((movie, senti))
            start = end 
            currdiv = nextdiv
            end = len(input)
        return answer

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")
          


        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        spell = []
        minimum = max_distance + 3
        spell_distances = []
        #spell[i] contains the index of the movie that has an edit distance of spell_distances[i]


        articles = ["a", "an", "the"]
        title = title.lower()
        realTitle = title
        containsYear = re.findall('\(\d{4}\)', title)

        for article in articles:
            size = len(article)
            if (title[0:size] == article):
                if(len(containsYear) == 0):
                    realTitle = title[size+1:].strip() + ", " + article 
                else:
                    realTitle = title[size+1:-6].strip() + ", " + article + " " + title[-6:]
        for i in range(len(self.movieTitles)):
            movie = self.movieTitles[i]
            trueTitle = ""
            if (len(containsYear) != 0):
                trueTitle = movie[0].lower()
            else:
                trueTitle = movie[0][:-7].lower()
    

            edit_dist = self.edit_distance(trueTitle, realTitle)
            #print(edit_dist)
            if (edit_dist <= max_distance):
                minimum = np.array([minimum, edit_dist]).min()
                spell.append(i)
                spell_distances.append(edit_dist)
            #else:
                #print("not ", trueTitle)

        smallSpell = []
        
        for i in range(len(spell)):
            if (spell_distances[i] <= minimum):
                #print(spell_distances[i])
                smallSpell.append(spell[i])
        return smallSpell

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        #clarification = "harry potter"

        funnel = []
        tokens = clarification.lower().split()
        #print("tokens: ", tokens)

        # disambiguate part 2
        for movie_index in candidates:
            title = re.sub(r'[()]', '', self.titles[movie_index][0])
            #print(movie_index, "title: ", title)
            match = True
            for t in tokens:
                if t not in title.lower().split():
                    match = False

            if match == True:
                funnel.append(movie_index)

        # disambiguate part 3 cases
        if len(funnel) == 0:
            order = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5,
            "seventh": 6, "eighth": 7, "nineth": 8, "tenth": 9}

            # if clarification is an integer
            if clarification.isdigit():
                index = int(clarification)
                funnel.append(candidates[index-1])

            # if clarification is just "most recent"
            elif clarification == "most recent":
                recent_year = float('-inf')
                recent_index = None
                for c in candidates:
                    year = re.findall(r'\(.*?\)', self.movieTitles[c][0])[0]
                    year = int(year.replace('(','').replace(')',''))
                    if year > recent_year:
                        recent_year = year
                        recent_index = c
                funnel.append(recent_index)

            # case 1: clarification contains an order word, i.e. "first", "third"
            # case 2: "[title tokens] one", i.e. "the goblet of fire one" (order_case = False)
            else:
                order_case = False
                for key in order:
                    if key in tokens:
                        order_case = True
                        funnel.append(candidates[order[key]])
                        break

                if order_case == False:
                    # print("input tokens", tokens)
                    if tokens[len(tokens)-1] == 'one':
                        for c in candidates:
                            curr_tokens = self.titles[c][0].lower().split()
                            # print("current tokens", c, curr_tokens)
                            if set(tokens[:-1]).issubset(set(curr_tokens)):
                                funnel.append(c)
                                break

        return funnel


    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        # binarized_ratings = ratings > threshold
       
        # binarized_ratings = np.where(ratings > threshold, 1, -1) and np.where
        
        for i in range(len(ratings)):
            for j in range(len(ratings[0])):
                if ratings[i][j] == 0:
                    binarized_ratings[i][j] = 0
                elif ratings[i][j] > threshold:
                    binarized_ratings[i][j] = 1
                else:
                    binarized_ratings[i][j] = -1
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        if  (np.linalg.norm(u) * np.linalg.norm(v)) == 0:
            return 0
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        # add if statment
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity
    
    def find_emotion(self, sentence):
        filepath = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        sentence = re.sub('"([^"]*)"', ' ', sentence)
        words = sentence.split(' ')
        negations = ["no", "not", "none", "noone", "nobody", "nothing", "neither", "nowhere", "never", "didn't", 
        "hardly", "scarcely", "barely", "doesn’t", "isn’t", "wasn’t", "shouldn’t", "wouldn’t", "couldn’t", "won’t", "can’t", "don’t"]
        for i in range(len(negations)):
            negations[i] =  self.stemmer.stem(negations[i])
        negationSeen = False
        shouldSwitch = False

        '''I got this lexicon from the merriam-webster theasuarus. I originally tried to use some lexicons online, but I ended up just using a theaserasus instead so that I didn;t have to import anything.''' 

        fear = ["fear", "alarm", "anxiety", "dread", "fearfulness", "fright", "horror", "panic", "scare", "terror", "trepidation", "phobia", "creeps", "jitters", "nervousness", "willies", "pang", "qualm", "twinge", 
        "agitation", "apprehension", "consternation", "discomposure", "disquiet", "funk", "perturbation", "concern", "dismay", "worry", "cowardice", "faintheartedness", "timidity", "timorousness"]
        anger = ["anger", "angriness", "choler", "furor", "fury", "indignation", "irateness", "ire", "lividity", "lividness", "mad", "madness", "mood", "outrage", "rage", "spleen", "wrath", "wrathfulness",
        "aggravation", "annoyance", "exasperation", "irritation", "vexation", "acrimoniousness", "acrimony", "animosity", "antagonism", "antipathy", "bile", "biliousness", "bitterness", "contempt",
        "embitterment", "empoisonment", "enmity", "grudge", "hostility", "rancor", "envy", "jaundice", "jealousy", "pique", "resentment", "malevolence", "malice", "spite", "vengefulness", "venom", "vindictiveness", "virulence", "vitriol",
        "belligerence", "contentiousness", "contrariness", "crankiness", "disputatiousness"]
        anticipation = ["anticipation", "contemplation", "expectance", "expectancy", "expectation", "prospect"]
        trust = ["trust", "rely", "believe", "assign", "charge", "commission", "entrust", "task", "confer", "impose", "commit", "confide", "consign", "delegate", "recommend", "relegate", "repose", "allocate", "allot", "authorize", "empower", "invest", 
        "commend", "commit", "confide", "consign", "delegate", "deliver", "entrust", "give", "vest"]
        surprise = ["bombshell", "jar", "jaw-dropper", "jolt", "stunner", "shock", "thunderclap", "eye-opener", "revelation", "shocker", "amazement", "marvel", "wonder", "fillip", "kick", "kicker", 'twist', "wrinkle"]
        disgust = ["aversion", "distaste", "horror", "loathing", "nausea", "repugnance", "repulsion", "revulsion", "abhorrence", "abomination", "antipathy", "execration", "hate", "hatred",
        "allergy", "averseness", "disapproval", "disfavor", "disinclination", "dislike", "disliking", "displeasure"]
        joy = ["elatedness", "love", "like", "pleasure", "good", "elation", "exhilaration", "exultation", "high", "intoxication", "ecstasy", "euphoria", "glory", "heaven", "nirvana", "paradise", "rapture", "rapturousness", "ravishment", "transport", "delectation", "delight", "enjoyment", "pleasure", 
        "cheer", "cheerfulness", "comfort", "exuberance", "gaiety", "gladsomeness", "glee", "gleefulness", "jocundity", "jollity", "joyfulness", "joyousness", "jubilance", "jubilation", "lightheartedness", "merriness", "mirth", "content", "contentedness", "gratification", "satisfaction", "triumph"]
        emotion_names = ["fear", "anger", "anticipation", "trust", "disgust", "surprise", "joy"]
        total_emotions = [fear, anger, anticipation, trust, disgust, surprise, joy]
        emotion_scores = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(total_emotions)):
            for j in range(len(total_emotions[i])):
                total_emotions[i][j] = wordy = self.stemmer.stem(total_emotions[i][j])


        for wordy in words:
            wordy = wordy.lower()
            wordy = self.stemmer.stem(wordy)
            if (wordy in negations):
                negationSeen = True
                shouldSwitch = True
            if (negationSeen and any(p in wordy and not p == "\'" for p in string.punctuation)): 
                shouldSwitch = not shouldSwitch
                negationSeen = False

            if (not shouldSwitch):
                for letter in wordy: 
                    if letter in string.punctuation: 
                        wordy = wordy.replace(letter, "")
            
                for i in range(len(total_emotions)):
                    if wordy in total_emotions[i]:
                        emotion_scores[i] += 1
        if (max(emotion_scores) > 0):
            return emotion_names[emotion_scores.index(max(emotion_scores))]
        else:
            return "no sentiment"
        
            

        

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        recommendations = []
        # loop through reccommendations until we get k recommendations we have not seen
        check_indices = []
        rated_indices = []
        for i in range(len(user_ratings)):
            if user_ratings[i] == 0:
                check_indices.append(i)
            else:
                rated_indices.append(i)
        rankings = []
        for i in check_indices:
            rxi = 0
            movie_vec = ratings_matrix[i]
            for j in rated_indices:
                movie_curr = ratings_matrix[j]
                cosine_similarity = self.similarity(movie_vec, movie_curr)
                rxi += cosine_similarity * user_ratings[j]
            rankings.append((i, rxi))            
        rankings.sort(key = lambda x: x[1], reverse=True)
        for i in range(k):
            recommendations.append(rankings[i][0])
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """
        return "Hi! My name is JERC and I'm a movie chatbot."


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
