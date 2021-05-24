# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import collections
import numpy as np
import string
from porter_stemmer import PorterStemmer

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=True):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'
        self.creative = creative
        self.stemmer = PorterStemmer()
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.movieTitles = util.load_titles('data/movies.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

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
        pattern = self.extract_titles(updatedLine)
        if len(pattern)> 0:
            print("So you loved ", pattern[0], ", huh?") 
            self.find_movies_by_title(pattern[0])
        else: 
            print("Sorry, I don't understand. Tell me about a movie that you have seen.")

        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

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
            return list(potential_titles.values())
        else: 
            return re.findall('"([^"]*)"', preprocessed_input)

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
        articles = ["a", "an", "the"]
        titles = []
        title = title.lower()
        realTitle = title
        containsYear = re.findall('\(\d{4}\)', title)
        ##Titanic  OR titanic, the (1997) the
        ##TODO: LOWERCASE VS UPPERCASE
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

        count = 0
        
        negationSeen = False
        shouldSwitch = False
        words = preprocessed_input.split(' ')
        for i in range(len(words)):
            word = self.stemmer.stem(words[i]).lower()
            # word = words[i].lower()
            # Another way to handle punctuation: word[len(word) - 1]
            # if reach negation -> switch
            # if negation has been seen and we reacch punctuation -> switch
            if (word in negations):
                #negationSeen = True
                shouldSwitch = True
                if (negationSeen and any(p in word for p in string.punctuation)):
                    shouldSwitch = not shouldSwitch
            if word in stemmed:
                if stemmed[word] == 'pos':
                    if (shouldSwitch):
                        count -= 1
                    else:
                        count += 1
                else:
                    if (shouldSwitch):
                        count += 1
                    else:
                        count -= 1
        
        if(count >= 1):
            return 1
        elif (count == 0):
            return 0
        return -1

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
        
        pass

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

        pass

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
        pass

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
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
