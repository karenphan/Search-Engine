import json
from bs4 import BeautifulSoup
from collections import Counter
import nltk
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import lxml


# nltk.download('popular')

def convert_pos_tag(pos_tag):
    """
    WordNetLemmatizer by default treats words as nouns so non-noun words don't
    get lemmatized correctly. Change the pos (part of speech) tag to the correct type
    in order to lemmatize all words.
    """
    if pos_tag == "N":
        return wordnet.NOUN
    elif pos_tag == "V":
        return wordnet.VERB
    elif pos_tag == "J":
        return wordnet.ADJ
    elif pos_tag == "R":
        return wordnet.ADV
    else:
        return None


def merge_bigrams(bigrams):
    """
    Merge bigram pairs into one string.
    """
    merged_bigrams = []
    for bigram in bigrams:
        merged_bigrams.append(bigram[0] + " " + bigram[1])
    return merged_bigrams


class IndexConstructor:
    def __init__(self):
        self.index = dict()  # store the index as a dictionary
        self.stop_words = set(stopwords.words('english'))

    def construct_index(self):
        """
        Create the index by iterating over all the documents and storing each unique word, along
        with the document(s) that the word appears in. The tf-idf of the word for each document along 
        with a tag weighting score is also stored in order to help with ranking the documents during
        the query search.
        Index format: {word: {docID: [tf-idf, tag weighting]}}
        """
        # open bookkeeping json file
        jsonFile = open("webpages/WEBPAGES_RAW/bookkeeping.json", "r")
        data = json.load(jsonFile)
        total_documents = len(data)
        lemmatizer = WordNetLemmatizer()
        # iterates over the json file's keys, which stores the path for the crawled files
        for location in data.keys():
            print(location)
            # open the html file
            html = open("webpages/WEBPAGES_RAW/" + location, mode='r', encoding="utf-8")
            document = BeautifulSoup(html, 'lxml')
            raw_text = document.get_text(' ')
            text = nltk.word_tokenize(raw_text.lower())  # creates tokens from raw html text
            tokens = nltk.pos_tag(text)
            filtered_tokens = self._filterTokens(tokens)  # filters tokens stopwords and non-alnum characters
            filtered_bigrams = nltk.bigrams(filtered_tokens)  # creates the bigram pairs
            counted_tokens = Counter(filtered_tokens)
            counted_bigrams = Counter(merge_bigrams(filtered_bigrams))
            counted_tokens = counted_tokens | counted_bigrams  # merges both Counter objects
            # calculates the tf of a location
            for token in counted_tokens:
                if token not in self.index:
                    self.index[token] = dict()
                self.index[token][location] = [1 + math.log(counted_tokens[token], 10), 0]

            # find important information (title, bold, heading tags)
            title = document.title
            boldTags = document.find_all('b')
            h1Tags = document.find_all('h1')
            h2Tags = document.find_all('h2')
            h3Tags = document.find_all('h3')
            anchorTags = document.find_all('a')

            # add weights to the index, includes the anchor text extra credit
            self._addWeightToImportantTokens(location, title, boldTags, h1Tags, h2Tags, h3Tags, anchorTags)

        jsonFile.close()
        # calculates the idf of a location and multiplies it with the tf producing the tf-idf
        for token in self.index:
            for location in self.index[token]:
                self.index[token][location][0] = self.index[token][location][0] * \
                                              math.log(total_documents / len(self.index[token]), 10)
        # write to json file
        with open("index.json", "w", encoding="utf-8") as file:
            json.dump(self.index, file, indent=4)
        file.close()

    def _filterTokens(self, tokens):
        """
        lemmatize and filter the tokens.
        """
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = []
        for word in tokens:
            if self._verify(word[0]):
                if convert_pos_tag(word[1]) is None:
                    filtered_tokens.append(word[0])
                else:
                    filtered_tokens.append(lemmatizer.lemmatize(word[0], pos=convert_pos_tag(word[1])))
        return filtered_tokens

    def _addWeightToImportantTokens(self, location, title, boldTags, h1Tags, h2Tags, h3Tags, anchorTags):
        """
        Parse the title, bold text, heading tags (h1,h2,h3) into tokens and add extra
        weights to those token in the index since they're more important.
        """
        try:
            if title and title.contents:    # title and its content exist
                titleTokens = nltk.word_tokenize(title.contents[0].lower()) # tokenize title's text
                tokens = nltk.pos_tag(titleTokens)
                filteredTitleTokens = self._filterTokens(tokens) # filter the tokens
                # increase the weight by 0.5 for each token in the title
                for token in filteredTitleTokens:
                    self.index[token][location][1] += 0.5
            if boldTags:
                for tag in boldTags:
                    if tag.string:
                        boldTokens = nltk.word_tokenize(tag.string.lower())
                        tokens = nltk.pos_tag(boldTokens)
                        filteredBoldTokens = self._filterTokens(tokens)
                        for token in filteredBoldTokens:
                            self.index[token][location][1] += 0.1
            if h1Tags:
                for tag in h1Tags:
                    if tag.string:
                        h1Tokens = nltk.word_tokenize(tag.string.lower())
                        tokens = nltk.pos_tag(h1Tokens)
                        filteredH1Tokens = self._filterTokens(tokens)
                        for token in filteredH1Tokens:
                            self.index[token][location][1] += 0.4
            if h2Tags:
                for tag in h2Tags:
                    if tag.string:
                        h2Tokens = nltk.word_tokenize(tag.string.lower())
                        tokens = nltk.pos_tag(h2Tokens)
                        filteredH2Tokens = self._filterTokens(tokens)
                        for token in filteredH2Tokens:
                            self.index[token][location][1] += 0.3
            if h3Tags:
                for tag in h3Tags:
                    if tag.string:
                        h3Tokens = nltk.word_tokenize(tag.string.lower())
                        tokens = nltk.pos_tag(h3Tokens)
                        filteredH3Tokens = self._filterTokens(tokens)
                        for token in filteredH3Tokens:
                            self.index[token][location][1] += 0.2
            # anchor tag extra credit
            if anchorTags:
                for tag in anchorTags:
                    if tag.string:
                        anchorTokens = nltk.word_tokenize(tag.string.lower())
                        tokens = nltk.pos_tag(anchorTokens)
                        filteredAnchorTokens = self._filterTokens(tokens)
                        for token in filteredAnchorTokens:
                            self.index[token][location][1] += 0.2
        except KeyError as e:
            print(e)

    def _verify(self, token):
        """
        Returns true if the token is in 0-127 ascii value, alphanumeric, and not a stop word.
        """
        if token.isascii() and token.isalnum() and token.lower() not in self.stop_words:
            return True
        return False


if __name__ == "__main__":
    indexConstructor = IndexConstructor()
    indexConstructor.construct_index()
