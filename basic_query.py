import json
import math
import nltk
import index_constructor
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import webbrowser


class Query:
    def __init__(self, indexPath):
        self.loadIndex(indexPath)
        self.stop_words = set(stopwords.words('english'))
        
        # Generate window for GUI
        self.window = tk.Tk()
        self.window.title("Search Query")
        self.window.geometry("1500x800")

    def loadIndex(self, path):
        """
        Load and store the index for query search.
        Load and store the bookkeeping file to obtain the urls of the documents.
        """
        indexFile = open(path, "r")
        self.index = json.load(indexFile)
        bookkeepingFile = open("webpages/WEBPAGES_RAW/bookkeeping.json", "r")
        self.bookkeeping = json.load(bookkeepingFile)
        indexFile.close()
        bookkeepingFile.close()

    def search(self, normalized=None):
        """
        Take in user input and search through the index for relevant documents.
        Cos normalization is performed between the query and each relevant document and 
        tag weighting is taken into consideration in order to find the most relevant documents.
        """
        # Get user query from input box
        query = self.show_input_box.get()
        self.url = []
        query_dict = dict()
        lemmatizer = WordNetLemmatizer()
        
        # Reset search_result box if it contains any text so that new search results
        # can be shown
        if self.search_result.size() != 0:
            self.search_result.delete(0, tk.END)
            
        # Tokenize query
        text = nltk.word_tokenize(query.lower())
        tokens = nltk.pos_tag(text)
        filtered_tokens = []
        
        # Lemmatize query
        for word in tokens:
            if word[0].isalnum() and word[0] not in self.stop_words:
                if index_constructor.convert_pos_tag(word[1]) is None:
                    filtered_tokens.append(word[0])
                else:
                    filtered_tokens.append(lemmatizer.lemmatize(word[0],
                                                                pos=index_constructor.convert_pos_tag(word[1])))
        
        filtered_bigrams = nltk.bigrams(filtered_tokens)  # creates possible bigrams from query
        counted_bigrams = Counter(filtered_bigrams)
        counted_tokens = Counter(filtered_tokens)
        counted_tokens = counted_tokens | counted_bigrams  # Merge token and bigram Counters
        
        # Deal with empty query or only spaces query
        if len(counted_tokens) == 0:
            return
        
        # Calculates the tf of query if query word is in self.index
        for token in counted_tokens:
            if token in self.index:
                query_dict[token] = 1 + math.log(counted_tokens[token], 10)

        # Calculates the tf-idf by multiplying the tf with idf (idf being frequency of the token in index)
        for token in query_dict:
                query_dict[token] = query_dict[token] * math.log(37497 / len(self.index[token]), 10)
        
        # Check if query can produce results
        if len(query_dict) == 0:
            self.search_result.insert(tk.END,('Search - ' + query + ' - did not match any documents'))
            return
        
        # Normalize td-idf score
        cos_normalization = 0
        for token in query_dict:
            cos_normalization += query_dict[token] * query_dict[token]
            
        # Check length value b/c 1/0 causes ZeroDivisionError
        if cos_normalization > 0:
            cos_normalization = 1 / math.sqrt(cos_normalization)
        for token in query_dict:
            query_dict[token] = query_dict[token] * cos_normalization
            
        # Populating document vector keys
        document_vectors = dict()
        for token in query_dict:
            for location in self.index[token]:
                if token not in document_vectors:
                    document_vectors[location] = []
                    
        # Generates the document vector, if there is a token intersection append tdidf score if there is no
        # intersection append a zero. Order of vector corresponds to the word order of query.
        for token in query_dict:
            for location in document_vectors:
                if location not in self.index[token]:
                    document_vectors[location].append(0)
                else:
                    document_vectors[location].append(self.index[token][location][0])
                    
        # Normalize document vectors
        for document in document_vectors:
            doc_normalization = 0
            for n in document_vectors[document]:
                doc_normalization += n * n
            doc_normalization = 1 / math.sqrt(doc_normalization)
            normalized_doc_vector = []
            for n in document_vectors[document]:
                normalized_doc_vector.append(n * doc_normalization)
            document_vectors[document] = normalized_doc_vector
            
        # Calculates the cosine similarity score of a query tfidf vector and a document tfidf score.
        cos_similarity_dict = dict()
        for document in document_vectors:
            cos_similarity_dict[document] = cosine_similarity([list(query_dict.values())],
                                                              [document_vectors[document]])[0][0]
                                                              
        # Applies tag weights to cos_similarity scores.
        for token in query_dict:
            for location in document_vectors:
                if location not in self.index[token]:
                    pass
                else:
                    cos_similarity_dict[location] += self.index[token][location][1]
                    
        # Sorts the dict by score highest to lowest
        sorted_dict = dict(sorted(cos_similarity_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Finds top 20 urls
        count = 0
        for doc in sorted_dict:
            if count < 20:
                html = open("webpages/WEBPAGES_RAW/" + doc, mode='r', encoding="utf-8")
                document = BeautifulSoup(html, 'lxml')
                text = document.get_text(' ').split()
                
                # Check if webpage has a title
                # If so, print title, url, and description (first 20 words following the title)
                # If not, print url and description (first 20 words)
                if document.find('title') != None:
                    title = document.title.text
                    self.search_result.insert(tk.END, (str(count + 1) + ": " + title))
                    self.search_result.insert(tk.END, self.bookkeeping[doc])
                    del text[:len(title.split())]  # delete title from text to get description
                else:
                    self.search_result.insert(tk.END, (str(count + 1) + ": "))
                    self.search_result.insert(tk.END, self.bookkeeping[doc])
                
                # Only print words that are alphanumeric or ascii
                self.search_result.insert(tk.END, (' '.join(word for word in text[:20] if word.isascii() or word.isalpha()) + ". . ."), "\n")
                count += 1
            else:
                break

    def run(self):
        """
        Opens a search query GUI that allows the user to type query into a text box.
        Pressing 'ENTER' prints the title, url, and the first 20 words for the 
        top 20 webpages related to the query. Pressing 'DONE' ends the program.
        """
        instruct = tk.Label(self.window,
                            text='Enter query: ',
                            font=('Helvetica', '15', 'bold'))
        instruct.place(x=50, y=50)
        
        # Create input box to take in user input as the query
        self.show_input_box = tk.Entry(self.window, width=20,
                                       font=('Helvetica', '15'))
        self.show_input_box.place(x=50, y=80)
        
        # Pressing the 'ENTER' button triggers the search function
        # Any text within the input box will be used as the user query
        enter = tk.Button(self.window, text='SEARCH', bg='SlateGray1',
                          activebackground='SlateGray2',
                          font=('Courier', '13'),
                          command=self.search)
        enter.place(x=320, y=75)
        
        # Pressing the 'DONE' button destroys the GUI window and terminates the program
        exit_search = tk.Button(self.window, text='DONE', bg='SlateGray1',
                                activebackground='SlateGray2',
                                font=('Courier', '13'),
                                command=self.window.destroy)
        exit_search.place(x=825, y=525)

        v_scroll = tk.Scrollbar(self.window, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = tk.Scrollbar(self.window, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Create search_result box to display the top 20 urls
        # Scrollbars will allow for scrolling if url is too long
        self.search_result = tk.Listbox(self.window,
                                        height=25, width=100,
                                        font=('Helvetica', '10'),
                                        yscrollcommand=v_scroll.set,
                                        xscrollcommand=h_scroll.set)
        self.search_result.place(x=500, y=75)
        # Double clicking a link in the search results box will trigger link function
        self.search_result.bind('<Double-Button>', self.link) 
        v_scroll.configure(command=self.search_result.yview)
        h_scroll.configure(command=self.search_result.xview)

        self.window.mainloop()

    def link(self, *args):
        """
        Creates a clickable link that will take user directly to the webpage.
        Parts of this function, specifically the parts on opening a Chrome browser, were written
        with the help of a stackoverflow post that was linked in the M2 discussion slides.
        Source: https://stackoverflow.com/questions/22445217/python-webbrowser-open-to-open-chrome-browser
        """
        # Path to Chrome dependent on OS
        
        # Windows
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
        
        # MacOS
        #chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
        
        # Linux
        # chrome_path = '/usr/bin/google-chrome %s'
        element = self.search_result.curselection()[0]
        link = self.search_result.get(element)
        
        # Check if clicked text is url
        if 'ics.' in link:
            webbrowser.get(chrome_path).open(link)
        

if __name__ == "__main__":
    searchEngine = Query("index.json")
    searchEngine.run()
