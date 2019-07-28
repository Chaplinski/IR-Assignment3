#Python 3.5.2
import re
import os
import collections
import time
import math
import random
#import other modules as needed

class index:
    def __init__(self,path):
        self.collection = path
        self.dictionary = {}
        self.query_terms = []
        self.query_string = ''
        self.query_dict = {}
        self.stop_words = []
        self.query_tf_idf_dict = {}
        self.index_tf_idf_dict = {}
        self.top_k = 20
        self.doc_lengths = []
        self.query_length = 0
        self.cluster_dict = {}
        self.doc_ID_list = []  # list to map docIDs to Filenames, docIDs rn=ange from 0 to n-1 where n is the number of documents.
        start = time.time()
        self.buildIndex()
        self.total_number_of_documents = len(self.doc_ID_list)
        self.calculate_idf()
        self.calculate_tf()
        end = time.time()
        print("TF-IDF Index built in ", (end - start), " seconds.")
        self.calculate_doc_lengths()
        self.exact_query()

    def buildIndex(self):
        # Function to read documents from collection, tokenize and build the index with tokens
        # index should also contain positional information of the terms in the
        # document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,..]),..]
        # use unique document IDs
        docID = -1
        for filename in os.listdir(self.collection):
            docID = docID + 1
            self.doc_ID_list.append(filename)
            # read a document
            lines = open(self.collection + filename).read()
            tokens = re.split('\W+', lines.lower());
            self.insert_terms(tokens, docID)
        self.dictionary = collections.OrderedDict(sorted(self.dictionary.items()))

    def insert_terms(self,tokens,docID):
        #add terms to the dict data structure
        #check if term already exists
        pos=-1
        for tok in tokens:
            pos+=1
            if tok not in self.stop_words:
                if tok == '':
                    pass
                elif tok in self.dictionary: # faster
                    flag=False
                    for index, item in enumerate(self.dictionary[tok]):
                        if item[0] == docID: #add new pos to existing docID entry
                            item[1].append(pos)
                            flag=True
                            break
                    if flag == False: #first pos for docID for this tok
                        item=self.dictionary[tok]
                        item.append((docID,[pos]))

                else:   #new docID for this tok
                    item=[(docID,[pos])]
                    self.dictionary[tok]=item

    def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma):
        # function to implement rocchio algorithm
        # pos_feedback - documents deemed to be relevant by the user
        # neg_feedback - documents deemed to be non-relevant by the user
        # Return the new query  terms and their weights
        print('')

    def query(self, query_terms, k):
        # function for exact top K retrieval using cosine similarity
        # Returns at the minimum the document names of the top K documents ordered
        # in decreasing order of similarity score
        print('')

    def print_dict(self):
        # function to print the terms and posting list in the index
        for key, value in self.dictionary.items():
            print(key, value)

    def print_doc_list(self):
        # function to print the documents and their document id
        for index, item in enumerate(self.doc_ID_list):
            print('Doc ID: ', index, ' ==> ', item)

    def get_tf_idf_dicts(self):
        self.get_query_tf_idf_dict()
        self.get_index_tf_idf_dict()

    def get_query_tf_idf_dict(self):
        # for each word in the query
        for word, list in self.query_dict.items():
            if word not in self.stop_words:
                # get tf-idf and store it
                tf_idf = self.query_dict[word][0] * self.query_dict[word][1]
                self.query_tf_idf_dict[word] = tf_idf

    def get_index_tf_idf_dict(self):
        # for each word in the query
        for key in self.query_tf_idf_dict:
            # get idf
            idf = self.dictionary[key][0]
            # print('idf:', idf)
            for item in self.dictionary[key]:
                # get every item
                if item != self.dictionary[key][0]:
                    tf = item[1]
                    tf_idf = idf * tf

                    if item[0] not in self.index_tf_idf_dict:
                        self.index_tf_idf_dict[item[0]] = {key: tf_idf}
                    else:
                        self.index_tf_idf_dict[item[0]][key] = tf_idf

    def calculate_idf(self):
        # for each term in dictionary calculate and add the IDF
        for key, value in self.dictionary.items():
            # calculate IDF
            idf = math.log10(self.total_number_of_documents/len(value))
            # add IDF to list in the first position
            value.insert(0, idf)

    def calculate_tf(self):
        # for each dictionary term
        for key, value in self.dictionary.items():
            i = 1
            for item in value:
                if type(item) is tuple:
                    item = list(item)
                    # measure the number of times a word appears in a doc
                    number_of_appearances_in_doc = len(item[1])
                    tf = number_of_appearances_in_doc
                    # calculate w
                    w = (1 + math.log10(tf))
                    # insert into list
                    item.insert(1, w)
                    item = tuple(item)
                    value[i] = item
                    i += 1

    # get query and store terms as a list
    def ask_for_query(self):
        self.query_dict.clear()
        self.query_tf_idf_dict.clear()
        self.index_tf_idf_dict.clear()
        self.query_terms.clear()
        self.query_string = input("Enter your query: ")
        temp_terms = self.convert_string_to_list(self.query_string)
        for word in temp_terms:
            if word in self.dictionary:
                self.query_terms.append(word)
        self.create_query_dict()

    def convert_string_to_list(self, contents):
        # remove all punctuation and numerals, all text is lowercase
        contents = contents.lower()
        contents = re.sub(r'\d+', '', contents)

        # remove punctuation, replace with a space
        for char in "-:\n":
            contents = contents.replace(char, ' ')

        # remove quotes and apostrophes, replace with empty string
        for char in ".,?!'();$%\"":
            contents = contents.replace(char, '')
        contents = contents.replace('\n', ' ')

        # convert string to list
        contents = contents.split(' ')
        # remove empty strings
        contents = list(filter(None, contents))

        return contents

    def create_query_dict(self):
        this_dict = {}
        # for each word in query
        for item in self.query_terms:
            # if the word is not in the dictionary add it with a value of 1 occurrence
            if item not in this_dict:
                this_dict.update({item: 1})
            else:
                number_of_occurrences = this_dict.get(item)
                number_of_occurrences += 1
                this_dict.update({item: number_of_occurrences})

        # for each word in the query, calculate the tf-idf
        for key, value in this_dict.items():
            # calculate tf
            number_of_appearances_in_query = value
            # divide it by the word count of the entire query
            tf = number_of_appearances_in_query
            # calculate w
            w = (1 + math.log10(tf))

            idf = 0
            # get idf that is already stored in inverted index
            if key in self.dictionary:
                idf = self.dictionary[key][0]

            # store w and idf as value for word key in this dict
            this_dict[key] = [w, idf]
        self.query_dict = this_dict

    def exact_query(self):
        # #function for exact top K retrieval (method 1)
        # #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        self.ask_for_query()
        self.query_helper('exact retrieval')

    def query_helper(self, retrieval_type):
        start = time.perf_counter()
        self.get_tf_idf_dicts()

        denominator_query = self.get_query_denominator()

        top_k_dictionary = {}
        # for each text id held in index_tf_idf_dict
        for index_key, index_dictionary in self.index_tf_idf_dict.items():
            numerator = 0
            for word in index_dictionary:
                index_word_tf_idf = index_dictionary[word]
                query_word_tf_idf = self.query_tf_idf_dict[word]
                numerator += query_word_tf_idf * index_word_tf_idf

            denominator_index = self.doc_lengths[index_key]
            denominator_final = denominator_query * denominator_index
            cosine_sim = numerator/denominator_final
            top_k_dictionary[index_key] = cosine_sim

        top_k_dictionary_sorted_keys = sorted(top_k_dictionary, key=top_k_dictionary.get, reverse=True)

        i = self.top_k
        end = time.perf_counter()
        print('Query length =', self.query_length)
        print('Top ', self.top_k, ' results for the query \'', self.query_string, '\' using ', retrieval_type, ' method are:', sep='')
        j = 1
        for r in top_k_dictionary_sorted_keys:
            for index, item in enumerate(self.doc_ID_list):
                if r == index:
                    print(j, '. ', item, ' with score ', top_k_dictionary[index], sep='')
                    i -= 1
                    j += 1
                    break
            if i == 0:
                break
        print('Results found in', end - start, 'seconds')
        print('')

    def get_query_denominator(self):
        denominator_query = 0
        for query_key in self.query_tf_idf_dict:
            denominator_query += self.query_tf_idf_dict[query_key] * self.query_tf_idf_dict[query_key]
        denominator_query = math.sqrt(denominator_query)
        self.query_length = denominator_query
        return denominator_query

    def calculate_doc_lengths(self):
        # initialize every doc length at 0
        for document in self.doc_ID_list:
            self.doc_lengths.append(0)

        # for each word
        for word, list in self.dictionary.items():
            # for every value in the list associated with every word
            for value in list:
                doc_length = 0
                # if the value is not the idf
                if value != list[0]:
                    tf_idf = value[1] * list[0]
                    # get tf-idf squared
                    doc_length += tf_idf * tf_idf
                    # and add it to the current doc length
                    self.doc_lengths[value[0]] += doc_length

        temp_list = []
        for doc in self.doc_lengths:
            root_length = math.sqrt(doc)
            temp_list.append(root_length)
        self.doc_lengths = temp_list

    # === Testing === #
a=index("collection/")
# print(a.query_terms)
# print(a.query_dict)
# === End of Testing === #
