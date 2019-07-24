#Python 3.5.2
import re
import os
import collections
import time
import math
import sys

class index:
    def __init__(self,path):
        self.collection=path
        self.dictionary={}
        # probably don't need self.word_count_per_doc=[]
        self.query_terms=''
        self.query_dict={}
        self.stop_words=[]
        self.query_tf_idf_dict={}
        self.index_tf_idf_dict={}
        self.top_k = 10
        self.doc_ID_list=[] # list to map docIDs to Filenames, docIDs rn=ange from 0 to n-1 where n is the number of documents.
        start = time.time()
        self.get_stop_words()
        self.buildIndex()
        self.total_number_of_documents = len(self.doc_ID_list)
        self.calculate_idf()
        self.calculate_tf()
        end = time.time()
        print("Index built in ", (end - start), " seconds.")
        # probably don't need print(self.word_count_per_doc)
        # self.print_dict()

        self.ask_for_query()

        print('exact query:')
        self.exact_query()

        self.ask_for_query()

        print('inexact query:')
        self.inexact_query_index_elimination()


    def buildIndex(self):
        #Function to read documents from collection, tokenize and build the index with tokens
        #index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,..]),..]
        #use unique document IDs
        docID=-1
        for filename in os.listdir(self.collection):
            docID=docID+1
            self.doc_ID_list.append(filename)
            #read a document
            lines = open(self.collection + filename).read()
            #tokens = re.split(r'[^A-Za-z]', lines.lower())
            tokens = re.split('\W+',lines.lower());
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
                #elif tok in self.dictionary.keys():
                    flag=False
                    for index, item in enumerate(self.dictionary[tok]):
                        if item[0] == docID: #add new pos to existing docID entry
                            item[1].append(pos)
                            #self.dictionary[tok][index]=item
                            flag=True
                            break
                    if flag == False: #first pos for docID for this tok
                        item=self.dictionary[tok]
                        item.append((docID,[pos]))

                else:   #new docID for this tok
                    item=[(docID,[pos])]
                    self.dictionary[tok]=item

    def print_dict(self):
    #function to print the terms and posting list in the index
        for key, value in self.dictionary.items():
            print(key, value)

    def get_postinglist(self, term):
        results=[]
        for index,item in enumerate(self.dictionary[term]):
            results.append(item[0])
        return results

    def print_doc_list(self):
    # function to print the documents and their document id
        for index, item in enumerate(self.doc_ID_list):
            print('Doc ID: ', index, ' ==> ',item)

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

    def exact_query(self):
        # #function for exact top K retrieval (method 1)
        # #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score

        self.get_tf_idf_dicts()

        top_k_dictionary = {}
        # for each text id held in index_tf_idf_dict
        for index_key, index_dictionary in self.index_tf_idf_dict.items():
            # print(index_key)
            # print(index_dictionary)
            # sys.exit()
            numerator = 0
            denominator_index = 0
            denominator_query = 0
            # print('index dictionary: ', index_dictionary)
            # loop through query terms
            for query_key, query_value in self.query_tf_idf_dict.items():
                # print(self.query_tf_idf_dict)
                # print('Query value: ', query_value)
                # for key, value in query_value.items():
                if query_key in self.query_terms:
                    # print(query_tf_idf_dict)
                    # sys.exit()
                    # if the query key exists in the index_dictionary
                    if query_key in index_dictionary:
                        index_tf_idf = index_dictionary[query_key]
                    else:
                        index_tf_idf = 0
                    query_tf_idf = query_value
                    add_to_numerator = index_tf_idf * query_tf_idf
                    numerator += add_to_numerator

                    denominator_index += index_tf_idf * index_tf_idf
                    denominator_query += query_tf_idf * query_tf_idf
                    # print('index tf-idf:', index_tf_idf)
                    # print('query tf-idf: ', query_tf_idf)
                    # print('Number added:', add_to_numerator)
            # TODO Why is vector so high for document 1?
            # sys.exit()

            # for word, tf_idf in query_tf_idf_dict.items():
            denominator_index_root = math.sqrt(denominator_index)
            denominator_query_root = math.sqrt(denominator_query)
            denominator = denominator_index_root * denominator_query_root
            vector = numerator / denominator
            # print('Document', index_key, 'Numerator:', numerator)
            # print('Document', index_key, 'Index denominator:', denominator_index_root)
            # print('Document', index_key, 'Query denominator:', denominator_query_root)
            # print('Document', index_key, 'Denominator:', denominator)
            # print('Document', index_key, 'Vector:', vector)
            top_k_dictionary[index_key] = vector

        top_k_dictionary_sorted_keys = sorted(top_k_dictionary, key=top_k_dictionary.get, reverse=True)
        i = self.top_k
        print('Top K results:')
        for r in top_k_dictionary_sorted_keys:
            for index, item in enumerate(self.doc_ID_list):
                if r == index:
                    print(item)
                    i -= 1
                    break
            if i == 0:
                break

    def inexact_query_index_elimination(self):
        # function for exact top K retrieval using index elimination (method 3)
        # Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        temp_dict = {}
        for key, value in self.query_dict.items():
            temp_dict[key] = value[1]

        temp_dict_sorted = sorted(temp_dict, key=temp_dict.get, reverse=True)
        # get dictionary count
        list_length = len(temp_dict_sorted)
        new_list_size = math.ceil(list_length/2)
        temp_dict_2 = {}
        temp_query_list = []
        for word in temp_dict_sorted:
            for key, value in self.query_dict.items():
                if word == key:
                    temp_dict_2[word] = value
                    temp_query_list.append(word)
            new_list_size -= 1
            if new_list_size == 0:
                break
        self.query_dict.clear()
        self.query_dict = temp_dict_2
        self.query_terms = temp_query_list
        print('query terms: ', self.query_terms)
        # now that query dict has been halved simply call exact_query to function on the halved query
        self.exact_query()

    def get_tf_idf_dicts(self):
        # build the dictionaries containing query terms and index terms and their ids and tf-idf

        # for each word in the query
        for word, list in self.query_dict.items():
            # if that word is not a stop word
            if word not in self.stop_words:
                # get tf-idf of this query term
                # print(word)
                # print(list)

                query_term_tf_idf = list[0] * list[1]
                self.query_tf_idf_dict[word] = query_term_tf_idf
                # print('list:', list, 'tf-idf:', query_term_tf_idf)
                # if the word appears in self.index
                if word in self.dictionary:
                    # get idf from the dictionary
                    dictionary_idf = self.dictionary[word][0]
                    # print(word, 'is in the index and its idf value is:', dictionary_idf)
                    for doc_id_list in self.dictionary[word]:
                        # skip the first list item since it is the idf
                        if doc_id_list != self.dictionary[word][0]:
                            # get doc id
                            doc_id = doc_id_list[0]
                            # get word tf
                            word_tf = doc_id_list[1]
                            # get tf-idf of word appearing in this document
                            doc_word_tf_idf = word_tf * dictionary_idf
                            if doc_id in self.index_tf_idf_dict:
                                # if doc id already exists in index_tf_idf_dict then add to the dictionary that
                                # is its value
                                self.index_tf_idf_dict[doc_id][word] = doc_word_tf_idf
                                # print('doc id is:', doc_id, 'and tf-idf is:', doc_word_tf_idf)
                            else:
                                # if doc id does not exist in index_tf_idf_dict then add doc id and dictionary
                                # as value. Also add current word and its tf-idf
                                self.index_tf_idf_dict[doc_id] = {}
                                self.index_tf_idf_dict[doc_id][word] = doc_word_tf_idf
                                # print('doc id is:', doc_id, 'and tf-idf is:', doc_word_tf_idf)
                                # sys.exit()

        print('Query tf-idf:', self.query_tf_idf_dict)
        print('Index tf-idf', self.index_tf_idf_dict)

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

    # get query and store terms as a list
    def ask_for_query(self):
        self.query_tf_idf_dict.clear()
        self.index_tf_idf_dict.clear()
        query = input("Enter your query: ")
        self.query_terms = self.convert_string_to_list(query)
        self.create_query_dict()
        # create dictionary to keep track of word occurrences in query

    def create_query_dict(self):
        print('query dict:', self.query_dict)
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

            idf = ''
            # get idf that is already stored in inverted index
            if key in self.dictionary:
                idf = self.dictionary[key][0]

            # store w and idf as value for word key in this dict
            this_dict[key] = [w, idf]

        self.query_dict = this_dict
        print('original query dict:', self.query_dict)

    def get_stop_words(self):
        f = open('stop-list/stop-list.txt', "r")
        contents = f.read()
        contents = self.convert_string_to_list(contents)
        self.stop_words = contents

    # === Testing === #
a=index("collection/")
# print(a.query_terms)
# print(a.query_dict)
# === End of Testing === #
