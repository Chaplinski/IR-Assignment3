#Python 3.5.2
import re
import os
import collections
import time
import math
import sys
import json
import random

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
        self.champion_list = {}
        self.top_k = 10
        self.doc_lengths=[]
        self.cluster_dict={}
        self.doc_ID_list=[] # list to map docIDs to Filenames, docIDs rn=ange from 0 to n-1 where n is the number of documents.
        start = time.time()
        self.get_stop_words()
        self.buildIndex()
        self.total_number_of_documents = len(self.doc_ID_list)
        self.calculate_idf()
        self.calculate_tf()
        end = time.time()
        print("Index built in ", (end - start), " seconds.")
        self.calculate_doc_lengths()
        # self.print_dict()
        # probably don't need print(self.word_count_per_doc)
        # self.print_dict()
        # self.ask_for_query()
        # print('exact query:')
        # self.exact_query()
        # self.ask_for_query()
        # print('inexact query:')
        # self.inexact_query_index_elimination()
        # self.inexact_query_champion()
        self.inexact_query_cluster_pruning()

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

    def inexact_query_cluster_pruning(self):
        # TODO do this function now
        lead_follow_list = self.get_leaders_and_followers()
        leaders = lead_follow_list[0]
        followers = lead_follow_list[1]
        print(leaders)
        print(followers)
        cluster_dict = {}
        for leader in leaders:
            cluster_dict[leader] = []

        numerator = 0
        # calculate the leader that each follower clusters with
        for follower in followers:
            if follower != 422:
                current_best_cosine = 0
                for leader in leaders:
                    if leader != 422:
                    # for every word in the dictionary
                        for key, value in self.dictionary.items():
                            leader_tf_idf = 0
                            follower_tf_idf = 0
                            # print(key, value)
                            doc_idf = value[0]
                            # for every document that contains this word
                            for doc in value:
                                # skip the idf at beginning of list
                                if doc != value[0]:
                                    # if this document is the leader document
                                    if doc[0] == leader:
                                        leader_tf_idf = doc[1] * doc_idf
                                        # print(doc)
                                        # print(doc[0])
                                        # print(doc[1])
                                        # print('leader tfidf:', leader_tf_idf)
                                        # sys.exit()
                                    elif doc[0] == follower:
                                        follower_tf_idf = doc[1] * doc_idf
                                        # print(doc)
                                        # print(doc[0])
                                        # print(doc[1])
                                        # print('follower tfidf:', follower_tf_idf)
                                        # sys.exit()
                                    # if tf-idf of leader and follower have already been found then move to next word
                                    if leader_tf_idf > 0 and follower_tf_idf > 0:
                                        break
                        # if there is a tfidf value for this word for the leader and follower add their product to numerator
                            if leader_tf_idf > 0 and follower_tf_idf > 0:
                                numerator += leader_tf_idf * follower_tf_idf
                        doc_similarity = numerator / (self.doc_lengths[leader] * self.doc_lengths[follower])
                        # print('leader', leader, 'follower', follower, 'similarity:', doc_similarity)

                        numerator = 0

                    if doc_similarity > current_best_cosine:
                        current_best_cosine = doc_similarity
                        current_best_leader = leader
                # print(current_best_cosine)
                # print(current_best_leader)
                cluster_dict[current_best_leader].append(follower)
                # print(cluster_dict)
                # sys.exit()
        self.cluster_dict = cluster_dict

    def get_leaders_and_followers(self):
        # function for exact top K retrieval using cluster pruning (method 4)
        # Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        i = 0
        doc_id_list = []
        # create int id list from string id list
        for doc in self.doc_ID_list:
            doc_id_list.append(i)
            i += 1
        # get number of docs to be leaders
        N = len(doc_id_list)
        N_root = math.floor(math.sqrt(N))
        leader_list = []
        follower_list = doc_id_list
        # get leader list and remove leaders from follower list
        for doc in doc_id_list:
            new_leader = random.choice(follower_list)
            leader_list.append(new_leader)
            follower_list.remove(new_leader)
            N_root -= 1
            if N_root == 0:
                break

        return [leader_list, follower_list]


    def inexact_query_champion(self):
        # Function for exact top K retrieval using champion list (method 2)
        # Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        for key, value in self.dictionary.items():
            num_of_docs_with_term = len(value) - 1
            # set r
            r = self.r_formula(num_of_docs_with_term)
            temp_list = []

            for list in value:
                if list != value[0]:
                    temp_list.append(list)
            # sort in descending order
            sorted_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
            temp_list = []
            temp_list.append(value[0])
            for list in sorted_list:
                temp_list.append(list)
                r -= 1
                if r == 0:
                    break
            self.champion_list[key] = temp_list
        self.dictionary.clear()
        self.dictionary = self.champion_list
        self.exact_query()

    def r_formula(self, num_of_docs_with_term):
        if num_of_docs_with_term >= 10:
            # the more docs a word appears in the less valuable it is
            r = 10
        elif num_of_docs_with_term > 15:
            # the more docs a word appears in the less valuable it is
            r = 5
        else:
            # since these docs are more rare all of them should appear
            r = num_of_docs_with_term

        return r

    def exact_query(self):
        # #function for exact top K retrieval (method 1)
        # #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score

        self.get_tf_idf_dicts()

        # get query denominator
        denominator_query = 0
        for query_key in self.query_tf_idf_dict:
            denominator_query += self.query_tf_idf_dict[query_key] * self.query_tf_idf_dict[query_key]
        denominator_query = math.sqrt(denominator_query)

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
        self.get_query_tf_idf_dict()
        self.get_index_tf_idf_dict()

    def get_query_tf_idf_dict(self):
        # print(self.query_dict)
        # for each word in the query
        for word, list in self.query_dict.items():
            if word not in self.stop_words:
                # get tf-idf and store it
                tf_idf = self.query_dict[word][0] * self.query_dict[word][1]
                self.query_tf_idf_dict[word] = tf_idf

    def get_index_tf_idf_dict(self):
        # for each word in the query
        for key in self.query_tf_idf_dict:
            # print('word:', key)
            # get idf
            idf = self.dictionary[key][0]
            # print('idf:', idf)
            for item in self.dictionary[key]:
                # get every item
                if item != self.dictionary[key][0]:
                    # print(item)
                    # sys.exit()
                    tf = item[1]
                    tf_idf = idf * tf

                    if item[0] not in self.index_tf_idf_dict:
                        self.index_tf_idf_dict[item[0]] = {key: tf_idf}
                    else:
                        self.index_tf_idf_dict[item[0]][key] = tf_idf

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
        # print('original query dict:', self.query_dict)

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
