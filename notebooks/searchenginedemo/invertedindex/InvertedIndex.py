#
# simpleBool.py @ https://github.com/pyparsing/pyparsing/blob/master/examples/simpleBool.py
#
# Example of defining a boolean logic parser using
# the operatorGrammar helper method in pyparsing.
#
# In this example, parse actions associated with each
# operator expression will "compile" the expression
# into BoolXXX class instances, which can then
# later be evaluated for their boolean value.
#
# Copyright 2006, by Paul McGuire
# Updated 2013-Sep-14 - improved Python 2/3 cross-compatibility
# Updated 2021-Sep-27 - removed Py2 compat; added type annotations
#
from typing import Callable, Iterable

from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas, ParserElement

ParserElement.enablePackrat()


# define classes to be built at parse time, as each matching
# expression type is parsed
class BoolOperand:
    def __init__(self, t):
        self.label = t[0]
        self.value = eval(t[0])

    def __bool__(self) -> bool:
        return self.value

    def __str__(self) -> str:
        return self.label

    __repr__ = __str__


class BoolNot:
    def __init__(self, t):
        self.arg = t[0][1]

    def __bool__(self) -> bool:
        v = bool(self.arg)
        return not v

    def __str__(self) -> str:
        return "~" + str(self.arg)

    __repr__ = __str__


class BoolBinOp:
    repr_symbol: str = ""
    eval_fn: Callable[
        [Iterable[bool]], bool
    ] = lambda _: False

    def __init__(self, t):
        self.args = t[0][0::2]

    def __str__(self) -> str:
        sep = " %s " % self.repr_symbol
        return "(" + sep.join(map(str, self.args)) + ")"

    def __bool__(self) -> bool:
        return self.eval_fn(bool(a) for a in self.args)


class BoolAnd(BoolBinOp):
    repr_symbol = "&"
    eval_fn = all


class BoolOr(BoolBinOp):
    repr_symbol = "|"
    eval_fn = any


class BoolEval:
    def __init__(self):
        # define keywords and simple infix notation grammar for boolean
        # expressions
        TRUE = Keyword("True")
        FALSE = Keyword("False")
        NOT = Keyword("not")
        AND = Keyword("and")
        OR = Keyword("or")
        boolOperand = TRUE | FALSE | Word(alphas, max=1)
        boolOperand.setParseAction(BoolOperand).setName("bool_operand")

        # define expression, based on expression operand and
        # list of operations in precedence order
        self.boolExpr = infixNotation(
            boolOperand,
            [
                (NOT, 1, opAssoc.RIGHT, BoolNot),
                (AND, 2, opAssoc.LEFT, BoolAnd),
                (OR, 2, opAssoc.LEFT, BoolOr),
            ],
        ).setName("boolean_expression")





import math

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, input_file_name, lower, col, udf, explode, collect_list, sum
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.types import *
from collections import namedtuple
from itertools import groupby

from collections import Counter
import functools as tool


class InvertedIndex:
    # boolean parser
    boolean_parser = BoolEval()

    # patterns for base transformation
    pattern_1 = ".*\\/" # for filenames
    pattern_2 = "\\s+"
    pattern_3 = "\\(?<.*?>\\)?"
    # further example patterns to remove more unnecessary words (surround removal with pattern_2 again to ensure no stackoverflow via regex)
    pattern_4_extra = "(@import.*?;)|" + "(\\/\\*\\s*?\\*\\/)|" + "(&#.*?;)|" + "\\s(\\^|\\[|\\]|\\/\\/|\\/|\\.|\\,|\"|\\:|\\-|\\||\\(|\\))\\s"
    pattern_5_extra = "\\s(\\^|\\[|\\]|\\/\\/|\\/|\\.|\\,|\"|\\:|\\-|\\||\\(|\\))|(\\^|\\[|\\]|\\/\\/|\\/|\\.|\\,|\"|\\:|\\-|\\||\\(|\\))\\s"

    def __init__(self, html=True):
        print("Starting to load up...")

        # setting up spark session
        self.conf = pyspark.SparkConf()
        self.conf.setAppName("spark-exercise-2")
        self.conf.setMaster("local")
        self.conf.set("spark.sql.shuffle.partitions", "10")

        # start session
        self.sc = pyspark.SparkContext.getOrCreate(conf=self.conf)
        self.spark = SparkSession(self.sc)

        self.html = html

        print("Reading the corpus...")

        # read corpus
        files_path = './invertedindex/corpora/wiki-a' if html else './invertedindex/corpora/reuters'
        self.directory_path = 'searchenginedemo/invertedindex/corpora/wiki-a/' if html else 'searchenginedemo/invertedindex/corpora/reuters/'
        self.corpus = self.spark.read.text(files_path, wholetext=True)
        self.corpus = self.corpus.withColumn("filename", input_file_name()).cache()

        print("Tokenizing the corpus...")

        # tokenize
        self.corpus = self.run_transformations_on_corpus(html)

        print("Creating the inverted index...")

        # create inverted index and create a map consisting of the according (k:file - v:file length in normalised tf-idf) relation
        self.inverted_index = self.create_inv_index().cache()
        self.corpus_file_lens = self.calculate_file_lens(self.corpus.count(), [(row[0], row[1]) for row in self.inverted_index.select("df", "tfs").collect()])

        print("Finished!")


    def create_inv_index(self):
        # 1. initial #InvIndEntry, 2. explode arrays of #InvIndEntry in a row to multiple, 3. select wanted columns described via help tuple
        split = self.corpus.select("filename", "tokenized_reduced") \
            .withColumn("initialInvInd", self.init_inv_index(col("filename"), col("tokenized_reduced"))) \
            .withColumn("initialInvInd", explode(col("initialInvInd"))) \
            .select("initialInvInd.term", "initialInvInd.df", "initialInvInd.tfs")

        # aggregate rows by terms -> df by count, tfs by collecting in list and then evaluating the list of maps to one
        aggregated_maps = split.groupBy("term").agg(sum("df").alias("df"), self.tfs_aggregator(collect_list("tfs")).alias("tfs"))
        return aggregated_maps

    """
    see https://stackoverflow.com/questions/58416527/pyspark-user-defined-functions-inside-of-a-class
    """
    @staticmethod
    @udf(returnType=ArrayType(
        StructType([
            StructField("term", StringType(), True),
            StructField("df", IntegerType(), True),
            StructField("tfs", MapType(StringType(), IntegerType()), True)
        ])
    ))
    # helper function to transform the occurrence of a term in a file into the wanted structure of #InvIndEntry(term, 1, (file -> tf))
    def init_inv_index(filename, terms):
        InvIndEntry = namedtuple("InvIndEntry", ["term", "df", "tfs"])

        terms.sort()

        # group by identity as in scala ish; example: ["a", "a", "b", "c", "d", "d"] -> [["a", "a"], ["b"], ["c"], ["d", "d"]]
        grouped_by_identity = [list(i) for j, i in groupby(terms)]
        # create wanted structure list of (term, 1, {filename: tf}) entries
        res = list(map(lambda l: InvIndEntry(l[0], 1, {filename: len(l)}), grouped_by_identity))

        return res

    @staticmethod
    @udf(returnType=MapType(StringType(), IntegerType()))
    # helper function to aggregate a list of maps into one complete map (same key -> add values)
    def tfs_aggregator(tfs_list):
        # reduces a map by counting values up if they have the same key
        res = tool.reduce(lambda d1, d2: dict(Counter(d1)+Counter(d2)), tfs_list)

        return res

    def calculate_file_lens(self, size, df_tfs):
        res = {}
        for df, tfs in df_tfs:
            # w = tf * log_e(N/df)
            # len = sum(w^2)
            for file in tfs.keys():
                tf = tfs[file]
                current_tf = res.get(file, 0)
                new_tf = current_tf + math.pow((tf*(math.log(size/df, math.e))), 2)

                res[file] = new_tf
        # normalise all values by taking the square root of each value
        for file in res.keys():
            tf = res[file]
            normalised_tf = math.sqrt(tf)
            res[file] = normalised_tf

        return res



    # the following 3 functions all select the rows in which the term column equals the wanted term
    # after, the desired column value(s) are selected and returned (or an err value in case none exists)
    def get_df_idf_according_to_term(self, term):
        row = self.inverted_index.filter(col("term") == term).select(col("df")).collect()

        if len(row) == 0:
            return 0,-1
        else:
            df = row[0][0]
            return df, math.log(self.inverted_index.count() / df, math.e)

    def get_tfs_according_to_term(self, term):
        row = self.inverted_index.filter(col("term") == term).select(col("tfs")).collect()

        if len(row) == 0:
            return {}
        else:
            tfs = row[0][0]
            return tfs

    def get_df_tfs_according_to_term(self, term):
        row = self.inverted_index.filter(col("term") == term).select(col("df"), col("tfs")).collect()

        if len(row) == 0:
            return 0, {}
        else:
            df = row[0][0]
            tfs = row[0][1]
            return df, tfs



    # boolean model assumes only key words of {and, or, not}
    def get_top_ten_boolean(self, query):
        query = query.lower()

        res = []

        terms_tmp = {}
        query_terms = query.replace('(', '').replace(')', '').split()
        for term in query_terms:
            # ignore key-words
            if term in ['and', 'or', 'not']:
                continue

            # save for each file which terms it includes from the query
            tfs = self.get_tfs_according_to_term(term)
            for file in tfs:
                current = terms_tmp.get(file, [])
                current.append(term)
                terms_tmp[file] = current

        # get ~first 10 files that fulfill the boolean expression
        for file in terms_tmp.keys():
            if(len(res) >= 10):
                break

            # copy query string as we have to use replace multiple times sequentially
            file_query = (query + '.')[:-1]

            # terms in file -> True
            intersecting_terms = terms_tmp.get(file)
            for term in intersecting_terms:
                file_query = file_query.replace(term, 'True')

            # terms not in file -> False
            disjoint_terms = [term for term in query_terms if term not in intersecting_terms and term not in ['and', 'or', 'not']]
            for term in disjoint_terms:
                file_query = file_query.replace(term, 'False')

            # add if fulfills expression
            bool_eval = self.boolean_parser.boolExpr.parseString(file_query)[0]
            if bool(bool_eval):
                res.append((file, self.directory_path))

        # fill up results + in case only not key-word is used
        if len(res) < 10:
            files = self.corpus.select(col("filename")).collect()
            for file in files:
                if len(res) >= 10:
                    break

                if file[0] not in res and file[0] not in terms_tmp.keys():
                    res.append((file[0], self.directory_path))

        return res

    def get_top_ten_vector(self, query):
        query = query.lower()

        res = {}

        query = query.split()
        for term in query:
            df, tfs = self.get_df_tfs_according_to_term(term)
            df = 1 if df == 0 else df # to avoid errors
            idf = math.log(self.inverted_index.count() / df, math.e)
            # dot product = sum((tf * idf)(file) * (idf*tf)(query))
            for file in tfs:
                current = res.get(file, 0)
                # simplified weights of query terms as every term only appears once
                res[file] = current + (idf * idf * tfs.get(file, 0))

        top_ten = list(dict(sorted(res.items(), key=lambda item: item[1], reverse=True)).keys())[:10]
        return [(name, self.directory_path) for name in top_ten]

    def get_top_ten_cosine(self, query):
        query = query.lower()

        res = {}

        query = query.split()
        query_len = 0
        for term in query:
            df, tfs = self.get_df_tfs_according_to_term(term)
            df = 1 if df == 0 else df # to avoid errors
            idf = math.log(self.inverted_index.count() / df, math.e)
            query_len += (idf*idf)
            # also dot product = sum((tf * idf)(file) * (idf*tf)(query)) first
            for file in tfs:
                current = res.get(file, 0)
                # simplified weights of query terms as every term only appears once
                res[file] = current + (idf * idf * tfs.get(file, 0))
        # update dot products by normalising them -> divide by the product of the lengths of their vectors (= cosine similarity)
        res = {k: v/(self.corpus_file_lens.get(k, 1) * math.sqrt(query_len)) for k, v in res.items()}

        top_ten = list(dict(sorted(res.items(), key=lambda item: item[1], reverse=True)).keys())[:10]
        return [(name, self.directory_path) for name in top_ten]



    def run_transformations_on_corpus(self, html_bool):
        self.corpus = self.corpus.withColumn("filename", regexp_replace("filename", self.pattern_1, ""))
        self.corpus = self.corpus.withColumn("value", lower(col("value")))

        if html_bool:
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_3, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))

            # only to improve html parsing a bit more
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_4_extra, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_4_extra, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_5_extra, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_5_extra, " "))
            self.corpus = self.corpus.withColumn("value", regexp_replace("value", self.pattern_2, " "))

        self.corpus = RegexTokenizer(inputCol="value", outputCol="tokenized", pattern="\\s").transform(self.corpus)
        self.corpus = StopWordsRemover(inputCol="tokenized", outputCol="tokenized_reduced").transform(self.corpus)
        self.corpus = self.corpus.drop(col("value"))
        self.corpus = self.corpus.drop(col("tokenized"))

        return self.corpus