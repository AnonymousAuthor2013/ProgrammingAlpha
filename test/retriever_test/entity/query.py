# -*- UTF-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from entity.answer import Answer
from entity.post import Post
from entity.question import Question
from retriver import search_es
from util import tokenizer
from util.preprocessor import PreprocessPostContent


class Query:
    def __init__(self, title, body, tag_list, created_date):
        self.title = title
        self.body = body
        self.tag_list = tag_list
        self.created_date = created_date
        self.searched_post_list = []

    def searchFromES(self, size=200):
        search_result_list= search_es.search(self.title, size)
        # Post list
        post_obj_list = []
        for result in search_result_list:
            # Question
            question = Question(result['question']['Title'], result['question']['Body'], result['question']['CommentCount'], result['question']['Score'], result['question']['Tags'], result['question']['CreationDate'])
            # Answer list
            # body, created_date, score=0, comment_count=0
            answers = result['answers']
            answer_list = []
            for answer in answers:
                # body, created_date, score=0, comment_count=0)
                answer = Answer(answer['Body'],answer['CreationDate'],answer['Score'],answer['CommentCount'])
                answer_list.append(answer)

            # Add Post to post list
            post_obj_list.append(Post(question, answer_list))

        self.searched_post_list = post_obj_list

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        body = " ".join(body_para_list)
        return body

    def __calculate_a_title_relevance(self, question_obj):
        query_title_word_list = tokenizer.tokenize(self.title)
        if len(query_title_word_list) == 0:
            return 0

        question_title_word_list = tokenizer.tokenize(question_obj.title)
        # lower
        question_title_word_list = [w.lower() for w in question_title_word_list]
        query_title_word_list = [w.lower() for w in query_title_word_list]

        overlap = [value for value in query_title_word_list if value in question_title_word_list]
        ret = len(overlap) / len(query_title_word_list)

        return ret

    def calculate_title_relevance(self, post_list):
        for post in post_list:
            post.set_title_relevance(self.__calculate_a_title_relevance(post.question_obj))

    def __calculate_a_tag_relevance(self, question_obj):
        if len(self.tag_list) == 0:
            return 0

        overlap = [value for value in self.tag_list if value in question_obj.tag_list]
        ret = len(overlap) / len(self.tag_list)

        return ret

    def calculate_tag_relevance(self, post_list):
        for post in post_list:
            post.set_tag_relevance(self.__calculate_a_tag_relevance(post.question_obj))

    def calculate_tf_idf(self, post_list=None, type="question_body"):
        """
        计算TF_IDF值
        :param post_list:
        :param type: question_body or answer_body
        :return:
        """
        corpus = []
        # question body语料 每一行是每一个documents(首行是query body(处理后的))
        corpus.append(self.parse_body())
        if type == "question_body":
            for post in post_list:
                corpus.append(post.question_obj.parse_body())
        elif type == "answer_body":
            for post in post_list:
                corpus.append(post.concat_answer_body())
        else:
            raise NameError("Type Error")

        # print(corpus)
        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)  # 计算个词语出现的次数
        # print(vectorizer.get_feature_names()) # 获取词袋中所有文本关键词
        # print(X.toarray()) # 查看词频结果

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值
        tfidf_array = tfidf.toarray()
        row, col = tfidf_array.shape
        if type == "question_body":
            for i in range(1, row):
                sum = 0
                for j in range(col):
                    if tfidf_array[0, j] != 0:
                        sum += tfidf_array[i, j]

                post_list[i - 1].set_question_body_tfidf(sum)
        else:
            for i in range(1, row):
                sum = 0
                for j in range(col):
                    if tfidf_array[0, j] != 0:
                        sum += tfidf_array[i, j]

                post_list[i - 1].set_answer_body_tfidf(sum)

    def __calculate_a_score(self, post_obj, alpha):
        comment_count = post_obj.question_obj.comment_count
        vote_score = post_obj.question_obj.score
        for answer_obj in post_obj.answer_obj_list:
            comment_count += answer_obj.comment_count
            vote_score += answer_obj.score

        score = (1 - alpha) * comment_count + alpha * vote_score
        return score

    def calculate_score(self, post_list, alpha=0.8):
        for post in post_list:
            post.set_score(self.__calculate_a_score(post, alpha))

    def __get_body_code(self):
        code_snippet_list = PreprocessPostContent().get_single_code(self.body)
        single_code_list = []
        for code_snippet in code_snippet_list:
            code_list = code_snippet.split()
            if len(code_list) == 1:
                single_code_list.extend(code_list)

        return single_code_list

    def __calculate_a_code_relevance(self, post_obj):
        query_code_list = self.__get_body_code()
        if len(query_code_list) == 0:
            return 0

        post_code_list = []
        post_code_list.extend(post_obj.get_question_body_code())
        post_code_list.extend(post_obj.get_answer_body_code())

        overlap = [code for code in query_code_list if code in post_code_list]
        code_relevance = len(overlap) / len(query_code_list)

        return code_relevance

    def calculate_code_relevance(self, post_list):

        for post in post_list:
            code_relevance = self.__calculate_a_code_relevance(post)
            post.set_code_relevance(code_relevance)

    def __normalize(self, score_list):
        max_val = max(score_list)
        min_val = min(score_list)
        normalized_score_list = []
        for score in score_list:
            if max_val == min_val:
                normalized_score_list.append(0)
            else:
                normalized_score_list.append((score-min_val)/(max_val-min_val))

        return normalized_score_list


    def normalized_post_score(self):
        title_relevance_list = []
        tag_relevance_list = []
        question_tfidf_list = []
        answer_tfidf_list = []
        score_list = []
        code_relevance_list = []

        for searched_post in self.searched_post_list:
            title_relevance_list.append(searched_post.title_relevance)
            tag_relevance_list.append(searched_post.tag_relevance)
            question_tfidf_list.append(searched_post.question_tfidf)
            answer_tfidf_list.append(searched_post.answer_tfidf)
            score_list.append(searched_post.score)
            code_relevance_list.append(searched_post.code_relevance)

        title_relevance_list = self.__normalize(title_relevance_list)
        tag_relevance_list = self.__normalize(tag_relevance_list)
        question_tfidf_list = self.__normalize(question_tfidf_list)
        answer_tfidf_list = self.__normalize(answer_tfidf_list)
        score_list = self.__normalize(score_list)
        code_relevance_list = self.__normalize(code_relevance_list)

        for i in range(len(self.searched_post_list)):
            self.searched_post_list[i].title_relevance = title_relevance_list[i]
            self.searched_post_list[i].tag_relevance = tag_relevance_list[i]
            self.searched_post_list[i].question_tfidf = question_tfidf_list[i]
            self.searched_post_list[i].answer_tfidf = answer_tfidf_list[i]
            self.searched_post_list[i].score = score_list[i]
            self.searched_post_list[i].code_relevance = code_relevance_list[i]


if __name__ == '__main__':
    tag_list1 = ['Java', '<java>', 'println']
    tag_list2 = ['<c++>', '<java>', '<python>', 'pycharm']
    tag_list3 = ['<c++>', '<JAVA>', '<python>', 'pycharm']

    query = Query("How to use println in java", "Please show me how to use <code>println()<code> in java", tag_list1, "2019-5-16")
    question1 = Question("your title question",
                         "<p>question1 para1 content 1 <code>test_function()</code><code>print()</code>.</p><p>question1 para2 content.</p><p>question1 para3 content.</p>",
                         2, 8, tag_list1, "2019-5-16")
    question2 = Question("your title2 question title2 title",
                         "<p>question2 para1 content. have some question1 info</p><p>question2 para2 content.<code>question2_function</code></p><p>question2 para3 content.</p>",
                         2, 10, tag_list2, "2019-5-16")
    question3 = Question("your title question title title",
                         "<p>question3 para1 content.</p><p>question3 para2 content.</p><p>question3 para3 content.<code>question3_para3_function</code></p>",
                         21, 100, tag_list3, "2019-5-16")

    answer1 = Answer("<p>question1 para1 is good</p>", "2019-5-16", score=10, comment_count=1)
    answer2 = Answer("<p>question1 and question2 is good</p><code>answer_code</code>", "2019-5-16", score=2, comment_count=10)
    answer3 = Answer("<p>question 3 is good</p>", "2019-5-16", score=5, comment_count=11)
    answer_list = [answer1, answer2, answer3]
    p1 = Post(question1, answer_list)
    p2 = Post(question2, answer_list)
    p3 = Post(question3, answer_list)
    p = Post(question1, answer_list)
    post_list = [p1, p2, p3]

    # query.calculate_tf_idf(post_list=post_list)
    # for post in post_list:
    #     print(post.tfidf)
    # print(p.concat_answer_body())
    query.calculate_tf_idf(post_list=post_list, type="question_body")
    query.calculate_tf_idf(post_list=post_list, type="answer_body")
    query.calculate_score(post_list)
    query.calculate_code_relevance(post_list)
    query.searchFromES()
    post_list = query.searched_post_list
    query.calculate_title_relevance(post_list)
    query.normalized_post_score()
    for post in query.searched_post_list:
        # print(post.question_tfidf)
        # print(post.answer_tfidf)
        # print(post.title_relevance)
        # print(post.tag_relevance)
        print(post.title_relevance)

    query.searchFromES()
    post_list = query.searched_post_list
