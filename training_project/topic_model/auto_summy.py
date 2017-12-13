#!/usr/bin/env python3
#-*-coding:utf-8-*-
# __all__=""
# __datetime__=""
# __purpose__="自动摘要根据tf/idf以及余弦相似性"
#!/user/bin/python
# coding:utf-8
import nltk
import numpy
import jieba
import codecs

class Topic_Summy:

    N=200#单词数量
    CLUSTER_THRESHOLD=5#单词间的距离
    TOP_SENTENCES=3#返回的top n句子

    def __init__(self,text):
        self.texts = text
        import os
        path = os.path.join(os.path.dirname(os.getcwd()),'stopwords.txt')
        self.path = path
    #分句
    def sent_tokenizer(self):
        start=0
        i=0#每个字符的位置
        sentences=[]
        punt_list='.!?。！？' #',.!?:;~，。！？：；～'.decode('utf8')
        for text in self.texts:
            if text in punt_list and token not in punt_list: #检查标点符号下一个字符是否还是标点
                sentences.append(self.texts[start:i+1])#当前标点符号位置
                start=i+1#start标记到下一句的开头
                i+=1
            else:
                i+=1#若不是标点符号，则字符位置继续前移
                token=list(self.texts[start:i+2]).pop()#取下一个字符
        if start<len(self.texts):
            sentences.append(self.texts[start:])#这是为了处理文本末尾没有标点符号的情况
        return sentences

    #停用词
    def load_stopwordslist(self):
        print('load stopwords...')
        stoplist=[line.strip() for line in codecs.open(self.path,'r',encoding='utf8').readlines()]
        stopwrods={}.fromkeys(stoplist)
        return stopwrods

    #摘要
    def summarize(self):
        stopwords=self.load_stopwordslist()
        sentences=self.sent_tokenizer()
        words=[w for sentence in sentences for w in jieba.cut(sentence) if w not in stopwords if len(w)>1 and w!='\t']
        wordfre=nltk.FreqDist(words)
        topn_words=[w[0] for w in sorted(wordfre.items(),key=lambda d:d[1],reverse=True)][:self.N]
        scored_sentences=self._score_sentences(sentences,topn_words)
        #approach 1,利用均值和标准差过滤非重要句子
        avg=numpy.mean([s[1] for s in scored_sentences])#均值
        std=numpy.std([s[1] for s in scored_sentences])#标准差
        mean_scored=[(sent_idx,score) for (sent_idx,score) in scored_sentences if score>(avg+0.5*std)]
        #approach 2，返回top n句子
        top_n_scored=sorted(scored_sentences,key=lambda s:s[1])[-self.TOP_SENTENCES:]
        top_n_scored=sorted(top_n_scored,key=lambda s:s[0])
        return dict(top_n_summary=[sentences[idx] for (idx,score) in top_n_scored],mean_scored_summary=[sentences[idx] for (idx,score) in mean_scored])

     #句子得分
    def _score_sentences(self,sentences,topn_words):
        scores=[]
        sentence_idx=-1
        for s in [list(jieba.cut(s)) for s in sentences]:
            sentence_idx+=1
            word_idx=[]
            for w in topn_words:
                try:
                    word_idx.append(s.index(w))#关键词出现在该句子中的索引位置
                except ValueError:#w不在句子中
                    pass
            word_idx.sort()
            if len(word_idx)==0:
                continue
            #对于两个连续的单词，利用单词位置索引，通过距离阀值计算族
            clusters=[]
            cluster=[word_idx[0]]
            i=1
            while i<len(word_idx):
                if word_idx[i]-word_idx[i-1]<self.CLUSTER_THRESHOLD:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster=[word_idx[i]]
                i+=1
            clusters.append(cluster)
            #对每个族打分，每个族类的最大分数是对句子的打分
            max_cluster_score=0
            for c in clusters:
                significant_words_in_cluster=len(c)
                total_words_in_cluster=c[-1]-c[0]+1
                score=1.0*significant_words_in_cluster*significant_words_in_cluster/total_words_in_cluster
                if score>max_cluster_score:
                    max_cluster_score=score
            scores.append((sentence_idx,max_cluster_score))
        return scores



if __name__=='__main__':
    a = """"
        周二，美国又有一名女演员站出来称自己若干年前遭到了性骚扰。骚扰她的并不是某大名鼎鼎的制作人，而是美国前总统、现年93岁的老布什(George H.W. Bush)。这位名叫林德(Heather Lind)的女演员周二在社交媒体Instagram上发文说，四年前，在她和老布什拍摄合影时，后者对她进行了性骚扰。几天前，美国在世的五位前总统集体现身一场为飓风灾区筹款的音乐会，并合影留言。之后，林德就此在Instagram发贴说，她看到奥巴马总统和老布什握手的照片后感到不安，“我之所以觉得不舒服，是因为我意识到这些前总统们为他们的贡献而受到了尊重，我为照片中的许多人感到骄傲和敬佩”。她接下来提到了四年前与老布什的一段经历，“然而，四年前我有机会在宣传参演的一部历史电视剧时见到了老布什，在我准备和他拍摄合影的时候，他性骚扰了我。他没有握我的手，而是坐在轮椅上从背后摸我，他夫人芭芭拉就站在旁边。他还给我讲了一个黄色笑话。”“然后，在合影的过程中，他又再次摸了我。芭芭拉当时翻了翻白眼，似乎是在说‘别再这样了’” ，林德这样写道。她还称，在活动过后，老布什的安全人员告诉她，她不应该站在这位前总统的旁边。林德的这个Instagram帖子目前已经被删除。对于林德的这一指控，老布什的发言人麦格拉思(Jim McGrath)在一份声明中表示“道歉”，声明称，“无论在任何情况下，布什总统从来都不会有意让任何人难受。如果他试图表达幽默的行为冒犯了林德女士，他在这里对她表示最诚挚的道歉”。最近，好莱坞著名制片人韦恩斯坦骚扰大批演艺界女性、模特的性丑闻被曝光，之后不断有人站出来披露自己受其骚扰的经历，社交媒体上也出现了相关热门话题。林德的Instagram帖子就使用了此类话题标签#metoo(我也是)。她在帖子中还详细描述了事情的前后经过以及自己的一些感受，她写道：“我们被指示叫他总统先生。对我而言，总统的权力是让他或她带来积极的改变，真正帮助人们，并作为我们民主制度的一个象征…他在我身上使用这种权力的时候，实际上就是放弃了这个权力。从他周围人的评论来看，在我之前有无数其他女性也有这样的遭遇。”老布什是美国第41任总统，他的儿子小布什则是第43任总统。老布什患有血管性帕金森综合症，近年来只能坐在轮椅上出行。林德说，她感谢奥巴马总统对老布什表现出的尊重的姿态，“但我不尊敬他”。

"""
    a = Topic_Summy(a)
    dict1 = a.summarize()


    print('-----------approach 1-------------')
    for sent in dict1['top_n_summary']:
        print(sent)
    print('-----------approach 2-------------')
    for sent in dict1['mean_scored_summary']:
        print(sent)