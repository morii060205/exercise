#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文维基百科 Word2Vec 词向量训练完整流程

实验目标:
1. 在中文维基百科语料上训练Word2Vec模型
2. 探索"国王-男人+女人≈王后"类比关系
3. 对比不同window_size对词向量的影响

环境要求:
- Python 3.8+
- gensim, jieba, opencc-python-reimplemented, tqdm, numpy

作者: 人工智能实训
"""

import os
import re
import logging
import multiprocessing
from tqdm import tqdm

# ============================================================
# 第一步：安装依赖
# ============================================================
"""
在命令行中执行:

pip install gensim jieba opencc-python-reimplemented tqdm numpy
pip install wget  # 用于下载

# 如果使用conda:
conda install -c conda-forge gensim jieba tqdm numpy
pip install opencc-python-reimplemented
"""

# ============================================================
# 第二步：下载中文维基百科数据
# ============================================================
"""
方法一：手动下载（推荐）
1. 访问: https://dumps.wikimedia.org/zhwiki/latest/
2. 下载: zhwiki-latest-pages-articles.xml.bz2 (约2.5GB)
3. 放到项目目录下

方法二：使用wget下载
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
"""

def download_wiki_dump():
    """下载维基百科数据（可选，建议手动下载）"""
    import urllib.request
    
    url = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
    filename = "zhwiki-latest-pages-articles.xml.bz2"
    
    if os.path.exists(filename):
        print(f"文件已存在: {filename}")
        return filename
    
    print(f"开始下载: {url}")
    print("文件较大(约2.5GB)，请耐心等待...")
    
    urllib.request.urlretrieve(url, filename)
    print(f"下载完成: {filename}")
    return filename


# ============================================================
# 第三步：解析维基百科XML并提取文本
# ============================================================
from gensim.corpora import WikiCorpus

def extract_wiki_text(input_file, output_file):
    """
    从维基百科XML dump中提取纯文本
    
    Args:
        input_file: 维基百科bz2压缩文件路径
        output_file: 输出的文本文件路径
    """
    print("=" * 60)
    print("Step 1: 提取维基百科文本")
    print("=" * 60)
    
    if os.path.exists(output_file):
        print(f"输出文件已存在: {output_file}")
        return
    
    # 使用gensim的WikiCorpus解析XML
    wiki = WikiCorpus(input_file, dictionary={})
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in tqdm(wiki.get_texts(), desc="提取文章"):
            # text是词列表，用空格连接
            f.write(' '.join(text) + '\n')
            count += 1
            
            if count % 10000 == 0:
                print(f"已处理 {count} 篇文章")
    
    print(f"提取完成，共 {count} 篇文章")
    print(f"保存至: {output_file}")


# ============================================================
# 第四步：繁简转换 + 中文分词
# ============================================================
import jieba
import opencc

def preprocess_chinese_text(input_file, output_file):
    """
    中文文本预处理：繁简转换 + 分词
    
    Args:
        input_file: 提取的原始文本
        output_file: 分词后的文本
    """
    print("=" * 60)
    print("Step 2: 繁简转换 + 中文分词")
    print("=" * 60)
    
    if os.path.exists(output_file):
        print(f"输出文件已存在: {output_file}")
        return
    
    # 繁体转简体转换器
    converter = opencc.OpenCC('t2s')  # Traditional to Simplified
    
    # 加载自定义词典（可选，提高分词准确性）
    # jieba.load_userdict("custom_dict.txt")
    
    # 统计
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, total=total_lines, desc="分词处理"):
            line = line.strip()
            if not line:
                continue
            
            # 1. 繁体转简体
            line = converter.convert(line)
            
            # 2. 清洗文本（去除特殊字符，保留中文、英文、数字）
            line = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', line)
            
            # 3. jieba分词
            words = jieba.cut(line, cut_all=False)
            
            # 4. 过滤：去除单字、停用词、纯数字
            words = [w.strip() for w in words 
                    if len(w.strip()) > 1 
                    and not w.strip().isdigit()]
            
            if words:
                fout.write(' '.join(words) + '\n')
    
    print(f"分词完成，保存至: {output_file}")


# 中文停用词表（常用）
STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '什么', '他', '她', '它', '我们', '你们', '他们', '这个',
    '那个', '之', '与', '为', '以', '及', '等', '或', '但', '如果', '因为', '所以',
    '可以', '这样', '那样', '什么', '怎么', '如何', '多少', '几', '哪', '哪里',
    '这里', '那里', '年', '月', '日', '时', '分', '秒', '个', '次', '种', '位'
])

def preprocess_with_stopwords(input_file, output_file):
    """带停用词过滤的预处理"""
    print("=" * 60)
    print("Step 2: 繁简转换 + 分词 + 停用词过滤")
    print("=" * 60)
    
    converter = opencc.OpenCC('t2s')
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, total=total_lines, desc="分词处理"):
            line = line.strip()
            if not line:
                continue
            
            line = converter.convert(line)
            line = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', line)
            words = jieba.cut(line, cut_all=False)
            
            # 过滤停用词
            words = [w.strip() for w in words 
                    if len(w.strip()) > 1 
                    and not w.strip().isdigit()
                    and w.strip() not in STOPWORDS]
            
            if words:
                fout.write(' '.join(words) + '\n')
    
    print(f"处理完成: {output_file}")


# ============================================================
# 第五步：训练Word2Vec模型
# ============================================================
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def train_word2vec(corpus_file, model_path, 
                   vector_size=300, 
                   window=5, 
                   min_count=5,
                   workers=None,
                   sg=1,
                   epochs=5):
    """
    训练Word2Vec模型
    
    Args:
        corpus_file: 分词后的语料文件
        model_path: 模型保存路径
        vector_size: 词向量维度 (默认300)
        window: 上下文窗口大小 (默认5)
        min_count: 最小词频 (默认5)
        workers: 并行线程数 (默认CPU核心数)
        sg: 1=Skip-gram, 0=CBOW (默认Skip-gram)
        epochs: 训练轮数 (默认5)
    
    Returns:
        训练好的Word2Vec模型
    """
    print("=" * 60)
    print("Step 3: 训练Word2Vec模型")
    print("=" * 60)
    
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    print(f"参数配置:")
    print(f"  - 词向量维度: {vector_size}")
    print(f"  - 窗口大小: {window}")
    print(f"  - 最小词频: {min_count}")
    print(f"  - 训练算法: {'Skip-gram' if sg else 'CBOW'}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 并行线程: {workers}")
    
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO
    )
    
    # 使用LineSentence迭代器（内存友好）
    sentences = LineSentence(corpus_file)
    
    print("\n开始训练...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        seed=42
    )
    
    # 保存模型
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 保存词向量（可用于其他工具）
    vector_path = model_path.replace('.model', '.vectors')
    model.wv.save_word2vec_format(vector_path, binary=False)
    print(f"词向量已保存: {vector_path}")
    
    # 打印统计信息
    print(f"\n模型统计:")
    print(f"  - 词汇量: {len(model.wv)}")
    print(f"  - 向量维度: {model.wv.vector_size}")
    
    return model


# ============================================================
# 第六步：模型评估与类比实验
# ============================================================
import numpy as np

def evaluate_model(model):
    """
    评估Word2Vec模型
    
    包含:
    1. 相似词查询
    2. 词类比测试
    3. 词向量计算
    """
    print("=" * 60)
    print("Step 4: 模型评估")
    print("=" * 60)
    
    wv = model.wv
    
    # ----- 1. 相似词查询 -----
    print("\n【1. 相似词查询】")
    test_words = ['国王', '中国', '北京', '科学', '计算机', '人工智能']
    
    for word in test_words:
        if word in wv:
            similar = wv.most_similar(word, topn=5)
            print(f"\n'{word}' 的相似词:")
            for w, score in similar:
                print(f"    {w}: {score:.4f}")
        else:
            print(f"\n'{word}' 不在词汇表中")
    
    # ----- 2. 词类比测试 -----
    print("\n" + "=" * 40)
    print("【2. 词类比测试】")
    print("=" * 40)
    
    # 经典类比: 国王 - 男人 + 女人 = ?
    analogies = [
        ('国王', '男人', '女人', '王后'),      # 性别类比
        ('中国', '北京', '日本', '东京'),      # 首都类比
        ('父亲', '男人', '女人', '母亲'),      # 家庭关系
        ('好', '更好', '坏', '更坏'),          # 程度类比
        ('中国', '中文', '日本', '日文'),      # 语言类比
        ('公司', '员工', '学校', '学生'),      # 组织成员
    ]
    
    for a, b, c, expected in analogies:
        try:
            # 计算: a - b + c = ?
            result = wv.most_similar(positive=[a, c], negative=[b], topn=5)
            
            print(f"\n{a} - {b} + {c} = ?  (期望: {expected})")
            print("  实际结果:")
            for word, score in result:
                marker = " ✓" if word == expected else ""
                print(f"    {word}: {score:.4f}{marker}")
                
        except KeyError as e:
            print(f"\n{a} - {b} + {c} = ?  (词不在词汇表: {e})")
    
    # ----- 3. 词向量运算 -----
    print("\n" + "=" * 40)
    print("【3. 词向量运算演示】")
    print("=" * 40)
    
    if '国王' in wv and '男人' in wv and '女人' in wv:
        # 获取词向量
        vec_king = wv['国王']
        vec_man = wv['男人']
        vec_woman = wv['女人']
        
        # 计算: 国王 - 男人 + 女人
        vec_result = vec_king - vec_man + vec_woman
        
        # 归一化
        vec_result = vec_result / np.linalg.norm(vec_result)
        
        # 找最相似的词
        similar = wv.similar_by_vector(vec_result, topn=10)
        print("\n国王 - 男人 + 女人 的计算结果:")
        for word, score in similar:
            print(f"  {word}: {score:.4f}")
    
    # ----- 4. 词对相似度 -----
    print("\n" + "=" * 40)
    print("【4. 词对相似度】")
    print("=" * 40)
    
    word_pairs = [
        ('国王', '王后'),
        ('男人', '女人'),
        ('中国', '美国'),
        ('北京', '上海'),
        ('计算机', '科学'),
    ]
    
    for w1, w2 in word_pairs:
        if w1 in wv and w2 in wv:
            sim = wv.similarity(w1, w2)
            print(f"  sim('{w1}', '{w2}') = {sim:.4f}")
        else:
            print(f"  '{w1}' 或 '{w2}' 不在词汇表中")


# ============================================================
# 第七步：窗口大小对比实验
# ============================================================
def window_size_experiment(corpus_file, output_dir='./models'):
    """
    对比不同window_size的效果
    
    较小窗口(3-5): 更能捕捉句法关系（词性、语法结构）
    较大窗口(8-15): 更能捕捉语义/主题相关性
    """
    print("=" * 60)
    print("Step 5: 窗口大小对比实验")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    window_sizes = [3, 5, 8, 10, 15]
    models = {}
    
    for window in window_sizes:
        print(f"\n{'='*40}")
        print(f"训练 window={window} 的模型")
        print(f"{'='*40}")
        
        model_path = os.path.join(output_dir, f'word2vec_w{window}.model')
        
        model = Word2Vec(
            sentences=LineSentence(corpus_file),
            vector_size=300,
            window=window,
            min_count=5,
            workers=multiprocessing.cpu_count(),
            sg=1,
            epochs=5,
            seed=42
        )
        
        model.save(model_path)
        models[window] = model
        print(f"模型已保存: {model_path}")
    
    # 对比实验
    print("\n" + "=" * 60)
    print("对比实验结果")
    print("=" * 60)
    
    test_word = '科学'
    print(f"\n查询 '{test_word}' 的相似词:")
    
    for window, model in models.items():
        if test_word in model.wv:
            similar = model.wv.most_similar(test_word, topn=5)
            print(f"\nwindow={window}:")
            for w, s in similar:
                print(f"  {w}: {s:.4f}")
    
    # 类比测试对比
    print("\n\n类比测试: 国王 - 男人 + 女人 = ?")
    for window, model in models.items():
        try:
            result = model.wv.most_similar(
                positive=['国王', '女人'], 
                negative=['男人'], 
                topn=3
            )
            print(f"\nwindow={window}:")
            for w, s in result:
                print(f"  {w}: {s:.4f}")
        except KeyError:
            print(f"\nwindow={window}: 词不在词汇表中")
    
    return models


# ============================================================
# 第八步：分析中文类比效果不佳的原因
# ============================================================
def analyze_chinese_analogy_issues(model):
    """
    分析中文词类比效果不如英文的原因
    """
    print("=" * 60)
    print("中文词类比问题分析")
    print("=" * 60)
    
    wv = model.wv
    
    print("""
【问题分析】

1. 词汇多义性
   - 中文的"王"既可指国王,也可指姓氏
   - "后"既可指王后,也可指"之后"、"后面"
   - 英文的king/queen语义相对单一
""")
    
    # 查看"王"和"后"的相似词
    for word in ['王', '后', '国王', '王后']:
        if word in wv:
            print(f"\n'{word}' 的相似词:")
            similar = wv.most_similar(word, topn=8)
            for w, s in similar:
                print(f"  {w}: {s:.4f}")
    
    print("""
2. 分词粒度问题
   - "国王"是否应该拆分为"国"+"王"？
   - 不同分词结果导致词向量不同
   
3. 语料差异
   - 中文维基百科的行文风格
   - 历史人物称谓习惯（皇帝、帝王、君主 vs 国王）
   
4. 建议改进方案
   - 使用更大的语料库
   - 尝试子词模型（FastText）
   - 针对性构建类比测试集
""")


# ============================================================
# 主函数：完整流程
# ============================================================
def main():
    """完整训练流程"""
    
    # 文件路径配置
    WIKI_DUMP = "zhwiki-latest-pages-articles.xml.bz2"  # 原始下载文件
    WIKI_TEXT = "zhwiki_text.txt"                        # 提取的文本
    WIKI_SEG = "zhwiki_seg.txt"                          # 分词后的文本
    MODEL_PATH = "word2vec_zh.model"                     # 模型保存路径
    
    print("=" * 60)
    print("中文维基百科 Word2Vec 训练流程")
    print("=" * 60)
    
    # 检查维基百科dump文件
    if not os.path.exists(WIKI_DUMP):
        print(f"\n错误: 找不到维基百科数据文件 {WIKI_DUMP}")
        print("请先下载: https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2")
        print("或运行: download_wiki_dump()")
        return
    
    # Step 1: 提取文本
    extract_wiki_text(WIKI_DUMP, WIKI_TEXT)
    
    # Step 2: 分词预处理
    preprocess_with_stopwords(WIKI_TEXT, WIKI_SEG)
    
    # Step 3: 训练模型
    model = train_word2vec(
        corpus_file=WIKI_SEG,
        model_path=MODEL_PATH,
        vector_size=300,
        window=5,
        min_count=5,
        sg=1,
        epochs=5
    )
    
    # Step 4: 评估模型
    evaluate_model(model)
    
    # Step 5: 分析类比问题
    analyze_chinese_analogy_issues(model)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


# ============================================================
# 快速测试：使用小规模语料
# ============================================================
def quick_test():
    """使用小规模语料快速测试流程"""
    
    print("=" * 60)
    print("快速测试模式（小规模语料）")
    print("=" * 60)
    
    # 创建测试语料
    test_corpus = """
    中国的首都是北京 北京是一个历史悠久的城市
    日本的首都是东京 东京是亚洲最大的城市之一
    国王统治着王国 王后是国王的妻子
    皇帝是中国古代的最高统治者 皇后是皇帝的妻子
    父亲是男人 母亲是女人 他们是一家人
    计算机科学是一门重要的学科 人工智能是计算机科学的分支
    深度学习是机器学习的一个方向 神经网络是深度学习的基础
    北京大学是中国著名的高等学府 清华大学也是中国顶尖的大学
    """
    
    # 分词
    sentences = []
    for line in test_corpus.strip().split('\n'):
        words = list(jieba.cut(line.strip()))
        words = [w for w in words if len(w) > 1 and w.strip()]
        if words:
            sentences.append(words)
    
    print(f"测试语料: {len(sentences)} 个句子")
    
    # 训练小模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=3,
        min_count=1,
        workers=4,
        sg=1,
        epochs=100,
        seed=42
    )
    
    print(f"词汇量: {len(model.wv)}")
    
    # 测试
    print("\n相似词测试:")
    for word in ['北京', '国王', '计算机']:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=3)
            print(f"  '{word}': {similar}")
    
    print("\n类比测试:")
    try:
        result = model.wv.most_similar(
            positive=['北京', '日本'], 
            negative=['中国'], 
            topn=3
        )
        print(f"  北京 - 中国 + 日本 = {result}")
    except:
        print("  类比测试失败（词汇量不足）")


# ============================================================
# 加载已训练模型
# ============================================================
def load_and_use_model(model_path):
    """加载已训练的模型"""
    
    print(f"加载模型: {model_path}")
    model = Word2Vec.load(model_path)
    
    print(f"词汇量: {len(model.wv)}")
    print(f"向量维度: {model.wv.vector_size}")
    
    return model


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 快速测试
        quick_test()
    else:
        # 完整流程
        main()
