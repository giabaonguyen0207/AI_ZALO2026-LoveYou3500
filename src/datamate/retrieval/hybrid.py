"""CPU lexical ranking and reciprocal-rank fusion; no additional dependencies."""
from collections import Counter
import math
import re
from ..agent.language import folded


def rank_candidates(chunks, queries, dense_scores, *, hybrid, limit):
    dense = sorted(range(len(chunks)),key=lambda i:(-float(dense_scores[i]),chunks[i].metadata['chunk_id']))
    if not hybrid or not chunks:
        return dense[:limit]
    tokenize = lambda text: re.findall(r'\w+',folded(text))
    documents = [Counter(tokenize(c.text)) for c in chunks]
    query = set(tokenize(' '.join(queries)))
    frequencies = Counter(term for doc in documents for term in query if term in doc)
    lengths = [sum(doc.values()) for doc in documents]
    mean = sum(lengths)/max(1,len(lengths)) or 1
    scores=[]
    for doc,length in zip(documents,lengths):
        value=0.0
        for term in query & doc.keys():
            idf=math.log(1+(len(chunks)-frequencies[term]+0.5)/(frequencies[term]+0.5))
            tf=doc[term]
            value += idf*(tf*2.2)/(tf+1.2*(0.25+0.75*length/mean))
        scores.append(value)
    lexical = sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],chunks[i].metadata['chunk_id']))
    fused={i:1/(60+r) for r,i in enumerate(dense,1)}
    for r,i in enumerate(lexical,1):
        fused[i]+=1/(60+r)
    return sorted(dense,key=lambda i:(-fused[i],chunks[i].metadata['chunk_id']))[:limit]
