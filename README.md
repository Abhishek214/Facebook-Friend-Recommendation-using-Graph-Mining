# Instagram-Friend-Recommendation-using-Graph-Mining
👫 In this challenge we are given a directed social graph, and we have to predict missing links to recommend users (Link Prediction in graph).

## Problem statement: 

Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph)

### About Dataset:
Our dataset is directed graph data
We have approx. 1.86M nodes and 9.43M edges.
Data was obtained from kaggle. You can get data from here https://www.kaggle.com/c/FacebookRecruiting
We have provided only connected nodes. i.e. 9.43M edges. But for each user among n user's, there is n-1 edges. So, for n nodes total possible edges are of 10^12 order.


### Performance metric:
* Both precision and recall is important so F1 score is good choice
* Confusion matrix


### Training Dataset preperation:
* If we consider y= 1 , if edge is present in between two nodes.
* We will assume y = 0 , if no edge is present.
* Generated Bad links from graph which are not in graph and whose shortest path is greater than 2

### Featurization:
Featurization is the most important part of this case study. Below is the list of extracted features

- Similarity measures
- Jaccard Distance
- Cosine distance
- Ranking Measure
- Page Ranking (https://en.wikipedia.org/wiki/PageRank)
- Graph Features
- Shortest Path
- Checking for same community
- Adamic/Adar Index
- Is following back
- Katz Centrality
- Hits Score
- num followers
- num followees
