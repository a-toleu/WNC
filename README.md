# Clustering-based topic modeling
Word network clustering (WNC), is a clustering-based approach to topic modeling via analyzing word networks based on the adaptation of a community detection algorithm. Word networks are constructed with different word representations, and two types of topic assignments are introduced. Topic coherence score and the document clustering results are reported for topic model evaluation. Experimental results showed that it achieved comparable results with the current best. It also showed that the proposed approach produced a higher performance as the number of most relevant words gets larger in Cv coherence score.

Related paper:
G. Tolegen, A. Toleu, R. Mussabayev and A. Krassovitskiy, "A Clustering-based Approach for Topic Modeling via Word Network Analysis," 2022 7th International Conference on Computer Science and Engineering (UBMK), Diyarbakir, Turkey, 2022, pp. 192-197, doi: 10.1109/UBMK55850.2022.9919530.

## Concepts
Clustering-based topic model decomposes the large search space into two sub-spaces: |T| × |W| and |D| × |T|. Each of the sub-spaces cannot be very large in practice since |T| ≪ |D| and |T| ≪ |W|. First of all, the hidden topics of documents are found with a clustering algorithm in a subspace |T | × |W |, then the topic assignments in another small sub-space |T | × |D|. To uncover the thematic structure of a collection of documents, we proposed an approach based on adapting a community detection algorithm on a word network where vertices are words and edges are the distances between words. Several word representations are utilized for word network construction, and their effectiveness was compared through a set of experiments. To better assign the document with topics, two topic assignment approaches were applied and compared.

## usage
A demo of the jupyter notebook is available in this repository.

<br />
Hyperparameters of the approach
| Blocks                                        | _Parameters_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vocabulary extraction and vectorization block | ·       _optimized = {0,1}_, default = 0, it uses simple frequency-based approach to extract vocabulary and calculate word representation with co-occurrence, when optimized = 1, vocabulary will be optimized with tfidf value, correspondingly, the representation will be calculated with this feature.                                                                                                                                                                                                                                                          |
| Kmean-based topic modeling                    | ·       _initA={_ kmeans++, random_}_,  it is parameter of initializing the kmeans algorithm, default=”random”, if initA=”kmeans++”, the initial centroid will be initialized with more advanced fashion. <br />·       _njobs = {0, n},_ default = 0, it defines to use how many processors to parallelize the computations.<br /> ·       _max_iter_, it defines a maximum number of iterations, default = 100.                                                                                                                                                              |
| WNC approach                                  | ·       _resolution_, default = 0.85, it determines the size of the clusters that are identified. Lower resolution values result in the recovery of smaller clusters, thus increasing their number. In contrast, higher resolution values lead to the identification of larger clusters that encompass a greater number of data points. <br />·       _nodeOpti,_ default = 50. It determines that how many neighboring nodes of each node will be used. <br /> ·       _ranking={0,1},_ default = 1, it ranks the internal words inside each topic by calculating its degree. |
