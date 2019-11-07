# Solution: DGA detection

Taking into consideration the context, 
*  in practice, if we have some data with the DGAs generated domains
   properly labelled, we can resolve this task using supervised
   learning;
*  otherwise, we can only use unsupervised learning. 

No matter the problem is supervised or unsupervised, it is a typical
classification (two-class or multi-class) problem. We quickly seek
answers for two questions from the literature:
* what are the candidate features
* what are the candidate algorithms

Since we only have domains, according to
[yu2017inline][http://faculty.washington.edu/mdecock/papers/byu2017a.pdf]
and
[bilge2011exposure][https://sites.cs.ucsb.edu/~chris/research/doc/ndss11_exposure.pdf],
the features extracted are summarised as follows.
* x1: length of domain 
* x2: length of numerical chars of domain
* x3: length of symbol chars of domain
* x4: length of vowels of domain
* x5: count of unique chars of domain
* x6: normalised entropy of a domain
* x7: Gini index of a domain
* x8: classification error of a domain

As suggested by
[Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs][2],
we have chosen Naive Bayes (NB), Logistic Regression (LR) and Support
Vector Machine (SVM) as candidate algorithms for classification. We
create an automatic process to select best algorithm/classifier and
feature set by running 3-fold cross validations over any subsets of the
candidate features and the 12 pre-constructed classifiers (as shown in
params.py), where the metric is highest classification accuracy. Since
the code is implemented in a multi-processing manner, this brute-forced
process does not take too much time (~1 min). The cross validation
process suggests that one LR classifier is able to achieve a
classification accuracy of 91\%, with the feature set
[x1, x2, x3, x4, x5, x6, x7]. We then train a classifier using the best
setting and apply it to predict any other domains (as 'legit' or 'dga').

However, we assume that it is not easy to obtain labelled data in
practical applications. Thus, we also realise an unsupervised detector.
We train a 1st order Markov chain with only the legit domains (which is
consistent with the assumption). Then, we are enabled to compute
sequence probability of 3-grams of a domain as a list of characters. For
example, given 'google', we compute sequence probabilities for 'goo',
'oog', 'ogl', 'gle'. We average these probabilities to generate a score
for each domain. We choose a low percentile of such scores obtained from
legit domains as threshold, say 1\%. If a domain's score is below that
threshold, it will be labelled as 'dga', otherwise 'legit'. We test with
the data, demonstrating that in theory we can achieve 91\% detection
accuracy with around 1\% false positive rate.