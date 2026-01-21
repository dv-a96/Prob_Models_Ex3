import numpy as np
import pandas as pd

dev_set_path = 'develop.txt'
TRESHOLD=1e-4
# pre process data
def development_set_preprocessing(dev_set_path):
    # given the deveploment set path, compute the total number of events in the development set |S| (include repetition)
    # clean rear words (with less than 4 appearances in the development set)
    # return a list of lissts - all (non rear) events in each article in development set
    dict_word_count = {}
    try:
        with open(dev_set_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # iterate every 2 lines to skip headers
            for i in range(2, len(lines), 4):
                words = lines[i].strip().split()
                for word in words:
                    if word in dict_word_count:
                        dict_word_count[word] += 1
                    else:
                        dict_word_count[word] = 1
            print(f"Total unique words before filtering: {len(dict_word_count)}")
            # filter out rear words
            unique_words = sorted([word for word, count in dict_word_count.items() if count >= 4])
            # create a mapping from word to index for fast lookup
            word_to_id = {word: i for i, word in enumerate(unique_words)}
             # second pass to filter out rare words
            all_articles_indexed = []
            for i in range(2, len(lines), 4):
                words = lines[i].strip().split()
                indexed_article = [word_to_id[w] for w in words if w in word_to_id]
                all_articles_indexed.append(indexed_article)
    except FileNotFoundError:
        print(f"Error: File {dev_set_path} not found.")
    print(f"Total unique words after filtering: {len(unique_words)}")
    print(f"Total articles processed: {len(all_articles_indexed)}")
    print(f"Shape of articales matrix: {pd.DataFrame(all_articles_indexed).shape}")
    # return list of all events    
    return all_articles_indexed, unique_words
    

def extract_labels(dev_set_path):
    # given the deveploment set path, extract all labels in the development set
    dict_article_label = {}
    try:
        with open(dev_set_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                labels = lines[i].strip().split()
                # remove the header the chars '<', 'TRAIN', '>' and digits
                labels = [label.replace('<',"") for label in labels]
                labels = [label.replace('>',"") for label in labels]
                labels = labels[2:]  # skip the first two tokens - 'TRAIN' and article id
                dict_article_label[i//4] = labels
    except FileNotFoundError:
        print(f"Error: File {dev_set_path} not found.")
    return dict_article_label

# initialize parameters
def parameter_initialization(all_articles_indexed, vocabulary, num_clusters=9, smoothing_factor=0.01):
    num_articles = len(all_articles_indexed)
    vocab_size = len(vocabulary)
    
    # ensure numpy float64 for precision
    ai = np.zeros(num_clusters, dtype=np.float64)
    Pik = np.zeros((num_clusters, vocab_size), dtype=np.float64)
    
    # assign articles to clusters in round-robin
    for idx, article in enumerate(all_articles_indexed):
        cluster_idx = idx % num_clusters
        ai[cluster_idx] += 1
    
    # ai normalization
    ai /= num_articles
    
    for i in range(num_clusters):
        # collect all words in the current cluster
        cluster_words = [word for idx, article in enumerate(all_articles_indexed) 
                         if idx % num_clusters == i for word in article]

        # fast counting using numpy
        counts = np.bincount(cluster_words, minlength=vocab_size).astype(np.float64)

        # calculation with Smoothing
        total_words_in_cluster = np.sum(counts)
        Pik[i] = (counts + smoothing_factor) / (total_words_in_cluster + smoothing_factor * vocab_size)
        
    return ai, Pik



def expectation_maximization(articles_indexed, vocab, smoothing_factor=1e-6):
    ai, Pik = parameter_initialization(articles_indexed, vocab, smoothing_factor=smoothing_factor)
    prev_log_likelihood = -float('inf')
    num_articles = len(articles_indexed)
    num_clusters = 9
    k_factor = 10
    delta = float('inf')
    while delta > TRESHOLD:
        # E-step
        total_log_likelihood = 0.0
        # matrix to hold weights
        weights = np.zeros((num_articles, num_clusters), dtype=np.float64)
        log_ai = np.log(ai)
        log_Pik = np.log(Pik)
        # iterate over articles
        for t, article in enumerate(articles_indexed):
            # count occurrences of each word in the article
            counts = np.bincount(article, minlength=len(vocab))
            relevant_indices = np.where(counts > 0)[0]
            
            # calculate zi for each cluster
            zi = np.zeros(num_clusters)
            for i in range(num_clusters):
                # formula (101): zi = ln(ai) + sum(n_tk * ln(Pik))
                zi[i] = log_ai[i] + np.dot(counts[relevant_indices], log_Pik[i, relevant_indices])
            
            # perform Scaling
            m = np.max(zi)
            diff = zi - m
            
            # calculate numerator: set to 0 if difference is less than -k
            numerator = np.where(diff < -k_factor, 0.0, np.exp(diff))
            denominator = np.sum(numerator)
            
            if denominator > 0:
                weights[t] = numerator / denominator
            # calculate log-likelihood with scaling
            total_log_likelihood += m + np.log(denominator)
        # M-step
        # update ai 
        ai = np.mean(weights, axis=0)
        
        alpha_threshold = 1e-6 
        if np.any(ai < alpha_threshold):
            ai = np.where(ai < alpha_threshold, alpha_threshold, ai)
            ai /= np.sum(ai) # נרמול מחדש כדי שסכום ההסתברויות יהיה 1 [cite: 122]

        # update Pik
        weighted_counts = np.zeros((num_clusters, len(vocab)), dtype=np.float64)
        
        for t, article in enumerate(articles_indexed):
            # count occurrences of each word in the article
            counts = np.bincount(article, minlength=len(vocab))
            # for each cluster i, add to the count (the weight of the article in cluster * number of occurrences of the word)
            for i in range(num_clusters):
                weighted_counts[i] += weights[t, i] * counts

        # smoothing and normalization
        for i in range(num_clusters):
            sum_weighted_counts = np.sum(weighted_counts[i]) # סך המילים המשוקללות באשכול i
            Pik[i] = (weighted_counts[i] + smoothing_factor) / (sum_weighted_counts + smoothing_factor * len(vocab))
        # Calculate log-likelihood
        delta = total_log_likelihood - prev_log_likelihood
        print(f"Log-Likelihood: {total_log_likelihood:.4f}, Improvement: {delta:.4f}")
        prev_log_likelihood = total_log_likelihood
        calculate_perplexity(total_log_likelihood, articles_indexed)
    return ai, Pik, weights, total_log_likelihood


def perform_hard_assignment(weights):
    #for each article, assign it to the cluster with the highest weight
    assignments = np.argmax(weights, axis=1)

    # print the distribution of articles among clusters (useful for reporting)
    cluster_counts = np.bincount(assignments, minlength=9)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count} articles")
        
    return assignments


def assign_labels():
    # Assign labels based on learned parameters
    pass

def calculate_perplexity(log_likelihood, articles_indexed):
    # calculate total number of words N
    total_words = sum(len(article) for article in articles_indexed)

    # Calculate mean perplexity according to the formula
    # Perplexity = exp(- (1/N) * LogLikelihood)
    perplexity = np.exp(-(1 / total_words) * log_likelihood)
    print(f"Mean Perplexity: {perplexity:.4f}")
    return perplexity, total_words


def calculate_confusion_matrix():
    # Calculate confusion matrix
    pass

def calculate_accuracy():
    pass

if __name__ == "__main__":
    development_set_preprocessing(dev_set_path)
    extract_labels(dev_set_path)
    articles_indexed, vocabulary = development_set_preprocessing(dev_set_path)
    ai, Pik, weights, log_likelihood = expectation_maximization(articles_indexed, vocabulary, smoothing_factor=1e-6)
    perform_hard_assignment(weights)
    