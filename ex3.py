# Submitted by Sapir Bar [211540562] and Dvir Adler [206923211]
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from collections import Counter

def development_set_preprocessing(develop_path: str = "develop.txt"):
    # compute vocabulary after filtering rare words (less than 4 appearances)
    # compute list of indexed articles
    
    try:
        with open(develop_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract all words and count frequencies
        all_words = [w for i in range(2, len(lines), 4) for w in lines[i].strip().split()]
        word_counts = Counter(all_words)
        #print(f"Total unique words before filtering: {len(word_counts)}")
        
        # Filter out rare words and create word-to-id mapping
        unique_words = sorted([word for word, count in word_counts.items() if count >= 4])
        word_to_id = {word: i for i, word in enumerate(unique_words)}
        
        # Convert articles to indexed format, keeping only non-rare words
        all_articles_indexed = [
            [word_to_id[w] for w in lines[i].strip().split() if w in word_to_id]
            for i in range(2, len(lines), 4)
        ]
        
        # print(f"Total unique words after filtering: {len(unique_words)}")
        # print(f"Total articles processed: {len(all_articles_indexed)}")
        return all_articles_indexed, unique_words
        
    except FileNotFoundError:
        print(f"Error: File {develop_path} not found.")
    

def extract_labels(develop_path: str = "develop.txt"):
    try:
        with open(develop_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract labels, remove '<' and '>', skip first two tokens per line
        dict_article_label = {
            i // 4: [label.replace('<', '').replace('>', '') for label in lines[i].strip().split()[2:]]
            for i in range(0, len(lines), 4)
        }
        return dict_article_label
        
    except FileNotFoundError:
        print(f"Error: File {develop_path} not found.")
        return {}

def parameter_initialization(articles, vocab, num_clusters=9, _lambda=1e-4, epsilon=1e-6):
    """Initialize with round-robin cluster assignment, compute a_i and P_ik with smoothing."""
    n_articles, vocab_size = len(articles), len(vocab)
    
    # Round-robin assignment: count articles per cluster
    ai = np.bincount([i % num_clusters for i in range(n_articles)], minlength=num_clusters).astype(np.float64) / n_articles
    ai = smooth_mixture_weights(ai, epsilon=epsilon)
    
    # Compute P_ik: word distributions per cluster
    Pik = np.zeros((num_clusters, vocab_size), dtype=np.float64)
    for i in range(num_clusters):
        cluster_words = np.concatenate([articles[j] for j in range(n_articles) if j % num_clusters == i])
        counts = np.bincount(cluster_words, minlength=vocab_size).astype(np.float64)
        Pik[i] = smooth_word_distribution_lidstone(counts, vocab_size, _lambda=_lambda)
    
    return ai, Pik


def smooth_mixture_weights(ai, epsilon=1e-6):
    """Fix a_i to epsilon and normalize to sum to 1."""
    ai = np.where(ai < epsilon, epsilon, ai)
    return ai / np.sum(ai)


def smooth_word_distribution_lidstone(weighted_counts, vocab_size, _lambda=1e-6):
    """Apply Lidstone smoothing: (count + lambda) / (total + lambda * vocab_size)."""
    return (weighted_counts + _lambda) / (np.sum(weighted_counts) + _lambda * vocab_size)


def expectation_maximization(articles_indexed, vocab, _lambda=1e-6, epsilon=1e-6, threshold=1e-4):
    ai, Pik = parameter_initialization(articles_indexed, vocab, _lambda=_lambda, epsilon=epsilon)
    log_likelihood_history = []
    perplexity_history = []
    total_words = sum(len(a) for a in articles_indexed)

    prev_log_likelihood = -float('inf')
    num_articles = len(articles_indexed)
    num_clusters = 9
    k_factor = 10
    delta = float('inf')
    while delta > threshold:
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
        # update ai with smoothing and normalization
        ai = np.mean(weights, axis=0)
        ai = smooth_mixture_weights(ai, epsilon=epsilon)

        # update Pik
        weighted_counts = np.zeros((num_clusters, len(vocab)), dtype=np.float64)
        
        for t, article in enumerate(articles_indexed):
            # count occurrences of each word in the article
            counts = np.bincount(article, minlength=len(vocab))
            # for each cluster i, add to the count (the weight of the article in cluster * number of occurrences of the word)
            for i in range(num_clusters):
                weighted_counts[i] += weights[t, i] * counts

        # Apply Lidstone smoothing to word distributions
        for i in range(num_clusters):
            Pik[i] = smooth_word_distribution_lidstone(weighted_counts[i], len(vocab), _lambda=_lambda)
        # Calculate log-likelihood
        delta = total_log_likelihood - prev_log_likelihood
        prev_log_likelihood = total_log_likelihood
        log_likelihood_history.append(total_log_likelihood)
        if total_words > 0:
            perplexity = np.exp(-(1 / total_words) * total_log_likelihood)
        else:
            perplexity = float('inf')
        perplexity_history.append(perplexity)
    return ai, Pik, weights, total_log_likelihood, log_likelihood_history, perplexity_history


def read_topics(topics_path: str = "topics.txt"):
    """
    Reads topics in the required Reuters order from topics.txt (one topic per line).
    If the file doesn't exist, falls back to sorted unique topics observed in develop.txt.
    """
    try:
        with open(topics_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        return topics
    except FileNotFoundError:
        return None


def assign_labels(assignments, article_labels, topics_order=None):
    """
    For each cluster i, pick its 'dominant' topic = the topic that appears in the
    largest number of articles assigned to cluster i.

    Returns:
        cluster_to_topic: dict[int, str]
        cluster_topic_counts: pd.DataFrame (clusters x topics) raw counts
    """
    num_clusters = int(np.max(assignments)) + 1 if len(assignments) else 0

    # Determine topics order
    if topics_order is None:
        topics_order = read_topics("topics.txt")

    topic_to_idx = {t: j for j, t in enumerate(topics_order)}
    counts = np.zeros((num_clusters, len(topics_order)), dtype=int)

    for art_id, cluster in enumerate(assignments):
        labs = article_labels.get(art_id, [])
        for t in labs:
            j = topic_to_idx.get(t)
            if j is not None:
                counts[cluster, j] += 1

    cluster_to_topic = {}
    for i in range(num_clusters):
        if counts.shape[1] == 0:
            cluster_to_topic[i] = None
        else:
            cluster_to_topic[i] = topics_order[int(np.argmax(counts[i]))]

    cluster_topic_counts = pd.DataFrame(counts, columns=topics_order)
    cluster_topic_counts.insert(0, "cluster", list(range(num_clusters)))
    return cluster_to_topic, cluster_topic_counts


def calculate_perplexity(log_likelihood, articles_indexed):
    # calculate total number of words N
    total_words = sum(len(article) for article in articles_indexed)

    # Calculate mean perplexity according to the formula
    # Perplexity = exp(- (1/N) * LogLikelihood)
    perplexity = np.exp(-(1 / total_words) * log_likelihood)
    return perplexity, total_words


def calculate_confusion_matrix(assignments, article_labels, topics_order, num_clusters=9, save_output=False, output_path="confusion_matrix.csv"):
    """
    Builds the 9x9 confusion matrix described in the exercise:
    - rows: clusters (sorted by cluster size, descending)
    - cols: topics (in topics.txt order)
    - cell (i,j): #articles assigned to cluster i that have topic j (topic can be one of many labels)
    - plus a 'cluster_size' column
    - Cluster indices are re-assigned: 1 for biggest, 2 for second biggest, etc.

    Returns:
        df: pd.DataFrame with index 'cluster' (1-9) and columns topics_order + ['cluster_size']
    """
    topic_to_idx = {t: j for j, t in enumerate(topics_order)}
    M = np.zeros((num_clusters, len(topics_order)), dtype=int)
    cluster_sizes = np.zeros(num_clusters, dtype=int)

    for art_id, cluster in enumerate(assignments):
        if cluster < 0 or cluster >= num_clusters:
            continue
        cluster_sizes[cluster] += 1
        for t in article_labels.get(art_id, []):
            j = topic_to_idx.get(t)
            if j is not None:
                M[cluster, j] += 1

    df = pd.DataFrame(M, columns=topics_order)
    df["cluster_size"] = cluster_sizes

    # sort rows by cluster size desc and re-index clusters 1-9 by size rank
    df = df.sort_values("cluster_size", ascending=False).reset_index(drop=True)
    df["cluster"] = range(1, num_clusters + 1)
    df = df.set_index("cluster")
    if save_output:
        df.to_csv(output_path)
    return df

def calculate_accuracy(assignments, article_labels, cluster_to_topic, save_output=False, output_path="accuracy.txt"):
    """
    Accuracy definition from the exercise:
    predicted topic for an article = dominant topic of its assigned cluster.
    It's counted correct if that predicted topic is one of the article's Reuters topics.
    """
    correct = 0
    total = len(assignments)
    for art_id, cluster in enumerate(assignments):
        pred_topic = cluster_to_topic.get(int(cluster))
        if pred_topic is not None and pred_topic in article_labels.get(art_id, []):
            correct += 1
    accuracy = correct / total if total else 0.0
    if save_output:
        with open(output_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
    return accuracy


def plot_training_curves(log_likelihood_history, perplexity_history, out_prefix="training"):
    """
    Plots two graphs:
      1) log-likelihood vs iteration
      2) perplexity vs iteration
    """
    iters = list(range(1, len(log_likelihood_history) + 1))

    # Log-likelihood plot
    plt.figure()
    plt.plot(iters, log_likelihood_history)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood per Iteration")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_log_likelihood_curve.png", dpi=200)
    plt.close()

    # Perplexity plot
    plt.figure()
    plt.plot(iters, perplexity_history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Perplexity per Word")
    plt.title("Perplexity per Iteration")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_perplexity_curve.png", dpi=200)
    plt.close()

    # save log-likelihood and perplexity history to a text file
    with open(f"{out_prefix}_log.txt", "w") as f:
        f.write("Iteration\tLog-Likelihood (x10^6)\tPerplexity\n")
        for i in range(len(iters)):
            f.write(f"{iters[i]}\t{log_likelihood_history[i]*1e-6:.4f}\t{perplexity_history[i]:.4f}\n")


def plot_cluster_histograms(confusion_df, topics_order, output_path="cluster_histograms.png", save_output=False):
    """
    Saves a single figure with 9 subplots (3x3) showing histograms for each cluster.
    Each subplot shows the topic distribution for that cluster.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (cluster_id, row) in enumerate(confusion_df.iterrows()):
        counts = [int(row[t]) for t in topics_order]
        dominant_topic = topics_order[int(np.argmax(counts))] if topics_order else "N/A"
        
        axes[idx].bar(range(len(topics_order)), counts)
        axes[idx].set_xticks(range(len(topics_order)))
        axes[idx].set_xticklabels(topics_order, rotation=45, ha="right")
        axes[idx].set_ylabel("Number of Articles")
        axes[idx].set_title(f"Cluster {cluster_id} (dominant: {dominant_topic})")
    
    plt.tight_layout()
    if save_output:
        plt.savefig(output_path, dpi=200)
    plt.close()

if __name__ == "__main__":

    develop_path = 'develop.txt'
    topics_path = 'topics.txt' 
    topics_order = read_topics(topics_path)
    os.makedirs("output", exist_ok=True)

    articles_indexed, vocabulary = development_set_preprocessing(develop_path)
    article_labels = extract_labels(develop_path)

    ai, Pik, weights, log_likelihood, ll_hist, ppl_hist = expectation_maximization(
        articles_indexed, vocabulary, _lambda=1, epsilon=1e-6, threshold=1e-4
    )

    assignments = np.argmax(weights, axis=1)

    cluster_to_topic, _ = assign_labels(assignments, article_labels, topics_order=topics_order)
    confusion_df = calculate_confusion_matrix(assignments, article_labels, topics_order, num_clusters=9, save_output=True, output_path=f"output/confusion_matrix.csv")
    acc = calculate_accuracy(assignments, article_labels, cluster_to_topic, save_output=True, output_path=f"output/accuracy.txt")
    plot_training_curves(ll_hist, ppl_hist, out_prefix=f"output/training")
    plot_cluster_histograms(confusion_df, topics_order,  save_output=True, output_path="output/cluster_histograms.png")