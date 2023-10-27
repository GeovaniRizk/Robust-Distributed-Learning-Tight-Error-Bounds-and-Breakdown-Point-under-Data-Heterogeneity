import torch, random, math
import misc

def average(aggreagator, vectors):
    return torch.stack(vectors).mean(dim=0)


def trmean(aggregator, vectors):
    if aggregator.nb_byz == 0:
        return torch.stack(vectors).mean(dim=0)
    return torch.stack(vectors).sort(dim=0).values[aggregator.nb_byz:-aggregator.nb_byz].mean(dim=0)


def median(aggregator, vectors):
    return torch.stack(vectors).quantile(q=0.5, dim=0)
    #return torch.stack(vectors).median(dim=0)[0]


def geometric_median(aggregator, vectors):
    return misc.smoothed_weiszfeld(vectors)


def krum(aggregator, vectors):
    #JS: Compute all pairwise distances
    distances = misc.compute_distances(vectors)
    #JS: return the vector with smallest score
    return misc.get_vector_best_score(vectors, aggregator.nb_byz, distances)


def multi_krum(aggregator, vectors):
    #JS: k is the number of vectors to average in the end
    k = len(vectors) - aggregator.nb_byz
    #JS: Compute all pairwise distances
    distances = misc.compute_distances(vectors)
    #JS: get scores of vectors, sorted in increasing order
    scores = misc.get_vector_scores(vectors, aggregator.nb_byz, distances)
    best_vectors = [vectors[worker_id] for _, worker_id in scores[:k]]
    #JS: return the average of the k vectors with lowest scores
    return torch.stack(best_vectors).mean(dim=0)


def nearest_neighbor_mixing(aggregator, vectors, numb_iter=1):
    for _ in range(numb_iter):
        mixed_vectors = list()
        for vector in vectors:
            #JS: Replace every vector by the average of its nearest neighbors
            mixed_vectors.append(misc.average_nearest_neighbors(vectors, aggregator.nb_byz, vector))
        vectors = mixed_vectors
    return robust_aggregators[aggregator.second_aggregator](aggregator, vectors)


def bucketing(aggregator, vectors):
    random.shuffle(vectors)
    number_buckets = math.ceil(len(vectors) / aggregator.bucket_size)
    buckets=[vectors[i:i + aggregator.bucket_size] for i in range(0, len(vectors), aggregator.bucket_size)]
    avg_vectors = list()

    for bucket_id in range(number_buckets):
        avg_vectors.append(torch.stack(buckets[bucket_id]).mean(dim=0))

    return robust_aggregators[aggregator.second_aggregator](aggregator, avg_vectors)


def pseudo_multi_krum(aggregator, vectors):
    k = len(vectors) - aggregator.nb_byz
    k_vectors = list()

    #JS: dictionary to hold pairwise distances
    distances = dict()
    indices = range(len(vectors))

    #JS: Run Pseudo Krum k times, and store result in list then average
    for _ in range(k):
        #JS: choose (f+1) vectors at random, and compute their pseudo-scores
        random_indices = random.sample(indices, aggregator.nb_byz + 1)
        #JS: compute the pseudo-scores of only these random vectors
        #JS: a pseudo-score is the same as a normal score, but computed only over a random set of (n-f) neighbors
        min_score = min_index = None

        for index in random_indices:
            #JS: vectors[index] is one of the candidates to be outputted by pseudo-Krum
            random_neighbors = random.sample(indices, k)
            score = 0
            for neighbor in random_neighbors:

                #JS: if index = neighbour, distance = 0 and score is unchanged
                if index == neighbor:
                    continue

                #JS: fetch the distance between vector and neighbor from dictionary (if found)
                #otherwise calculate it and store it in dictionary
                key = (min(index, neighbor), max(index, neighbor))

                if key in distances:
                    dist = distances[key]
                else:
                    dist = vectors[index].sub(vectors[neighbor]).norm().item()
                    distances[key] = dist

                score += dist**2

            if min_score is None or score < min_score:
                min_score = score
                min_index = index
        
        #JS: append the vector with the smallest score (among the considered f+1) to the list
        k_vectors.append(vectors[min_index])

    #JS: return the average of the k vectors
    return torch.stack(k_vectors).mean(dim=0)


def centered_clipping(aggregator, vectors, L_iter=3, clip_thresh=1):
    #JS: v is the returned vector, as per the algorithm of CC
    v = aggregator.prev_momentum
    for _ in range(L_iter):
        clipped_distances = misc.compute_distance_vectors(vectors, v, clip_thresh)
        # avg_dist = torch.stack(clipped_distances).mean(dim=0)
        avg_dist = sum(clipped_distances).div(len(clipped_distances))
        v.add_(avg_dist)
        torch.cuda.empty_cache()
        import gc
        del clipped_distances
        gc.collect()
    return v


def minimum_diameter_averaging(aggregator, vectors):
    selected_subset = misc.compute_min_diameter_subset(vectors, aggregator.nb_byz)
    selected_vectors = [vectors[j] for j in selected_subset]
    return torch.stack(selected_vectors).mean(dim=0)


def minimum_variance_averaging(aggregator, vectors):
    selected_subset = misc.compute_min_variance_subset(vectors, aggregator.nb_byz)
    selected_vectors = [vectors[j] for j in selected_subset]
    return torch.stack(selected_vectors).mean(dim=0)


def MoNNA(aggregator, vectors):
    # Compute n-f closest vectors to the pivot vector (i.e., the vector of the honest worker in question)
    closest_vectors = misc.compute_closest_vectors(vectors, aggregator.nb_byz)
    # Return the average of closest_vectors
    return torch.stack(closest_vectors).mean(dim=0)


def meamed(aggregator, vectors):
    vectors_stacked = torch.stack(vectors)
    median_vector = robust_aggregators["median"](aggregator, vectors)
    nb_workers, dimension = vectors_stacked.shape
    m = nb_workers - aggregator.nb_byz
    #JS: compute and aggregate (n-f) vectors closest to median (per dimension)
    bottom_indices = vectors_stacked.sub(median_vector).abs().topk(m, dim=0, largest=False, sorted=False).indices
    bottom_indices.mul_(dimension).add_(torch.arange(0, dimension, dtype=bottom_indices.dtype, device=bottom_indices.device))
    return vectors_stacked.take(bottom_indices).mean(dim=0)


#JS: Dictionary mapping every aggregator to its corresponding function
robust_aggregators = {"average": average, "trmean": trmean, "median": median, "geometric_median": geometric_median, "krum": krum, "multi_krum": multi_krum,
                      "nnm": nearest_neighbor_mixing, "bucketing": bucketing, "pmk": pseudo_multi_krum, "cc": centered_clipping, "mda": minimum_diameter_averaging,
                      "mva": minimum_variance_averaging, "MoNNA": MoNNA, "meamed": meamed}

class RobustAggregator(object):

    def __init__(self, aggregator_name, second_aggregator, bucket_size, nb_byz, model_size, device):

        self.aggregator_name = aggregator_name
        self.second_aggregator = second_aggregator
        self.bucket_size = bucket_size

        self.nb_byz = nb_byz

        #JS; previous value of aggregated momentum, used for example for CC
        self.prev_momentum = torch.zeros(model_size, device=device)

    def aggregate(self, vectors):
        aggregate_vector = robust_aggregators[self.aggregator_name](self, vectors)
        #JS: Update the value of the previous momentum (e.g., for Centered Clipping aggregator)
        self.prev_momentum = aggregate_vector
        return aggregate_vector