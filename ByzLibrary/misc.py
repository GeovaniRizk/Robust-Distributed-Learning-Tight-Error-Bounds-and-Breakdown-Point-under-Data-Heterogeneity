import torch, math
from itertools import combinations

# ---------------------------------------------------------------------------- #

#Criterions to evaluate accuracy of models. Used in worker.py and p2pWorker.py

def topk(output, target, k=1):
      """ Compute the top-k criterion from the output and the target.
      Args:
        output Batch × model logits
        target Batch × target index
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      res = (output.topk(k, dim=1)[1] == target.view(-1).unsqueeze(1)).any(dim=1).sum()
      return torch.cat((res.unsqueeze(0), torch.tensor(target.shape[0], dtype=res.dtype, device=res.device).unsqueeze(0)))


def sigmoid(output, target):
      """ Compute the sigmoid criterion from the output and the target.
      Args:
        output Batch × model logits (expected in [0, 1])
        target Batch × target index (expected in {0, 1})
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      correct = target.sub(output).abs_() < 0.5
      res = torch.empty(2, dtype=output.dtype, device=output.device)
      res[0] = correct.sum()
      res[1] = len(correct)
      return res

# ---------------------------------------------------------------------------- #

#Functions for manipulating gradients and model parameters

#flatten list of tensors. Used for model parameters and gradients
def flatten(list_of_tensors):
    return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensors))

#unflatten a flat tensor. Used when setting model parameters and gradients
def unflatten(flat_tensor, model_shapes):
    c = 0
    returned_list = [torch.zeros(shape) for shape in model_shapes]
    for i, shape in enumerate(model_shapes):
        count = 1
        for element in shape:
            count *= element
        returned_list[i].data = flat_tensor[c:c + count].view(shape)
        c = c + count
    return returned_list

# ---------------------------------------------------------------------------- #

#Functions for robust aggregators

#Approximation algorithm used for geometric median
def smoothed_weiszfeld(vectors, nu=0.1, T=3):
    """ Smoothed Weiszfeld algorithm
    Args:
        vectors: non-empty list of vectors to aggregate
        alphas: scaling factors
        nu: RFA parameter
        T: number of iterations to run the smoothed Weiszfeld algorithm
    Returns:
        Aggregated vector
    """
    n = len(vectors)
    z = torch.zeros_like(vectors[0])
    alphas = [1 / len(vectors)] *  len(vectors)

    for _ in range(T):
        betas = list()
        for i in range(n):
            #Compute the norm of z - vector
            distance = z.sub(vectors[i]).norm().item()
            if math.isnan(distance):
                #distance is infinite
                betas.append(0)
            else:
                betas.append(alphas[i] / max(distance, nu))

        z = torch.zeros_like(vectors[0])
        for vector, beta in zip(vectors, betas):
            if beta != 0:
                z.add_(vector, alpha=beta)

        z.div_(sum(betas))
    return z

#used for Krum, Multi-Krum , and MDA aggregators
def compute_distances(vectors):
    """ Compute all pairwise distances between vectors"""
    distances = dict()
    all_pairs = list(combinations(range(len(vectors)), 2))
    for (x,y) in all_pairs:
        dist = vectors[x].sub(vectors[y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[(x,y)] = dist
    return distances

#used for Krum aggregator
def get_vector_best_score(vectors, nb_byz, distances):
    """ Get the vector with the smallest score."""

    #compute the scores of all vectors
    min_score = math.inf

    for worker_id in range(len(vectors)):
        distances_squared_to_vector = list()

        #Compute the distances of all other vectors to vectors[worker_id]
        for neighbour in range(len(vectors)):
            if neighbour != worker_id:
                dist = distances.get((min(worker_id, neighbour), max(worker_id, neighbour)), 0)
                distances_squared_to_vector.append(dist**2)

        distances_squared_to_vector.sort()
        score = sum(distances_squared_to_vector[:len(vectors) - nb_byz - 1])

        #update min score
        if score < min_score:
            min_score, min_index = score, worker_id

    #return the vector with smallest score
    return vectors[min_index]

#get the scores of vectors sorted increasingly
def get_vector_scores(vectors, nb_byz, distances):

    #compute the scores of vectors
    scores = list()

    for worker_id in range(len(vectors)):
        distances_squared_to_vector = list()

        #Compute the distances of all other vectors to vectors[worker_id]
        for neighbor in range(len(vectors)):
            if neighbor != worker_id:
                dist = distances.get((min(worker_id, neighbor), max(worker_id, neighbor)), 0)
                distances_squared_to_vector.append(dist**2)

        distances_squared_to_vector.sort()
        score = sum(distances_squared_to_vector[:len(vectors) - nb_byz - 1])
        scores.append((score, worker_id))

    scores.sort(key=lambda x: x[0])
    return scores


#Compute the average of the n-f closest vectors to pivot
def average_nearest_neighbors(vectors, f, pivot):
    vector_scores = list()
    
    for i in range(len(vectors)):
        #compute distance to pivot
        distance = vectors[i].sub(pivot).norm().item()
        vector_scores.append((i, distance))
    
    #sort vector_scores by increasing distance to pivot
    vector_scores.sort(key=lambda x: x[1])
    
    #Return the average of the n-f closest vectors to pivot
    closest_vectors = [vectors[vector_scores[j][0]] for j in range(len(vectors) -f)]
    return torch.stack(closest_vectors).mean(dim=0)


 #Compute clipped distances of workers' vectors from the result of previous aggregation previous_aggregate
 #Used for Centered Clipping aggregator
def compute_distance_vectors(vectors, previous_aggregate, clip_thresh):
    clipped_distances = list()
    for vector in vectors:
        distance = vector.sub(previous_aggregate)
        distance_norm = distance.norm().item()
        if distance_norm > clip_thresh:
            #clip the distance vector
            distance.mul_(clip_thresh / distance_norm)
        clipped_distances.append(distance)
    return clipped_distances


#Compute the subset of (n-f) gradients of minimum diameter 
def compute_min_diameter_subset(vectors, nb_byz):
    #compute all pairwise distances
    distances = compute_distances(vectors)
    min_diameter = math.inf

    n = len(vectors)
    #Get all subsets of size n - f
    all_subsets = list(combinations(range(n), n - nb_byz))
    for subset in all_subsets:
        subset_diameter = 0
        
        #Compute diameter of subset
        for i, vector1 in enumerate(subset):
            for vector2 in subset[i+1:]:
                distance = distances.get((vector1, vector2), 0)
                subset_diameter = distance if distance > subset_diameter else subset_diameter
        
        #Update min diameter (if needed)
        if min_diameter > subset_diameter:
            min_diameter = subset_diameter
            min_subset = subset

    return min_subset


#Compute the subset (indices of vectors) of (n-f) vectors of minimum variance 
def compute_min_variance_subset(vectors, nb_byz):
    n = len(vectors)
    #Get all subsets of size n - f
    all_subsets = list(combinations(range(n), n - nb_byz))
    min_variance = math.inf

    for subset in all_subsets:
        subset_vectors = [vectors[j] for j in subset]
        avg_vector = torch.stack(subset_vectors).mean(dim=0)

        #Compute variance of subset
        current_variance = 0
        for vector in subset_vectors:
            dist_from_avg = vector.sub(avg_vector).norm().item()
            current_variance += dist_from_avg**2

        if min_variance > current_variance:
            min_variance = current_variance
            min_subset = subset
        
    return min_subset


#Compute the n-f closest vectors to the honest vector in question (last element of vectors)
#Used in MoNNA aggregation rule
def compute_closest_vectors(vectors, nb_byz):
    pivot_vector = vectors[-1]
    vector_scores = list()

    for i, vector_i in enumerate(vectors):
        #compute distance to pivot_vector
        distance = vector_i.sub(pivot_vector).norm().item()
        vector_scores.append((i, distance))
    #sort vector_scores by increasing distance to pivot_vector
    vector_scores.sort(key=lambda x: x[1])
    #return the n-f closest vectors to pivot_vector
    return [vectors[vector_scores[j][0]] for j in range(len(vectors) - nb_byz)]


#used for Auto ALIE and Auto FOE
def line_maximize(scape, evals=16, start=0., delta=1., ratio=0.8):
  """ Best-effort arg-maximize a scape: ℝ⁺⟶ ℝ, by mere exploration.
  Args:
    scape Function to best-effort arg-maximize
    evals Maximum number of evaluations, must be a positive integer
    start Initial x evaluated, must be a non-negative float
    delta Initial step delta, must be a positive float
    ratio Contraction ratio, must be between 0.5 and 1. (both excluded)
  Returns:
    Best-effort maximizer x under the evaluation budget
  """
  # Variable setup
  best_x = start
  best_y = scape(best_x)
  evals -= 1
  # Expansion phase
  while evals > 0:
    prop_x = best_x + delta
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
      delta *= 2
    else:
      delta *= ratio
      break
  #Contraction phase
  while evals > 0:
    if prop_x < best_x:
      prop_x += delta
    else:
      x = prop_x - delta
      while x < 0:
        x = (x + prop_x) / 2
      prop_x = x
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
    # Reduce delta
    delta *= ratio
  #Return found maximizer
  return best_x