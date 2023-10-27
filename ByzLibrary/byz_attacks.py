import torch, math
import misc

def signflipping(attack, honest_vectors, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    byzantine_vector = torch.mul(avg_honest_vector, -1)
    return [byzantine_vector] * attack.nb_real_byz


def labelflipping(attack, flipped_vectors, **kwargs):
    avg_flipped_vector = torch.stack(flipped_vectors).mean(dim=0)
    return [avg_flipped_vector] * attack.nb_real_byz


def fall_of_empires(attack, honest_vectors, attack_factor=3, negative=False, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    attack_vector = avg_honest_vector.neg()
    if negative:
        attack_factor = - attack_factor
    byzantine_vector = avg_honest_vector.add(attack_vector, alpha=attack_factor)
    return [byzantine_vector] * attack.nb_real_byz


def auto_FOE(attack, honest_vectors, aggregator, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_FOE(factor):
        byzantine_vectors = fall_of_empires(attack, honest_vectors, attack_factor=factor)
        distance = aggregator.aggregate(honest_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_FOE)
    return fall_of_empires(attack, honest_vectors, attack_factor=best_factor)


def a_little_is_enough(attack, honest_vectors, attack_factor=3, negative=False, **kwargs):
    stacked_vectors = torch.stack(honest_vectors)
    avg_honest_vector = stacked_vectors.mean(dim=0)
    attack_vector = stacked_vectors.var(dim=0).sqrt_()
    if negative:
        attack_factor = - attack_factor
    byzantine_vector = avg_honest_vector.add(attack_vector, alpha=attack_factor)
    return [byzantine_vector] * attack.nb_real_byz


def auto_ALIE(attack, honest_vectors, aggregator, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_ALIE(factor):
        byzantine_vectors = a_little_is_enough(attack, honest_vectors, attack_factor=factor)
        distance = aggregator.aggregate(honest_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_ALIE)
    return a_little_is_enough(attack, honest_vectors, attack_factor=best_factor)


def mimic(attack, honest_vectors, **kwargs):
    return [honest_vectors[0]] * attack.nb_real_byz


def mimic_heuristic(attack, honest_vectors, current_step, learning_phase, **kwargs):
    if current_step < learning_phase:
        return [honest_vectors[0]] * attack.nb_real_byz

    current_max = None
    best_worker_to_mimic = None
    for i, vector in enumerate(honest_vectors):
        dot_product = torch.dot(vector, attack.z_mimic).norm().item()
        if current_max is None or dot_product > current_max:
            current_max = dot_product
            best_worker_to_mimic = i

    return [honest_vectors[best_worker_to_mimic]] * attack.nb_real_byz

def inf(attack, honest_vectors, **kwargs):
    byzantine_vector = torch.empty_like(honest_vectors[0])
    byzantine_vector.copy_(torch.tensor((math.inf,), dtype=byzantine_vector.dtype))
    return [byzantine_vector] * attack.nb_real_byz


byzantine_attacks = {"SF": signflipping, "LF": labelflipping, "FOE": fall_of_empires, "ALIE": a_little_is_enough, "mimic": mimic,
                     "mimic_heuristic": mimic_heuristic, "auto_ALIE": auto_ALIE, "auto_FOE": auto_FOE, "inf": inf}

class ByzantineAttack(object):

    def __init__(self, attack_name, nb_real_byz, model_size, device, learning_phase, robust_aggregator):

        self.attack_name = attack_name
        self.nb_real_byz = nb_real_byz

        self.z_mimic = torch.rand(model_size, device=device)
        self.mu_mimic = torch.zeros(model_size, device=device)
        self.learning_phase_mimic = learning_phase

        self.robust_aggregator = robust_aggregator

        self.model_size = model_size
        self.device = device


    def generate_byzantine_vectors(self, honest_vectors, flipped_vectors, current_step):
        if self.nb_real_byz == 0:
            return list()
        
        if self.attack_name == "mimic_heuristic" and current_step < self.learning_phase_mimic:
            self.update_mimic_heurstic(honest_vectors, current_step)

        return byzantine_attacks[self.attack_name](self, honest_vectors=honest_vectors, flipped_vectors=flipped_vectors, current_step=current_step,
                                                   learning_phase=self.learning_phase_mimic, aggregator=self.robust_aggregator)


    def update_mimic_heurstic(self, honest_vectors, current_step):
        avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
        self.mu_mimic = torch.add(torch.mul(self.mu_mimic, (current_step+1)/(current_step+2)), avg_honest_vector, alpha=1/(current_step+2))

        self.z_mimic.mul_((current_step+1)/(current_step+2))
        cumulative = torch.zeros(self.model_size, device=self.device)
        for vector in honest_vectors:
            deviation = torch.sub(vector, self.mu_mimic)
            temp_value = torch.dot(deviation, self.z_mimic).norm().item()
            deviation.mul_(temp_value)
            cumulative.add_(deviation)

        cumulative.div_(cumulative.norm().item())
        self.z_mimic = torch.add(torch.mul(self.z_mimic, (current_step+1)/(current_step+2)), cumulative, alpha=1/(current_step+2))
        self.z_mimic.div_(self.z_mimic.norm().item())