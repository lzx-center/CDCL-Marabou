import os
import random

from nnet import NNet, EquationType

"""The Projected Gradient Descent attack."""
import numpy as np
import torch
from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.utils import clip_eta


def clamp(input_vec, norm_min, norm_max):
    input_vec = input_vec.detach()
    for i in range(input_vec.shape[0]):
        if float(input_vec[i]) < norm_min[i]:
            input_vec[i] = norm_min[i]
        elif float(input_vec[i]) > norm_max[i]:
            input_vec[i] = norm_max[i]
    input_vec.requires_grad_(True)
    return input_vec


def select_input_to_split_by_interval(norm_min, norm_max):
    max_index, max_val = 0, norm_max[0] - norm_min[0]
    for i in range(len(norm_min)):
        interval = norm_max[i] - norm_min[i]
        if max_val < interval:
            max_index = i
            max_val = interval
    return max_index


class Attacker:

    def fast_gradient_method(
            self,
            model_fn,
            x,
            eps,
            norm,
            clip_min=None,
            clip_max=None,
            y=None,
            targeted=False,
            sanity_checks=False,
    ):
        """
        PyTorch implementation of the Fast Gradient Method.
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param x: input tensor.
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param y: (optional) Tensor with true labels. If targeted is true, then provide the
                  target label. Otherwise, only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used
                  as labels to avoid the "label leaking" effect (explained in this paper:
                  https://arxiv.org/abs/1611.01236). Default is None.
        :param targeted: (optional) bool. Is the attack targeted or untargeted?
                  Untargeted, the default, will try to make the label incorrect.
                  Targeted will instead try to move in the direction of being more like y.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                  memory or for unit tests that intentionally pass strange input)
        :return: a tensor for the adversarial example
        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError(
                "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
            )
        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return x
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_ge)

        if clip_max is not None:
            assert_le = torch.all(
                torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_le)

        # x needs to be a leaf variable, of floating point type and have requires_grad being True for
        # its grad to be computed and stored properly in a backward call
        x = x.clone().detach().to(torch.float).requires_grad_(True)
        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            _, y = torch.max(model_fn(x), 1)

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model_fn(x), y)
        # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        loss.backward()
        optimal_perturbation = optimize_linear(x.grad, eps, norm)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                )
            adv_x = clamp(adv_x, clip_min, clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x

    def projected_gradient_descent(
            self,
            model_fn,
            x,
            eps,
            eps_iter,
            nb_iter,
            norm,
            clip_min=None,
            clip_max=None,
            y=None,
            targeted=False,
            rand_init=True,
            rand_minmax=None,
            sanity_checks=True,
    ):
        """
        This class implements either the Basic Iterative Method
        (Kurakin et al. 2016) when rand_init is set to False. or the
        Madry et al. (2017) method if rand_init is set to True.
        Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
        Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param x: input tensor.
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param eps_iter: step size for each attack iteration
        :param nb_iter: Number of attack iterations.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param y: (optional) Tensor with true labels. If targeted is true, then provide the
                  target label. Otherwise, only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used
                  as labels to avoid the "label leaking" effect (explained in this paper:
                  https://arxiv.org/abs/1611.01236). Default is None.
        :param targeted: (optional) bool. Is the attack targeted or untargeted?
                  Untargeted, the default, will try to make the label incorrect.
                  Targeted will instead try to move in the direction of being more like y.
        :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
        :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
                  which the random perturbation on x was drawn. Effective only when rand_init is
                  True. Default equals to eps.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                  memory or for unit tests that intentionally pass strange input)
        :return: a tensor for the adversarial example
        """
        if norm == 1:
            raise NotImplementedError(
                "It's not clear that FGM is a good inner loop"
                " step for PGD when norm=1, because norm=1 FGM "
                " changes only one pixel at a time. We need "
                " to rigorously test a strong norm=1 PGD "
                "before enabling this feature."
            )
        if norm not in [np.inf, 2]:
            raise ValueError("Norm order must be either np.inf or 2.")
        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return x
        if eps_iter < 0:
            raise ValueError(
                "eps_iter must be greater than or equal to 0, got {} instead".format(
                    eps_iter
                )
            )
        if eps_iter == 0:
            return x

        assert eps_iter <= eps, (eps_iter, eps)
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_ge)

        if clip_max is not None:
            assert_le = torch.all(
                torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_le)

        # Initialize loop variables
        if rand_init:
            if rand_minmax is None:
                rand_minmax = eps
            eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
        else:
            eta = torch.zeros_like(x)

        # Clip eta
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = clamp(adv_x, clip_min, clip_max)

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            _, y = torch.max(model_fn(x), 1)

        i = 0
        while i < nb_iter:
            adv_x = self.fast_gradient_method(
                model_fn,
                adv_x,
                eps_iter,
                norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y,
                targeted=targeted,
            )

            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x
            eta = clip_eta(eta, norm, eps)
            adv_x = x + eta

            # Redo the clipping.
            # FGM already did it, but subtracting and re-adding eta can add some
            # small numerical error.
            if clip_min is not None or clip_max is not None:
                adv_x = clamp(adv_x, clip_min, clip_max)
            i += 1

        asserts.append(eps_iter <= eps)
        # if norm == np.inf and clip_min is not None:
        #     # TODO necessary to cast clip_min and clip_max to x.dtype?
        #     asserts.append(eps + clip_min <= clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x


class InputGenerate:

    def __init__(self):
        self.samples = []

    def inputs_generate(self, norm_min, norm_max, sample_num):
        self.samples.clear()
        sample_vals = []
        for i in range(len(norm_min)):
            sample_vals.append(np.random.uniform(norm_min[i], norm_max[i], sample_num))
        self.samples = [[vals[i] for vals in sample_vals] for i in range(sample_num)]
        return self.samples

    def clear(self):
        self.samples.clear()

    def inputs_generate_by_split_inputs(self, norm_min, norm_max, remain_split_num):
        if remain_split_num == 0:
            self.samples.append([(norm_max[i] + norm_min[i]) / 2.0 for i in range(len(norm_min))])
            return self.samples
        index = select_input_to_split_by_interval(norm_min, norm_max)
        n_min, n_max = [val for val in norm_min], [val for val in norm_max]
        mid = (n_max[index] + n_min[index]) / 2

        n_min[index] = mid
        self.inputs_generate_by_split_inputs(n_min, n_max, remain_split_num - 1)

        n_min[index] = norm_min[index]
        n_max[index] = mid
        self.inputs_generate_by_split_inputs(n_min, n_max, remain_split_num - 1)
        return self.samples


def attack_single_sample(net, sample, is_target, print_info=False):
    sample = torch.tensor(sample, requires_grad=True)
    y = np.zeros(net.num_outputs())
    y[0] = 1.0
    y = torch.tensor(y)
    adv = Attacker().projected_gradient_descent(net, sample, 0.2, 0.01, 10, torch.inf,
                                                clip_min=net.norm_mins, clip_max=net.norm_maxes, y=y,
                                                targeted=is_target)
    sat = net.satisfy_property(adv)
    loss_fn = torch.nn.CrossEntropyLoss()
    sample = sample.clone().detach().to(torch.float).requires_grad_(True)
    loss = loss_fn(net(sample), y)

    if sat and print_info:
        print("---" * 20)
        print(f"o: {sample.tolist()}\n")
        print(f"min: {net.norm_mins}")
        print(f"adv: {adv.data.tolist()}")
        print(f"max: {net.norm_maxes}")
        print(f"loss: {loss}\ntarget: {is_target}")
    return sat, adv, loss


def attack_network(net_path, property_path):
    net = NNet(net_path)
    net.load_property(property_path)
    is_target = False
    for equation in net.property_equation:
        if equation.type == EquationType.ge:
            is_target = True
    attacker = Attacker()
    # generate input
    # samples = InputGenerate().inputs_generate_by_split_inputs(net.norm_mins, net.norm_maxes, 10)
    samples = InputGenerate().inputs_generate(net.norm_mins, net.norm_maxes, 1000)
    for sample in samples:
        sat, adv, _ = attack_single_sample(net, sample, is_target)
        if sat:
            print(f"file {net_path}")
            print(f"file {property_path}")
            print("---" * 20)
            return True, adv
    return False, None


def main():
    net_folders = [
        "/home/center/CDCL-Marabou/sat_example/prop2/",
        "/home/center/CDCL-Marabou/sat_example/prop3",
        "/home/center/CDCL-Marabou/sat_example/prop4",
        # "/home/center/CDCL-Marabou/resources/nnet/acasxu",
    ]
    property_paths = [
        "/home/center/CDCL-Marabou/sat_example/prop2/acas_property_2.txt",
        "/home/center/CDCL-Marabou/sat_example/prop3/acas_property_3.txt",
        "/home/center/CDCL-Marabou/sat_example/prop4/acas_property_4.txt",
        # "/home/center/CDCL-Marabou/resources/properties/acas_property_1.txt",
    ]
    counter = 0
    success_num = 0
    can_not_attack = []
    for i in range(len(property_paths)):
        net_folder = net_folders[i]
        for home, dirs, files in os.walk(net_folder):
            for file in files:
                if file.endswith(".nnet"):
                    net_path = os.path.join(net_folder, file)
                    property_path = property_paths[i]
                    success, _ = attack_network(net_path, property_path)
                    counter += 1
                    if success:
                        success_num += 1
                    else:
                        can_not_attack.append(file)
    print(f"total {counter}, success {success_num}, ratio: {success_num / counter * 100}%")
    print(f"Can not attack {can_not_attack}")


if __name__ == "__main__":
    main()
