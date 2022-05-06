from typing import List, Optional
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .precondition import KFAC, DiagNaturalGradient
from .fisher import FISHER_EXACT, FISHER_MC
from .kernel import batch, empirical_implicit_ntk, empirical_class_wise_direct_ntk, get_preconditioned_kernel_fn
from .utils import add_value_to_diagonal, nvtx_range


__all__ = [
    'FROMP',
]


_precond_classes = {'kron': KFAC, 'diag': DiagNaturalGradient}
_fisher_types = {'exact': FISHER_EXACT, 'mc': FISHER_MC}
_kernel_fns = {'implicit': empirical_implicit_ntk, 'class_wise': empirical_class_wise_direct_ntk}

TYPE_MEMORABLE_PAST = 1
TYPE_ERROR_CORRETION = 2


class PastTask:
    def __init__(self, memorable_points, class_ids=None, memorable_points_indices=None, memorable_points_indices_global=None, memorable_points_true_targets=None, memorable_points_types=None):
        self.memorable_points = memorable_points
        self.kernel_inv = None
        self.mean = None
        self.class_ids = class_ids
        self.memorable_points_indices = memorable_points_indices
        self.memorable_points_indices_global = memorable_points_indices_global
        self.memorable_points_true_targets = memorable_points_true_targets
        self.memorable_points_types = memorable_points_types
        self.n_memorable_points = len(self.memorable_points)

    def update_kernel(self, model, kernel_fn, eps=1e-5):
        memorable_points = self.memorable_points
        if isinstance(memorable_points, DataLoader):
            kernel = batch(kernel_fn, model, memorable_points)
        else:
            kernel = kernel_fn(model, memorable_points)
        n, c = kernel.shape[0], kernel.shape[-1]  # (n, n, c, c) or (n, n, c)
        ndim = kernel.ndim
        if ndim == 4:
            kernel = kernel.transpose(1, 2).reshape(n * c, n * c)  # (nc, nc)
        elif ndim == 3:
            kernel = kernel.transpose(0, 2)  # (c, n, n)
        else:
            raise ValueError(f'Invalid kernel ndim: {ndim}. ndim must be 3 or 4.')

        kernel = add_value_to_diagonal(kernel, eps)
        self.kernel_inv = torch.linalg.inv(kernel).detach_()

    @torch.no_grad()
    def update_mean(self, model, max_mem_per_batch=50, memory_loss_mode='soft_all', memory_residual_frac=1.0):#100):#200):#500):
        if memory_loss_mode == 'hard_all':
            self.mean = self._compute_memory_true_targets_one_hot()
            return

        import numpy as np
        n_batches = int(np.ceil(len(self.memorable_points) / max_mem_per_batch))

        if n_batches == 1:
            self.mean = self._evaluate_mean(model).cpu()
        else:
            # Split forward passes into mini-batches to save memory
            mem_batch_indices = np.array_split(range(len(self.memorable_points)), n_batches)
            self.mean = torch.cat([self._evaluate_mean(model, idx=idx).cpu() for idx in mem_batch_indices])
 
        if memory_loss_mode in ['soft_correct', 'soft_correct_hard_rest', 'soft_low_residual_hard_rest']:
            if self.class_ids != list(range(len(self.class_ids))):
                for i, class_id in enumerate(self.class_ids):
                    self.memorable_points_true_targets[self.memorable_points_true_targets == class_id] = i

            if memory_loss_mode == 'soft_low_residual_hard_rest':
                true_targets_one_hot = self._compute_memory_true_targets_one_hot()
                residuals = (self.mean - true_targets_one_hot).abs().sum(axis=1)

                # sort indices by residuals (across full dataset)
                n_memory_residual = int(self.n_memorable_points * memory_residual_frac)
                large_residual_indices = torch.argsort(residuals, descending=True)[:n_memory_residual]
                self.mean[large_residual_indices] = true_targets_one_hot[large_residual_indices]

            else:
                if memory_loss_mode == 'soft_correct':
                    mean_list = [m for m in self.mean]
                    memorable_points_list = [m for m in self.memorable_points]
                for idx in range(self.n_memorable_points)[::-1]:
                    model_prediction = self.mean[idx].argmax().item()
                    true_target = self.memorable_points_true_targets[idx]
                    if isinstance(true_target, torch.Tensor):
                        true_target = true_target.argmax().item() - min(self.class_ids)
                    if model_prediction != true_target:
                        if memory_loss_mode == 'soft_correct':
                            # Discard memory points with incorrect model predictions
                            del self.memorable_points_indices[idx]
                            del self.memorable_points_true_targets[idx]
                            del mean_list[idx]
                            del memorable_points_list[idx]
                        elif memory_loss_mode == 'soft_correct_hard_rest':
                            # Set incorrect model predictions to true targets
                            self.mean[idx] = _convert_targets_to_one_hot([true_target], len(self.class_ids))[0]

            if memory_loss_mode == 'soft_correct':
                self.mean = torch.stack(mean_list)
                self.memorable_points = torch.stack(memorable_points_list)
                self.n_memorable_points = len(self.memorable_points)

    def _compute_memory_true_targets_one_hot(self):
        """ return the true targets of the memory points as one-hot vectors """
        true_targets = []
        for true_target in self.memorable_points_true_targets:
            if isinstance(true_target, torch.Tensor):
                true_target = true_target.argmax().item() - min(self.class_ids)
            true_targets.append(true_target)
        return _convert_targets_to_one_hot(true_targets, len(self.class_ids))

    def _evaluate_mean(self, model, n_memorable_points_sub=None, idx=None):
        means = []
        memorable_points = self.memorable_points
        device = next(model.parameters()).device
        if isinstance(memorable_points, DataLoader):
            for i, (inputs, _) in enumerate(self.memorable_points):
                if n_memorable_points_sub is not None and n_memorable_points_sub < (i+1) * inputs.shape[0]:
                    break
                inputs = inputs.to(device)
                means.append(model(inputs))
            return torch.cat(means)  # (n, c)
        else:
            if idx is not None:
                return model(memorable_points[idx, :].to(device))
            else:
                return model(memorable_points.to(device))

    def get_penalty(self, model, n_memorable_points_sub=None, idx=None, use_kprior_penalty=False):
        assert self.mean is not None
        kernel_inv = self.kernel_inv  # None or (nc, nc) or (c, n, n)
        
        current_mean = self._evaluate_mean(model, n_memorable_points_sub, idx)  # (n, c)
        mean = self.mean[idx] if idx is not None else self.mean  # (n, c)
        mean = mean.to(current_mean.device)

        if use_kprior_penalty:
            return cross_entropy_with_probs(current_mean, mean)

        b = current_mean - mean     # (n, c)
        if kernel_inv is None:
            # kernel_inv == identity matrix
            b = b.flatten()  # (nc,)
            v = b  # (nc,)
        elif kernel_inv.ndim == 2:
            # kernel_inv: (nc, nc)
            b = b.flatten()  # (nc,)
            v = torch.mv(kernel_inv, b)  # (nc,)
        else:
            # kernel_inv: (c, n, n)
            b = b.transpose(0, 1).unsqueeze(2)  # (c, n, 1)
            v = torch.matmul(kernel_inv, b)  # (c, n, 1)
            v = v.transpose(0, 1).flatten()  # (nc,)
            b = b.flatten()  # (nc,)

        return torch.dot(b, v)


def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted

    Source: https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
    """

    assert input.shape == target.shape, "Inputs and targets must have same shape!"

    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


class FROMP:
    """
    Implementation of a functional-regularisation method called
    Functional Regularisation of Memorable Past (FROMP):
    Pingbo Pan et al., 2020
    Continual Deep Learning by Functional Regularisation of Memorable Past
    https://arxiv.org/abs/2004.14070

    Example::

        >>> import torch
        >>> from asdfghjkl import FROMP
        >>>
        >>> model = torch.nn.Linear(5, 3)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> fr = FROMP(model, tau=1.)
        >>>
        >>> for data_loader in data_loader_list:
        >>>     for x, y in data_loader:
        >>>         optimizer.zero_grad()
        >>>         loss = loss_fn(model(x), y)
        >>>         if fr.is_ready:
        >>>             loss += fr.get_penalty()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     fr.update_regularization_info(data_loader)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 tau=1.,
                 temp=1.,
                 eps=1e-5,
                 max_tasks_for_penalty=None,
                 n_memorable_points=10,
                 memorable_points_frac=None,
                 n_memorable_points_sub=10,
                 memory_select_method="lambda_descend",
                 memory_loss_mode="soft_all",
                 memory_residual_frac=1.0,
                 use_nn_error_correction=False,
                 ggn_shape='diag',
                 ggn_type='exact',
                 prior_prec=1e-5,
                 n_mc_samples=1,
                 kernel_type='implicit',
                 use_identity_kernel=False,
                 use_temp_correction=False,
                 penalty_type='fromp',
                 ):
        assert ggn_type in _fisher_types, f'ggn_type: {ggn_type} is not supported.' \
                                          f' choices: {list(_fisher_types.keys())}'
        assert ggn_shape in _precond_classes, f'ggn_shape: {ggn_shape} is not supported.' \
                                              f' choices: {list(_precond_classes.keys())}'
        assert kernel_type in _kernel_fns, f'kernel_type: {kernel_type} is not supported.' \
                                           f' choices: {list(_kernel_fns.keys())}'

        self.model = model
        self.tau = tau
        self.temp = temp
        self.eps = eps
        self.max_tasks_for_penalty = max_tasks_for_penalty
        self.n_memorable_points = n_memorable_points
        self.memorable_points_frac = memorable_points_frac
        self.n_memorable_points_sub = None if (not n_memorable_points or n_memorable_points <= n_memorable_points_sub) else n_memorable_points_sub
        self.memory_select_method = memory_select_method
        self.memory_loss_mode = memory_loss_mode
        self.memory_residual_frac = memory_residual_frac
        self.use_nn_error_correction = use_nn_error_correction
        self.use_identity_kernel = use_identity_kernel
        self.use_temp_correction = use_temp_correction
        self.penalty_type = penalty_type

        if isinstance(model, DDP):
            # As DDP disables hook functions required for Fisher calculation,
            # the underlying module will be used instead.
            model_precond = model.module
        else:
            model_precond = model
        self.precond = _precond_classes[ggn_shape](model_precond,
                                                   fisher_type=_fisher_types[ggn_type],
                                                   pre_inv_postfix='all_tasks_ggn',
                                                   n_mc_samples=n_mc_samples,
                                                   damping=prior_prec)
        self.kernel_fn = get_preconditioned_kernel_fn(_kernel_fns[kernel_type], self.precond)
        self.observed_tasks: List[PastTask] = []

    @property
    def is_ready(self):
        return len(self.observed_tasks) > 0

    def update_regularization_info(self,
                                   data_loader: DataLoader,
                                   class_ids: List[int] = None,
                                   memorable_points_as_tensor=True,
                                   is_distributed=False,
                                   empty_gpu_cache_=False):
        model = self.model
        if isinstance(model, DDP):
            # As DDP disables hook functions required for Kernel calculation,
            # the underlying module will be used instead.
            model = model.module
        model.eval()

        if self.penalty_type == 'fromp' and not self.use_identity_kernel:
            # update GGN and inverse for the current task
            with customize_head(model, class_ids):
                self.precond.update_curvature(data_loader=data_loader)
            if is_distributed:
                self.precond.reduce_curvature()
            self.precond.accumulate_curvature(to_pre_inv=True)
            self.precond.update_inv()

        # register the current task with the memorable points
        with customize_head(model, class_ids):
            (memorable_points,
            memorable_points_indices,
            memorable_points_indices_global,
            memorable_points_true_targets,
            memorable_points_types) = collect_memorable_points(model,
                                                        data_loader,
                                                        self.n_memorable_points,
                                                        self.memorable_points_frac,
                                                        self.memory_select_method,
                                                        self.memory_residual_frac,
                                                        self.use_nn_error_correction,
                                                        memorable_points_as_tensor,
                                                        is_distributed,
                                                        self.n_memorable_points_sub)

        self.observed_tasks.append(PastTask(memorable_points,
                                            class_ids,
                                            memorable_points_indices,
                                            memorable_points_indices_global,
                                            memorable_points_true_targets,
                                            memorable_points_types))

        # update information (kernel & mean) for each observed task
        for i, task in enumerate(self.observed_tasks):
            with customize_head(model, task.class_ids, softmax=self.penalty_type!='der', temp=self.temp):
                if self.penalty_type == 'fromp' and not self.use_identity_kernel:
                    task.update_kernel(model, self.kernel_fn, self.eps)
                if empty_gpu_cache_:
                    empty_gpu_cache(f"pre task.update_mean for task #{i+1}")
                task.update_mean(model, memory_loss_mode=self.memory_loss_mode, memory_residual_frac=self.memory_residual_frac)
                if empty_gpu_cache_:
                    empty_gpu_cache(f"post task.update_mean for task #{i+1}")

    def get_penalty(self, tau=None, temp=None, max_tasks=None, mem_indices=None, use_kprior_penalty=False):
        assert self.is_ready, 'Functional regularization is not ready yet, ' \
                              'call FROMP.update_regularization_info(data_loader).'
        if tau is None:
            tau = self.tau
        if temp is None:
            temp = self.temp
        if max_tasks is None:
            max_tasks = self.max_tasks_for_penalty
        model = self.model
        model.eval()
        observed_tasks = self.observed_tasks

        # collect indices of tasks to calculate regularization penalty
        n_observed_tasks = len(observed_tasks)
        indices = list(range(n_observed_tasks))
        if max_tasks and max_tasks < n_observed_tasks:
            import random
            indices = random.sample(indices, max_tasks)

        # get regularization penalty on all the selected tasks
        with disable_broadcast_buffers(model):
            total_penalty = 0
            for idx in indices:
                task = observed_tasks[idx]
                with customize_head(model, task.class_ids, softmax=self.penalty_type!='der', temp=temp):
                    total_penalty += task.get_penalty(model, self.n_memorable_points_sub, mem_indices, use_kprior_penalty)

        temp_corr = temp**2 if self.use_temp_correction else 1.

        return 0.5 * tau * temp_corr * total_penalty


@torch.no_grad()
def collect_memorable_points(model,
                             data_loader: DataLoader,
                             n_memorable_points,
                             memorable_points_frac,
                             select_method="lambda_descend",
                             memory_residual_frac=1.0,
                             use_nn_error_correction=False,
                             as_tensor=True,
                             is_distributed=False,
                             n_memorable_points_sub=None):
    device = next(model.parameters()).device
    dataset = data_loader.dataset

    assert data_loader.batch_size is not None, 'DataLoader w/o batch_size is not supported.'
    if is_distributed:
        indices = range(dist.get_rank(), len(dataset), dist.get_world_size())
        dataset = Subset(dataset, indices)

    assert select_method in ['lambda_descend', 'random', 'lambda_descend_global', 'random_global'], \
        'Invalid memorable points selection method.'

    n_task_data = len(dataset.get_hard_task_targets()) if hasattr(dataset, 'task_indices') else len(dataset)
    if n_memorable_points is None:
        n_memorable_points = int(memorable_points_frac * n_task_data)
    if use_nn_error_correction and memory_residual_frac > 0:
        n_error_correction_points = int(n_memorable_points * memory_residual_frac)
        n_memorable_points -= n_error_correction_points

    memorable_points_kwargs = dict(model=model, data_loader=data_loader, dataset=dataset, device=device,
                                    n_memorable_points=n_memorable_points, select_method=select_method)
    if n_memorable_points >= n_task_data:
        # Use ALL data points as memorable points
        memorable_points_indices = list(range(n_task_data))
    elif 'global' in select_method:
        memorable_points_indices = _collect_memorable_points(**memorable_points_kwargs)
    else:
        memorable_points_indices = _collect_memorable_points_class_balanced(**memorable_points_kwargs)
    memorable_points_types = [TYPE_MEMORABLE_PAST] * n_memorable_points

    # append points for NN error correction
    if use_nn_error_correction and memory_residual_frac > 0:
        print(f"Collecting {n_error_correction_points} highest-residual points for error correction (+{n_memorable_points} memory points)...")
        error_correction_points_indices = _collect_error_correction_points(**memorable_points_kwargs, n_error_correction_points=n_error_correction_points, correction_select_method='random')
        memorable_points_indices += error_correction_points_indices
        memorable_points_types += [TYPE_ERROR_CORRETION] * n_error_correction_points

    # Convert within-task (i.e. starting at 0) to across-task memorable points indices
    memorable_points_indices_global = [dataset.globalize_memory_index(idx) for idx in memorable_points_indices]

    if as_tensor:
        # create a Tensor for memorable points on model's device
        memorable_points = [dataset[idx][0] for idx in memorable_points_indices_global]
        memorable_points = torch.stack(memorable_points).to(device)
    else:
        # create a DataLoader for memorable points
        memorable_points = Subset(dataset, memorable_points_indices_global)
        if n_memorable_points_sub is not None:
            batch_size = n_memorable_points_sub
        else:
            batch_size = min(n_memorable_points, data_loader.batch_size)
        memorable_points = DataLoader(memorable_points, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False)

    memorable_points_true_targets = [dataset[idx][1] for idx in memorable_points_indices_global]
    return memorable_points, memorable_points_indices, memorable_points_indices_global, memorable_points_true_targets, memorable_points_types


def _collect_memorable_points_class_balanced(model, data_loader, dataset, device, n_memorable_points, select_method):
    """ collect memorable points (class-balanced) """

    # extract dataset targets
    if hasattr(dataset, 'task_indices'):
        targets = torch.tensor(dataset.get_hard_task_targets())
    elif hasattr(dataset, 'targets'):
        targets = torch.tensor(dataset.targets)
    else:
        targets = torch.tensor([d[1] for d in dataset])

    # define number of memorable points per class
    n_classes = len(targets.unique())
    n_memorable_points_per_class = int(n_memorable_points / n_classes)

    if select_method == 'lambda_descend':
        # compute Hessian traces
        hessian_traces = _compute_dataset_scores(model, data_loader, dataset, device, 'lambda_descend')

    # for each class, select a uniformly random subset of data points
    memorable_points_indices = []
    for cls in targets.unique():
        class_indices = (targets == cls).nonzero(as_tuple=False).flatten()

        if select_method == 'lambda_descend':
            # sort indices by Hessian trace (for current class) 
            select_indices = torch.argsort(hessian_traces[class_indices], descending=True)
        else:
            # obtain uniformly random indices (for current class)
            import numpy as np
            select_indices = np.random.permutation(len(class_indices))

        memorable_points_indices.append(class_indices[select_indices[:n_memorable_points_per_class]])

    return torch.cat(memorable_points_indices).tolist()


def _collect_memorable_points(model, data_loader, dataset, device, n_memorable_points, select_method):
    """ collect memorable points (not class-balanced) """

    if select_method == 'lambda_descend_global':
        # sort indices by Hessian trace (across full dataset)
        hessian_traces = _compute_dataset_scores(model, data_loader, dataset, device, 'lambda_descend')
        select_indices = torch.argsort(hessian_traces, descending=True)
    else:
        # obtain uniformly random indices (across full dataset)
        import numpy as np
        select_indices = np.random.permutation(len(dataset))

    return select_indices[:n_memorable_points].tolist()


def _collect_error_correction_points(model, data_loader, dataset, device, n_memorable_points, select_method, n_error_correction_points, correction_select_method):
    """ collect points for NN error correction (not class-balanced) """

    if correction_select_method == 'residual_descend':
        # sort indices by residuals (across full dataset)
        residuals = _compute_dataset_scores(model, data_loader, dataset, device, 'residual_descend')
        select_indices = torch.argsort(residuals, descending=True)
    elif correction_select_method == 'error_descend':
        # sort indices by errors, i.e. logits times residuals (across full dataset)
        errors = _compute_dataset_scores(model, data_loader, dataset, device, 'error_descend')
        select_indices = torch.argsort(errors, descending=True)
    else:
        # obtain uniformly random indices (across full dataset)
        import numpy as np
        select_indices = np.random.permutation(len(dataset))

    return select_indices[:n_error_correction_points].tolist()


def _compute_dataset_scores(model, data_loader, dataset, device, scoring_method):
    """ compute scores for selecting memorable points with the given method """

    assert scoring_method in ['lambda_descend', 'residual_descend', 'error_descend']

    # create a data loader w/o shuffling so that indices in the dataset are stored
    no_shuffle_loader = DataLoader(dataset,
                                   batch_size=data_loader.batch_size,
                                   num_workers=data_loader.num_workers,
                                   pin_memory=True,
                                   drop_last=False,
                                   shuffle=False)
    # collect scores
    all_scores = []
    for batch in no_shuffle_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)  # (n, c)
        probs = F.softmax(logits, dim=1) # (n, c)
        if scoring_method == 'lambda_descend':
            scores = _compute_hessian_traces(probs)
        elif scoring_method == 'residual_descend':
            scores = _compute_residuals(probs, targets)
        elif scoring_method == 'error_descend':
            scores = _compute_errors(logits, probs, targets)
        all_scores.append(scores)  # [(n,)]

    return torch.cat(all_scores).cpu()


def _compute_hessian_traces(probs):
    """ compute Hessian traces for selecting memorable points using the lambda_descend method """

    diag_hessian = probs - probs * probs    # (n, c)
    return diag_hessian.sum(dim=1)          # (n,)


def _compute_residuals(probs, targets):
    """ compute residuals for selecting memorable points for NN error correction """

    targets_one_hot = _convert_targets_to_one_hot(targets, probs.shape[1])  # (n, c)
    return (probs - targets_one_hot.to(probs.device)).abs().sum(dim=1)   # (n,)


def _compute_errors(logits, probs, targets):
    """ compute errors (i.e. logits times residuals) for selecting memorable points for NN error correction """

    targets_one_hot = _convert_targets_to_one_hot(targets, probs.shape[1])  # (n, c)
    residuals = (probs - targets_one_hot.to(probs.device)).unsqueeze(2)    # (n, c, 1)
    logits = logits.unsqueeze(1)    # (n, 1, c)
    return torch.bmm(logits, residuals).flatten() # (n,)


def _convert_targets_to_one_hot(targets, one_hot_len):
    targets_one_hot = []
    for target in targets:
        target_one_hot = torch.zeros(one_hot_len)
        target_one_hot[target] = 1.
        targets_one_hot.append(target_one_hot)
    return torch.stack(targets_one_hot)


@contextmanager
def customize_head(module: torch.nn.Module, class_ids: List[int] = None, softmax=False, temp=1.):

    def forward_hook(module, input, output):
        output /= temp
        if class_ids is not None:
            output = output[:, class_ids]
        if softmax:
            return F.softmax(output, dim=1)
        else:
            return output

    handle = module.register_forward_hook(forward_hook)
    yield
    handle.remove()
    del forward_hook


@contextmanager
def disable_broadcast_buffers(module):
    tmp = False
    if isinstance(module, DDP):
        tmp = module.broadcast_buffers
        module.broadcast_buffers = False
    yield
    if isinstance(module, DDP):
        module.broadcast_buffers = tmp


def empty_gpu_cache(name):
	import torch
	import subprocess
	
	print(f"Emptying GPU cache ({name})...")
	
	print(f"\tbefore:")
	print(subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout)
	
	torch.cuda.empty_cache()
	
	print(f"\tafter:")
	print(subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout)
