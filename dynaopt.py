import torch
from functools import reduce

class DynamicOptimizer(torch.optim.Optimizer):
    """
    Implements custom dynamic learning rate scheduling.

    Arguments:
        - lr (float): base learning rate. (default=1e-3)
        - lr_choices ([float]): list of learning rates to try for every step. (default=[1e-4,0.1])
        - debug (bool): flag for debug mode. (default=False)
    """

    def __init__(self, params, lr=1e-3, lr_choices=[1e-4,0.1], debug=False):
        """
        Initializes the optimizer class.
        """
        defaults = dict(lr=lr, debug=debug, lr_choices=lr_choices + [lr])
        super(DynamicOptimizer, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.eval_count = 0

    def _numel(self):
        """
        Calculate the number of elements in params. From: https://github.com/nlesc-dirac/pytorch/blob/124a693b713c58bb6475db53090d4a5c83194653/lbfgsnew.py
        """
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _make_copy(self):
        """
        Make a deep copy of self._params. From: https://github.com/nlesc-dirac/pytorch/blob/124a693b713c58bb6475db53090d4a5c83194653/lbfgsnew.py
        """
        offset = 0
        new_params = []
        for p in self._params:
            numel = p.numel()
            new_param1=p.data.clone().contiguous().view(-1)
            offset += numel
            new_params.append(new_param1)
        assert offset == self._numel()
        return torch.cat(new_params,0)

    def _reset_params(self, new_params):
        """
        Reset self._params with newparams. From: https://github.com/nlesc-dirac/pytorch/blob/124a693b713c58bb6475db53090d4a5c83194653/lbfgsnew.py
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(new_params[offset:offset+numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _try_update(self, lr, curr_params, curr_grad):
        """
        Try an update for self._params using learning rate lr,
        current parameters and gradient curr_params and curr_grad.
        """
        zipped = zip(self._params, curr_params, curr_grad)
        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - lr * g_current

    def _search_lr(self, closure, curr_params, curr_grad, curr_lr, lr_choices, debug):
        """
        Return the lr that gives the lowest loss out of all learning rates in 
        lr_choices.
        """
        # initialize
        best_lr = curr_lr
        best_loss = float(closure())

        # try each lr choice
        for choice in lr_choices:
            # update params
            self._try_update(choice, curr_params, curr_grad)
            # get loss
            curr_loss = float(closure())
            if curr_loss < best_loss:
                best_lr = choice
                best_loss = curr_loss
            # reset params
            self._reset_params(curr_params)
        
        return best_lr, best_loss

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # check if closure is valid
        if closure==None:
            raise Exception('DynamicOptimizer requires closure.')

        # save relevant parameters
        group = self.param_groups[0]
        lr = group['lr']
        debug = group['debug']
        lr_choices = group['lr_choices']

        # save a copy of current params and gradient
        curr_params = self._make_copy()
        curr_grad = [p.grad for p in self._params]

        # calculate best lr and associated val loss
        with torch.no_grad():
            best_lr, best_loss = self._search_lr(closure, curr_params, curr_grad, lr, lr_choices, debug)
        
        if debug:
            self.eval_count += 1
            if self.eval_count % 80 == 79:
                print("Found best lr {} for batch {}.".format(best_lr, self.eval_count))

        # update lr and params
        group['lr'] = best_lr
        self._try_update(best_lr, curr_params, curr_grad)

        return best_loss