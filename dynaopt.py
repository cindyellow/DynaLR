import torch
from functools import reduce
from copy import deepcopy
import math


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
        self.eval_count = 0

    def _make_copy(self):
        """
        Make a deep copy of self._params.
        """
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _reset_params(self, new_params):
        """
        Reset self._params with newparams. 
        Adapted from: https://github.com/nlesc-dirac/pytorch/blob/124a693b713c58bb6475db53090d4a5c83194653/lbfgsnew.py
        """
        for p, p_current in zip(self._params, new_params):
            p.data.copy_(p_current)

    def _try_update(self, lr, curr_params, curr_grad):
        """
        Try an update for self._params using learning rate lr,
        current parameters and gradient curr_params and curr_grad.
        """
        # p.data -= g_current*lr
        for p, g_current in zip(self._params, curr_grad):
            p.data.sub_(g_current.data*lr)

    def _search_lr(self, closure, curr_params, curr_grad, lr_choices, debug):
        """
        Return the lr that gives the lowest loss out of all learning rates in 
        lr_choices.
        """
        # initialize
        best_lr = -1
        best_loss = math.inf

        # try each lr choice
        for choice in lr_choices:
            # reset params
            self._reset_params(curr_params)
            # update params
            self._try_update(choice, curr_params, curr_grad)
            # get loss
            curr_loss = float(closure())
            if curr_loss < best_loss:
                best_lr = choice
                best_loss = curr_loss
        
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
        debug = group['debug']
        lr_choices = group['lr_choices']

        # save a copy of current params and gradient
        curr_params = self._make_copy()
        curr_grad = [p.grad for p in self._params]

        # calculate best lr and associated val loss
        with torch.no_grad():
            best_lr, best_loss = self._search_lr(closure, curr_params, curr_grad, lr_choices, debug)

        # reset before updating lr and params
        self._reset_params(curr_params)
        self._try_update(best_lr, curr_params, curr_grad)

        if debug:
            self.eval_count += 1
            if self.eval_count % 80 == 79:
                print("Found best lr {} for batch {}.".format(best_lr, self.eval_count))
            # print(self._params[0].shape, curr_params[0].shape)
            # print("Are equal: ", torch.eq(self._params[0], curr_params[0]))

        return best_loss