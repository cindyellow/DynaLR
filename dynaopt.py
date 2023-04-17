import torch
import torch.nn as nn
import math

class DynamicOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, debug=False, lr_choices=[1e-6, 1e-5, 1e-1, 1]):
        defaults = dict(lr=lr, debug=debug, lr_choices=lr_choices)
        super(DynamicOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        group = self.param_groups[0]
        lr = group['lr']
        debug = group['debug']
        lr_choices = group['lr_choices']

        if closure==None:
            raise Exception('DynamicOptimizer requires closure.')
        
        with torch.enable_grad():
            # get loss using current lr and compute gradient
            loss = closure()
            loss.backward()
        
        best_loss = loss
        best_lr = lr

        # using for loop, eval f and check against condition
        for choice in lr_choices:
            group['lr'] = choice
            # TODO: check how to access original inputs?
            loss_next = closure()

            if loss_next < best_loss:
                if debug:
                    print("Found better learning rate.")
                best_loss = loss_next
                best_lr = choice
        
        # update params
        # TODO: check if this links properly
        group['lr'] = best_lr

        return best_loss