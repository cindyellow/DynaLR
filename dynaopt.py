import torch
import torch.nn as nn
import math
import copy
import time

class DynamicOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, debug=False, lr_choices=[1e-4, 0.1]):
        defaults = dict(lr=lr, debug=debug, lr_choices=lr_choices + [lr])
        super(DynamicOptimizer, self).__init__(params, defaults)

    def try_update(self, lr, params, curr_params, curr_grad):
        zipped = zip(params, curr_params, curr_grad)
        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - lr * g_current
        # print(type(curr_grad), type(lr))
        # params.data = curr_params - 2*curr_grad

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
        params = group['params']

        curr_params = copy.deepcopy(params)
        curr_grad = [p.grad for p in params]
        
        if closure==None:
            raise Exception('DynamicOptimizer requires closure.')
        
        with torch.enable_grad():
            # get loss using current lr and compute gradient
            loss = closure()
            loss.backward()
        
        best_loss = loss
        best_lr = lr
        best_params = params

        # using for loop, eval f and check against condition
        for choice in lr_choices:
            self.try_update(choice, params, curr_params, curr_grad)
            loss_next = closure()

            if loss_next < best_loss:
                if debug:
                    print("Found better learning rate {} with loss {}".format(best_lr, best_loss))
                best_loss = loss_next
                best_lr = choice
                best_params = params
            # reset params
            params = curr_params

        # update lr
        # TODO: check if this links properly
        group['lr'] = best_lr
        group['params'] = best_params

        return best_loss