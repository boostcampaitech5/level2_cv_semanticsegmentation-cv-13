import torch 
import torch.nn as nn 
import torch.nn.functional as F 

_criterion_entrypoints = {
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
}


def create_criterion(criterion_name, **kwargs): 
    if criterion_name in _criterion_entrypoints: 
        create_fn = _criterion_entrypoints[criterion_name] 
        criterion = create_fn(**kwargs)
    else: 
        return RuntimeError('Unknown loss (%s)' % criterion_name) 
    return criterion 