Module invertransforms.functions
================================

Classes
-------

`Identity(*args, **kwargs)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible

`Lambda(lambd, tf_inv=None, repr_str=None)`
:   Apply a user-defined lambda as a transform.
    
    Args:
        lambd (function): Lambda/function to be used for transform.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Lambda
    * invertransforms.util.invertible.Invertible

`TransformIf(transform, condition)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible