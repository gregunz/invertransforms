Module invertransforms.linear_transformation
============================================

Classes
-------

`LinearTransformation(transformation_matrix, mean_vector)`
:   Transform a tensor image with a square transformation matrix and a mean_vector computed
    offline.
    Given transformation_matrix and mean_vector, will flatten the torch.*Tensor and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.
    
    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
        perform SVD on this matrix and pass it as transformation_matrix.
    
    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
        mean_vector (Tensor): tensor [D], D = C x H x W

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.LinearTransformation
    * invertransforms.util.invertible.Invertible