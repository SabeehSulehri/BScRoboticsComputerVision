import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
                
   ## code in c             
   # {
    # int x,y,p,q,i,j,catch,out[100000][100000];
    # for(p=0;p<=Hi;p++)
        # for(q=0;q<=Hi;q++)
            # for(i=0;i<=Hk;i++)
                # for(j=0;j<=k;j++)
                    # x=(p+Hk/2)-i;
                    # y=(q+Wk/2)-j;
                    # catch =0;
                    # if(x>=0 && y>=0 && x<Hi && y < Wi;
                        # catch = image[x][y] * kernel[i][j]
                        # out[p][q] = out[p][q] + catch

    
    
    for p in range(0, Hi):
        for q in range(0, Wi):
            for i in range(Hk):
                for j in range(Wk):
                    x , y = (p + Hk // 2 - i, q + Wk // 2 - j)
                    catch = 0
                    # print('kernel[', i, '][', j, ']*', 'image[', x, ']', '[', y, ']')
                    if x >= 0 and y >= 0 and x < Hi and y < Wi:
                        catch = image[x][y] * kernel[i][j]
                    out[p][q] = out[p][q] + catch
                    
    
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    out = np.zeros((H + (2 * pad_height), W + (2 * pad_width)))
    # making a zero matrix after adding the padding
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    # replaces the selected matrix values with the image matrix
    
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    flip_kernel = np.flip(np.flip(kernel, 1), 0)
    
    padding_height = Hk // 2
    padding_width = Wk // 2
    
    padding_image = zero_pad(image, padding_height, padding_width)
    
    for m in range(padding_height, padding_height + Hi):
        for n in range(padding_width, padding_width + Wi):
            value = np.sum(
                padding_image[m - padding_height:m + (Hk - padding_height), n - padding_width:n + (Wk - padding_width)] * flip_kernel)
            out[ m - padding_height, n - padding_width] = value
    
    
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    flip_g = np.flip(np.flip(g, 1), 0)
    out = conv_fast(f, flip_g)
    ## convolution without the flip
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    gm = np.mean(g)
    ## to get the mean value
    ## and then we remove it from g to get the anti-flip convulation of g
    out = cross_correlation(f, g - gm)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros((Hf, Wf))

    padding_height = Hg // 2
    padding_width = Wg // 2
    
    padding_f = zero_pad(f, padding_height, padding_width)

    gm = np.mean(g)
    g_standard = np.std(g)
    g_normal = (g - gm) / g_standard

    for m in range(padding_height, padding_height + Hf):
        for n in range(padding_width, padding_width + Wf):
            patch_f = padding_f[m - padding_height:m + (Hg - padding_height), n - padding_width:n + (Wg - padding_width)]
            patch_fm = np.mean(patch_f)
            patch_f_standard = np.std(patch_f)
            patch_normal = (patch_f - patch_fm) / patch_f_standard
            value = np.sum(patch_normal * g_normal)
            out[m - padding_height, n - padding_width] = value
    ### END YOUR CODE
    pass
    ### END YOUR CODE

    return out
