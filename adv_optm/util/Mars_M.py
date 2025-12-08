def approx_mars_correction(mars_gamma, beta1, current_grad, last_grad):
    '''
    Applies the logic of an approximated Variance reduction technique (MARS-M)
    proposed in the paper "MARS-M: When Variance Reduction Meets Matrices":
    https://arxiv.org/abs/2510.21800
    Formula: c_t = g_t + gamma * beta / (1 - beta) * (g_t - g_{t-1})
    '''

    mars_factor = mars_gamma * beta1 / (1.0 - beta1)

    # Compute corrected gradient c_t
    # c_t = grad + mars_factor * (grad - last_grad)
    # NOTE: The original MARS-M uses internal grad norm clip, but I don't think it's
    # necessary, external grad norm clip should do the job.
    corrected_grad = current_grad.sub(last_grad).mul_(mars_factor).add_(current_grad)

    return corrected_grad