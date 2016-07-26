uniform_order_statistic_cdf <- function(i, n, x)
    # uniform_order_statistic_cdf(i, n, x) -> Pr[U_(i) < x]
    # 
    # Let U_1, ..., U_n ~ Uniform[0,1] be n independent random variables
    # and let U_(1) < ... < U_(n) denote the same variables in sorted order.  Then
    #     U_{(i)} ~ Beta(i, n-i+1)
    # 
    # This function returns the Cumulative Distribution function of U_(i),
    # i.e. the return value is Pr[U_(i) < x]
    # 
    # This function also works for array inputs
{
    return(pbeta(x, i, n-i+1));
};

compute_p_values <- function(uniform01_samples)
{
    stopifnot(all((uniform01_samples >= 0) & (uniform01_samples <= 1)))
    n <- length(uniform01_samples)
    sorted_uniform01_samples <- sort(uniform01_samples)
    p_values <- uniform_order_statistic_cdf(1:n, n, sorted_uniform01_samples)
    return(p_values)
}

Mn <- function(uniform01_samples)
    # Mn(uniform01_samples) -> Mn statistic (two sided)
    # 
    # Compute the two sided Mn test statistic for a list of samples assumed
    # to come from a known (and fully specified) continuous distribution F.
    # 
    # Args:
    #     uniform01_samples - data samples after being transformed using
    #     the cumulative distribution of the null hypothesis.
    #     This should be a list of numbers in the range 0 to 1.
    # 
    # Returns:
    #     The two-sided Mn statistic defined in Eq. (3.2) of the paper
    #         http://arxiv.org/pdf/1311.3190v5.pdf
{
    p_values <- compute_p_values(uniform01_samples) 
    return(min(min(p_values), 1.0-max(p_values)))
}

Mn_plus <- function(uniform01_samples)
    # Mn_plus(uniform01_samples) -> Mn+ score (one-sided sided)
    # 
    # Like the Mn() function, but for testing deviations towards lower values only.
{
    p_values <- compute_p_values(uniform01_samples) 
    return(min(p_values))
}

Mn_minus <- function(uniform01_samples)
    # Mn_minus(uniform01_samples) -> Mn- score (one-sided sided)
    # 
    # Like the Mn() function, but for testing deviations towards higher values only.
{
    p_values <- compute_p_values(uniform01_samples) 
    return(1-max(p_values))
}
