# Contains utilities for handling gpu kernel
function check_launch(n, p, q; T=Float32, throw_error=true)
    max_threads_per_block = T==Float32 ? 1024 : 256

    isgood = true

    if p > n; isgood = false; throw_error && error("p must be less than or equal to n"); end
    if p*q >= max_threads_per_block; isgood = false; throw_error && error("p*q must be less than $max_threads_per_block"); end
    if q > p; isgood = false; throw_error && error("q must be less than or equal to p"); end
    if n % p != 0; isgood = false; throw_error && error("n must be divisible by p"); end
    if p % q != 0; isgood = false; throw_error && error("p must be divisible by q"); end

    return isgood
end

function get_launch_config(nt; T=Float32, p_max=256)
    divs_n = divisors(nt)
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_n)
        if div <= p_max
            p = div
            ip = i
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    if nt <= 2^13
        divs_p = divs_n
        for i in 1:length(divs_n)
            for j in 1:length(divs_n)
                isgood = check_launch(nt, divs_n[i], divs_p[j]; T=T, throw_error=false)
                if isgood && (divs_n[i] <= p_max)
                    # Check if this is the max achievable ij value
                    # in the p, q choice matrix
                    obj_val = i_weight*i+j_weight*j
                    if obj_val >= max_ij
                        max_ij = obj_val
                        p = divs_n[i]
                        q = divs_p[j]
                    end
                end
            end
        end
    end

    return p, q
end
