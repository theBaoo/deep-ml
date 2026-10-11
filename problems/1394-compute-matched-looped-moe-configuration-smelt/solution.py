import math


def block_flops(d, d_ff, top_k):
    # TODO: 4*d*d + top_k*3*d*d_ff
    return 4*d*d + top_k*3*d*d_ff


def block_params(d, d_ff, n_experts):
    # TODO: 4*d*d + n_experts*3*d*d_ff
    return 4*d*d + n_experts*3*d*d_ff


def smelt_match(d, n_blocks, n_experts, top_k, ff_mult, kv_heads, head_dim, loop_fraction=0.5):
    # TODO: applications, narrowed width (multiple of 8), experts to restore params, fewer kv heads
    # TODO: return dict with d_new, d_ff_new, experts_new, kv_heads_new, applications, flops_ratio, params_ratio, kv_ratio
    lam = 1 + loop_fraction
    applications = round(n_blocks * lam)

    d_new = math.floor(d / math.sqrt(lam))
    d_new = (d_new // 8) * 8

    d_ff = ff_mult * d
    d_ff_new = ff_mult * d_new

    base_params = block_params(d, d_ff, n_experts)

    experts_new = 1

    while (
        block_params(d_new, d_ff_new, experts_new) < base_params
    ):
        experts_new += 1

    kv_heads_new = max(1, round(kv_heads / lam))

    base_flops = n_blocks * block_flops(
        d, d_ff, top_k
    )

    looped_flops = applications * block_flops(
        d_new, d_ff_new, top_k
    )

    flops_ratio = round(
        looped_flops / base_flops, 4
    )

    params_ratio = round(
        block_params(d_new, d_ff_new, experts_new) / base_params, 4
    )

    base_kv = n_blocks * 2 * kv_heads * head_dim

    looped_kv = (
        applications * 2 * kv_heads_new * head_dim
    )

    kv_ratio = round(looped_kv / base_kv, 4)

    return {

        "d_new": d_new,

        "d_ff_new": d_ff_new,

        "experts_new": experts_new,

        "kv_heads_new": kv_heads_new,

        "applications": applications,

        "flops_ratio": flops_ratio,

        "params_ratio": params_ratio,

        "kv_ratio": kv_ratio,

    }
