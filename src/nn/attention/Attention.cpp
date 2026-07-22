#include "nn/attention/Attention.hpp"

#include "nn/Linear.hpp"

Attention::Attention(
    int64_t query_dim,
    int64_t heads,
    int64_t dim_head,
    float dropout
)
    : heads_(heads),
      dim_head_(dim_head > 0 ? dim_head : query_dim / heads),
      inner_dim_(heads_ * dim_head_)
{
    modules["to_q"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            true
        );

    modules["to_k"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            true
        );

    modules["to_v"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            true
        );


    /*
       Diffusers:
       to_out = ModuleList([
           Linear(inner_dim, query_dim),
           Dropout()
       ])
    */

    modules["to_out.0"] =
        std::make_shared<Linear>(
            inner_dim_,
            query_dim,
            true
        );
}

Tensor Attention::forward(
    ggml_context* ctx,
    Tensor hidden_states
)
{
    auto shape = hidden_states.shape();

    const int64_t batch = shape[0];
    const int64_t channels = shape[1];
    const int64_t height = shape[2];
    const int64_t width = shape[3];

    const int64_t tokens = height * width;


    /*
        NCHW -> B,HW,C
    */

    auto x =
        hidden_states
            .permute({0, 2, 3, 1})
            .reshape({batch, tokens, channels});


    auto q =
        std::static_pointer_cast<Linear>(
            modules["to_q"])
        ->forward(ctx, x);

    auto k =
        std::static_pointer_cast<Linear>(
            modules["to_k"])
        ->forward(ctx, x);

    auto v =
        std::static_pointer_cast<Linear>(
            modules["to_v"])
        ->forward(ctx, x);


    /*
        Split heads:
        B,T,C -> B,H,T,D
    */

    q = q.reshape({
        batch,
        tokens,
        heads_,
        dim_head_
    })
    .permute({
        0,
        2,
        1,
        3
    });


    k = k.reshape({
        batch,
        tokens,
        heads_,
        dim_head_
    })
    .permute({
        0,
        2,
        3,
        1
    });


    v = v.reshape({
        batch,
        tokens,
        heads_,
        dim_head_
    })
    .permute({
        0,
        2,
        1,
        3
    });


    /*
        Attention scores:

        [B,H,T,D] x [B,H,D,T]

        -> [B,H,T,T]
    */

    auto scores =
        ggml_mul_mat(
            ctx,
            *k,
            *q
        );


    scores =
        ggml_scale(
            ctx,
            scores,
            1.0f / std::sqrt(
                (float)dim_head_
            )
        );


    scores =
        ggml_soft_max(
            ctx,
            scores
        );


    /*
        Attention output:

        [B,H,T,T] x [B,H,T,D]

        -> [B,H,T,D]
    */

    auto output =
        ggml_mul_mat(
            ctx,
            scores,
            *v
        );


    Tensor out(
        ctx,
        output/* TODO: determine shape
        output.shape()*/
    );


    /*
        Merge heads:

        B,H,T,D
        ->
        B,T,C
    */

    out =
        out.permute({
            0,
            2,
            1,
            3
        })
        .reshape({
            batch,
            tokens,
            inner_dim_
        });


    out =
        std::static_pointer_cast<Linear>(
            modules["to_out.0"])
        ->forward(
            ctx,
            out
        );


    /*
        B,T,C -> B,C,H,W
    */

    return out
        .reshape({
            batch,
            height,
            width,
            channels
        })
        .permute({
            0,
            3,
            1,
            2
        });
}