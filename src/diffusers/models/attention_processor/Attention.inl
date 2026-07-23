#include "diffusers/models/attention_processor/Attention.hpp"
#include "nn/Linear.hpp"

template <class AttnOp>
Attention<AttnOp>::Attention(
    int64_t query_dim,
    int64_t heads,
    int64_t dim_head,
    float dropout,
    bool bias
)
    : heads_(heads),
      dim_head_(dim_head > 0 ? dim_head : query_dim / heads),
      inner_dim_(heads_ * dim_head_)
{
    modules["to_q"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            bias
        );

    modules["to_k"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            bias
        );

    modules["to_v"] =
        std::make_shared<Linear>(
            query_dim,
            inner_dim_,
            bias
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
            query_dim
        );
}

template <class AttnOp>
Tensor Attention<AttnOp>::forward(
    ggml_context* ctx,
    Tensor hidden_states
)
{
    auto shape = hidden_states.shape();

    const int64_t batch = shape[0];
    const int64_t sequence = shape[1];


    auto q =
        std::static_pointer_cast<Linear>(
            modules["to_q"])
        ->forward(
            ctx,
            hidden_states
        );

    auto k =
        std::static_pointer_cast<Linear>(
            modules["to_k"])
        ->forward(
            ctx,
            hidden_states
        );

    auto v =
        std::static_pointer_cast<Linear>(
            modules["to_v"])
        ->forward(
            ctx,
            hidden_states
        );


    /*
        [B, S, H*D]
            ->
        [B, S, H, D]
            ->
        [B, H, S, D]

        This matches:
        torch.nn.functional.scaled_dot_product_attention()
    */

    q = q.reshape({
        batch,
        sequence,
        heads_,
        dim_head_
    });


    k = k.reshape({
        batch,
        sequence,
        heads_,
        dim_head_
    });


    v = v.reshape({
        batch,
        sequence,
        heads_,
        dim_head_
    });


    AttnOp attention_backend;

    auto out =
        attention_backend(
            ctx,
            q,
            k,
            v
        );


    /*
        [B, H, S, D]
            ->
        [B, S, H, D]
            ->
        [B, S, H*D]
    */

    out = out.reshape({
        batch,
        sequence,
        inner_dim_
    });


    out =
        std::static_pointer_cast<Linear>(
            modules["to_out.0"])
        ->forward(
            ctx,
            out
        );


    return out;
}
