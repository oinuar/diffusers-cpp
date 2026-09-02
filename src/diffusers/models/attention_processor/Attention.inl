#include "diffusers/models/attention_processor/Attention.hpp"
#include "nn/Linear.hpp"
#include "nn/ModuleList.hpp"

template <class AttnOp>
Attention<AttnOp>::Attention(
    int64_t query_dim,
    int64_t heads,
    int64_t dim_head,
    float dropout,
    bool bias,
    bool residual_connection,
    std::optional<int64_t> norm_num_groups,
    float eps,
    float rescale_output_factor,
    bool upcast_softmax
)
    :
    heads_(heads),
    dim_head_(dim_head > 0 ? dim_head : query_dim / heads),
    inner_dim_(heads * dim_head_),
    residual_connection_(residual_connection),
    norm_num_groups_(norm_num_groups),
    rescale_output_factor_(rescale_output_factor),
    upcast_softmax_(upcast_softmax)
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


    modules["to_out"] = std::shared_ptr<Module>(new ModuleList({
        std::shared_ptr<Module>(new Linear(
            inner_dim_,
            query_dim,
            true
        ))
    }));

    if (norm_num_groups_)
        modules["group_norm"] =
            std::make_shared<GroupNorm>(
                *norm_num_groups_,
                query_dim,
                eps
            );
}

template <class AttnOp>
Tensor Attention<AttnOp>::forward(Scope scope, Tensor hidden_states)
{
    auto shape = hidden_states.shape();

    bool is_4d = shape.rank() == 4;

    int64_t batch;
    int64_t sequence;
    int64_t channels;
    int64_t height = 0;
    int64_t width = 0;

    Tensor residual = hidden_states;

    if (norm_num_groups_) {

        hidden_states =
            std::static_pointer_cast<GroupNorm>(
                modules["group_norm"])
            ->forward(
                scope,
                hidden_states
            );
    }

    if (is_4d) {

        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];

        sequence = height * width;


        // [B,C,H,W]
        // ->
        // [B,HW,C]

        hidden_states =
            hidden_states
                .permute({
                    0,
                    2,
                    3,
                    1
                })
                .contiguous()
                .reshape({
                    batch,
                    sequence,
                    channels
                });
    }
    else {

        batch = shape[0];
        sequence = shape[1];
        channels = shape[2];
    }

    auto q =
        std::static_pointer_cast<Linear>(
            modules["to_q"])
        ->forward(
            scope,
            hidden_states
        );


    auto k =
        std::static_pointer_cast<Linear>(
            modules["to_k"])
        ->forward(
            scope,
            hidden_states
        );


    auto v =
        std::static_pointer_cast<Linear>(
            modules["to_v"])
        ->forward(
            scope,
            hidden_states
        );


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
            scope,
            q,
            k,
            v
        );


    out =
        out.reshape({
            batch,
            sequence,
            inner_dim_
        });


    auto to_out = std::static_pointer_cast<ModuleList>(modules["to_out"]);

    out = std::static_pointer_cast<Linear>((*to_out)[0])->forward(scope, out);

    if (is_4d) {

        // [B,HW,C]
        // ->
        // [B,C,H,W]

        out =
            out
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
                })
                .contiguous();
    }

    if (residual_connection_)
        out = out + residual;

    return out / rescale_output_factor_;
}
