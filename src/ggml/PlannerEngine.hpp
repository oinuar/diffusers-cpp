#pragma once

#include "ggml/Engine.hpp"
#include "ggml/Distribution.hpp"
#include <ggml.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
#include <stdexcept>
#include <iostream>

class PlannerEngine : public Engine {
public:
    using TensorId = uint64_t;
    using OperationId = uint64_t;

    static constexpr TensorId InvalidTensor =
        std::numeric_limits<TensorId>::max();

    static constexpr OperationId InvalidOperation =
        std::numeric_limits<OperationId>::max();

    struct TensorInfo {
        TensorId id = InvalidTensor;

        ggml_type type = GGML_TYPE_COUNT;

        int n_dims = 0;
        int64_t ne[GGML_MAX_DIMS] = {};

        size_t bytes = 0;

        std::string name;
    };

    struct OperandRequirement {
        TensorId tensor = InvalidTensor;
        Distribution distribution;
    };

    struct Candidate {
        Distribution output;

        std::vector<OperandRequirement> operands;

        /*
         * Communication required by this candidate itself.
         *
         * This is separate from communication required to move an
         * operand into the required distribution.
         */
        double communication_cost = 0.0;

        /*
         * Cost representing how poorly this candidate utilizes
         * available compute resources.
         *
         * Lower is better.
         */
        double compute_cost = 0.0;

        /*
         * Optional memory pressure term.
         */
        double memory_cost = 0.0;

        /*
         * Candidate-specific descriptive information.
         */
        std::string description;
    };

    struct Operation {
        OperationId id = InvalidOperation;

        std::string name;

        std::vector<TensorId> inputs;
        TensorId output = InvalidTensor;

        std::vector<Candidate> candidates;
    };

    struct SelectedCandidate {
        OperationId operation = InvalidOperation;
        size_t candidate = 0;

        Distribution output;

        std::vector<OperandRequirement> operands;

        double local_cost = 0.0;
        double accumulated_cost = 0.0;
    };

    struct ExecutionPlan {
        std::unordered_map<TensorId, Distribution> distributions;

        std::unordered_map<OperationId, SelectedCandidate> operations;

        double total_cost = 0.0;

        const Distribution& distribution(TensorId id) const {
            auto it = distributions.find(id);

            if (it == distributions.end())
                throw std::runtime_error(
                    "Tensor has no resolved distribution");

            return it->second;
        }
    };

public:
    explicit PlannerEngine(
        int device_count,
        double communication_weight = 1.0,
        double compute_weight = 1.0,
        double memory_weight = 0.0);

    /*
     * Tensor registration.
     */
    TensorId tensor(
        ggml_tensor* tensor,
        const std::string& name = {});

    /*
     * Same primitive interface as Executor.
     *
     * These methods DO NOT call the corresponding ggml operation.
     *
     * They record the operation and create a symbolic ggml tensor
     * descriptor for the Tensor wrapper.
     */
    ggml_tensor* new_tensor(
        ggml_context* ctx,
        ggml_type type,
        int n_dims,
        const int64_t* ne) override;

    ggml_tensor* new_tensor_1d(
        ggml_context* ctx,
        ggml_type type,
        int64_t ne0) override;

    void set_input(ggml_tensor* tensor) override;

    ggml_tensor* fill(
        ggml_tensor* a,
        float value) override;

    ggml_tensor* cont(
        ggml_tensor* a) override;

    ggml_tensor* dup(
        ggml_tensor* a) override;

    ggml_tensor* cast(
        ggml_tensor* a,
        ggml_type type) override;

    ggml_tensor* cpy(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* neg(
        ggml_tensor* a) override;

    ggml_tensor* abs(
        ggml_tensor* a) override;

    ggml_tensor* sqrt(
        ggml_tensor* a) override;

    ggml_tensor* exp(
        ggml_tensor* a) override;

    ggml_tensor* log(
        ggml_tensor* a) override;

    ggml_tensor* sin(
        ggml_tensor* a) override;

    ggml_tensor* cos(
        ggml_tensor* a) override;

    ggml_tensor* add(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* sub(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* mul(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* div(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* scale(
        ggml_tensor* a,
        float value) override;

    ggml_tensor* clamp(
        ggml_tensor* a,
        float min,
        float max) override;

    ggml_tensor* mul_mat(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* reshape_1d(
        ggml_tensor* a,
        int64_t ne0) override;

    ggml_tensor* reshape_2d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1) override;

    ggml_tensor* reshape_3d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) override;

    ggml_tensor* reshape_4d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) override;

    ggml_tensor* permute(
        ggml_tensor* a,
        int axis0,
        int axis1,
        int axis2,
        int axis3) override;

    ggml_tensor* view_1d(
        ggml_tensor* a,
        int64_t ne0,
        size_t offset) override;

    ggml_tensor* view_2d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        size_t nb1,
        size_t offset) override;

    ggml_tensor* view_3d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        size_t nb1,
        size_t nb2,
        size_t offset) override;

    ggml_tensor* view_4d(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb1,
        size_t nb2,
        size_t nb3,
        size_t offset) override;

    ggml_tensor* repeat(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* concat(
        ggml_tensor* a,
        ggml_tensor* b,
        int dim) override;

    ggml_tensor* sum_rows(
        ggml_tensor* a) override;

    ggml_tensor* silu(
        ggml_tensor* tensor) override;

    /*
     * Planning.
     */
    void finalize();

    void print_plan(std::ostream& os = std::cout) const;

    const ExecutionPlan& plan() const {
        if (!plan_)
            throw std::runtime_error("Planner has not been finalized");

        return *plan_;
    }

    void clear();

private:
    struct Node {
        TensorInfo info;

        bool is_input = false;

        Distribution initial_distribution =
            Distribution::replicated();

        OperationId producer = InvalidOperation;
    };

    struct DPState {
        double cost = std::numeric_limits<double>::infinity();

        OperationId producer = InvalidOperation;

        size_t candidate =
            std::numeric_limits<size_t>::max();

        Distribution distribution;
    };

    /*
     * Generic operation recording.
     */
    ggml_tensor* unary(
        const char* name,
        ggml_tensor* input,
        ggml_type output_type,
        const int64_t* output_ne,
        int n_dims);

    ggml_tensor* binary(
        const char* name,
        ggml_tensor* a,
        ggml_tensor* b);

    ggml_tensor* create_output(
        ggml_context* ctx,
        ggml_type type,
        int n_dims,
        const int64_t* ne);

    TensorId id_of(ggml_tensor* tensor) const;

    Node& node(TensorId id);

    const Node& node(TensorId id) const;

    Operation& operation(OperationId id);

    const Operation& operation(OperationId id) const;

    OperationId record_operation(
        const std::string& name,
        const std::vector<TensorId>& inputs,
        TensorId output,
        std::vector<Candidate> candidates);

    /*
     * Candidate generation.
     */
    std::vector<Candidate> unary_candidates(
        TensorId input,
        const char* operation_name) const;

    std::vector<Candidate> elementwise_candidates(
        TensorId lhs,
        TensorId rhs,
        const char* operation_name) const;

    /*
     * Cost model.
     */
    double communication_cost(
        const Distribution& from,
        const Distribution& to,
        const TensorInfo& tensor) const;

    double distribution_compute_cost(
        const Distribution& distribution,
        const TensorInfo& tensor) const;

    double candidate_local_cost(
        const Operation& operation,
        const Candidate& candidate) const;

    /*
     * Dynamic programming.
     */
    void solve();

    void solve_operation(
        const Operation& operation,
        std::unordered_map<
            TensorId,
            std::unordered_map<Distribution, DPState, DistributionHash>>&
            states);

    DPState best_state_for_operand(
        TensorId tensor,
        const Distribution& required,
        const std::unordered_map<
            TensorId,
            std::unordered_map<Distribution, DPState, DistributionHash>>&
            states) const;

private:
    int device_count_;

    double communication_weight_;
    double compute_weight_;
    double memory_weight_;

    TensorId next_tensor_id_ = 0;
    OperationId next_operation_id_ = 0;

    /*
     * Symbolic ggml tensor -> planner node.
     */
    std::unordered_map<ggml_tensor*, TensorId> tensor_ids_;

    std::vector<Node> nodes_;
    std::vector<Operation> operations_;

    std::unique_ptr<ExecutionPlan> plan_;
};
