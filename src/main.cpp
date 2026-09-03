#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <limits>
#include <sstream>
#include <cmath>
#include <random>
#include <numeric>
#include <functional>
#include <unordered_map>

// =========================================================================
// Mock GGML types (to allow compilation without actual GGML headers)
// =========================================================================
struct ggml_tensor { int dummy; };
struct ggml_context { int dummy; };
typedef int ggml_type;
void ggml_set_name(ggml_tensor* t, const char* name) {}
void ggml_time_init() {}
void ggml_log_set(void*, void*) {}
void ggml_backend_load_all() {}
enum ggml_log_level { GGML_LOG_LEVEL_INFO };

// =========================================================================
// Engine Interface (As provided)
// =========================================================================
class Engine {
public:
    virtual ~Engine() = default;

    virtual ggml_tensor* new_tensor(ggml_context* ctx, ggml_type type, int n_dims, const int64_t* ne) = 0;
    virtual ggml_tensor* new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0) = 0;
    virtual void set_input(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* fill(ggml_tensor* tensor, float value) = 0;

    virtual ggml_tensor* cont(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* dup(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* cast(ggml_tensor* tensor, ggml_type type) = 0;
    virtual ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) = 0;

    virtual ggml_tensor* neg(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* abs(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* sqrt(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* exp(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* log(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* sin(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* cos(ggml_tensor* tensor) = 0;

    virtual ggml_tensor* add(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* sub(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* mul(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* div(ggml_tensor* lhs, ggml_tensor* rhs) = 0;

    virtual ggml_tensor* scale(ggml_tensor* tensor, float value) = 0;
    virtual ggml_tensor* clamp(ggml_tensor* tensor, float min, float max) = 0;

    virtual ggml_tensor* mul_mat(ggml_tensor* lhs, ggml_tensor* rhs) = 0;

    virtual ggml_tensor* reshape_1d(ggml_tensor* tensor, int64_t ne0) = 0;
    virtual ggml_tensor* reshape_2d(ggml_tensor* tensor, int64_t ne0, int64_t ne1) = 0;
    virtual ggml_tensor* reshape_3d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2) = 0;
    virtual ggml_tensor* reshape_4d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) = 0;

    virtual ggml_tensor* permute(ggml_tensor* tensor, int axis0, int axis1, int axis2, int axis3) = 0;

    virtual ggml_tensor* view_1d(ggml_tensor* tensor, int64_t ne0, size_t offset) = 0;
    virtual ggml_tensor* view_2d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) = 0;
    virtual ggml_tensor* view_3d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) = 0;
    virtual ggml_tensor* view_4d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) = 0;

    virtual ggml_tensor* repeat(ggml_tensor* tensor, ggml_tensor* target) = 0;
    virtual ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) = 0;

    virtual ggml_tensor* sum_rows(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* silu(ggml_tensor* tensor) = 0;
};

// =========================================================================
// Planner Core Types
// =========================================================================
enum class DistType { R, S, P };

struct Dist {
    DistType type;
    int axis; 
    
    bool operator==(const Dist& o) const { return type == o.type && axis == o.axis; }
    bool operator<(const Dist& o) const {
        if (type != o.type) return type < o.type;
        return axis < o.axis;
    }
};

std::string dist_to_string(const Dist& d) {
    if (d.type == DistType::R) return "R";
    if (d.type == DistType::S) return "S(" + std::to_string(d.axis) + ")";
    if (d.type == DistType::P) return "P(" + std::to_string(d.axis) + ")";
    return "V";
}

struct Candidate {
    Dist output;
    std::vector<Dist> inputs;
    double comm_cost;
    double comp_cost;
};

struct TraceNode {
    int id;
    std::string op_name;
    std::vector<int> input_ids;
    std::vector<Candidate> candidates;
    bool is_param = false;
    bool is_input = false;
};

struct DPState {
    double cost;
    int best_cand_idx;
    std::vector<Dist> actual_in_dists;
};

struct TracebackInfo {
    Dist best_prod_dist;
    int best_cand_idx;
    std::vector<Dist> actual_in_dists;
};

struct PlanNode {
    int id;
    std::string op_name;
    Dist final_dist;
    Dist produced_dist;
    std::string comm_op;
};

struct Plan {
    std::vector<PlanNode> plan_nodes;
    std::map<int, Dist> param_distributions;

    std::string to_string() const {
        std::stringstream ss;
        ss << "=== Execution Plan ===\n";
        for (auto it = plan_nodes.rbegin(); it != plan_nodes.rend(); ++it) {
            const auto& pn = *it;
            ss << "Node " << pn.id << " (" << pn.op_name << "): ";
            if (pn.produced_dist == pn.final_dist) {
                ss << dist_to_string(pn.final_dist);
            } else {
                ss << dist_to_string(pn.produced_dist) << " --[" << pn.comm_op << "]--> " << dist_to_string(pn.final_dist);
            }
            ss << "\n";
        }
        ss << "\n=== Parameter Sharding ===\n";
        for (const auto& [id, dist] : param_distributions) {
            ss << "Param Node " << id << " -> " << dist_to_string(dist) << "\n";
        }
        ss << "======================\n";
        return ss.str();
    }
};

// =========================================================================
// PlannerEngine Implementation
// =========================================================================
class PlannerEngine : public Engine {
    int device_count;
    double w_comm, w_comp, w_mem;
    
    std::vector<TraceNode> nodes;
    std::unordered_map<ggml_tensor*, int> tensor_to_id;
    
    std::map<int, std::map<Dist, DPState>> memo;
    std::map<int, std::map<Dist, double>> min_cost_memo;
    std::map<int, std::map<Dist, TracebackInfo>> tb_memo;

    int trace_op(const std::string& name, const std::vector<int>& ins, const std::vector<Candidate>& cands) {
        int id = nodes.size();
        TraceNode node;
        node.id = id;
        node.op_name = name;
        node.input_ids = ins;
        node.candidates = cands;
        nodes.push_back(node);
        return id;
    }

    ggml_tensor* make_tensor(int id) {
        ggml_tensor* t = new ggml_tensor();
        tensor_to_id[t] = id;
        return t;
    }

    int get_id(ggml_tensor* t) {
        return tensor_to_id[t];
    }

    std::vector<Candidate> get_unary_candidates() {
        return {
            {{DistType::R, -1}, {{DistType::R, -1}}, 0.0, 1.0 * w_comp},
            {{DistType::S, 0}, {{DistType::S, 0}}, 0.0, 0.5 * w_comp},
            {{DistType::S, 1}, {{DistType::S, 1}}, 0.0, 0.5 * w_comp},
            {{DistType::P, 0}, {{DistType::P, 0}}, 0.0, 0.5 * w_comp}
        };
    }

    std::vector<Candidate> get_binary_candidates() {
        return {
            {{DistType::R, -1}, {{DistType::R, -1}, {DistType::R, -1}}, 0.0, 1.0 * w_comp},
            {{DistType::S, 0}, {{DistType::S, 0}, {DistType::S, 0}}, 0.0, 0.5 * w_comp},
            {{DistType::S, 1}, {{DistType::S, 1}, {DistType::S, 1}}, 0.0, 0.5 * w_comp},
            {{DistType::S, 0}, {{DistType::R, -1}, {DistType::S, 0}}, 0.0, 0.5 * w_comp},
            {{DistType::S, 1}, {{DistType::R, -1}, {DistType::S, 1}}, 0.0, 0.5 * w_comp}
        };
    }

    std::vector<Candidate> get_passthrough_candidates() {
        return {
            {{DistType::R, -1}, {{DistType::R, -1}}, 0.0, 0.0},
            {{DistType::S, 0}, {{DistType::S, 0}}, 0.0, 0.0},
            {{DistType::S, 1}, {{DistType::S, 1}}, 0.0, 0.0},
            {{DistType::P, 0}, {{DistType::P, 0}}, 0.0, 0.0}
        };
    }

    // Realistic communication costs relative to compute
    double get_comm_cost(const Dist& from, const Dist& to) {
        if (from == to) return 0.0;
        if (from.type == DistType::P && to.type == DistType::R) return 0.5 * w_comm; // AllReduce is relatively cheap
        if (from.type == DistType::S && to.type == DistType::R) return 1.0 * w_comm;  // AllGather
        if (from.type == DistType::R && to.type == DistType::S) return 0.0;           // Local Slice (Free)
        if (from.type == DistType::P && to.type == DistType::S) return 3.0 * w_comm; // ReduceScatter
        if (from.type == DistType::S && to.type == DistType::S && from.axis != to.axis) return 1.5 * w_comm; // AllToAll
        return 1e9;
    }

    std::string get_comm_op_name(const Dist& from, const Dist& to) {
        if (from.type == DistType::P && to.type == DistType::R) return "AllReduce";
        if (from.type == DistType::S && to.type == DistType::R) return "AllGather";
        if (from.type == DistType::R && to.type == DistType::S) return "Slice";
        if (from.type == DistType::P && to.type == DistType::S) return "ReduceScatter";
        if (from.type == DistType::S && to.type == DistType::S && from.axis != to.axis) return "AllToAll";
        return "UnknownComm";
    }

    DPState solve(int node_idx, Dist req_dist) {
        if (memo.count(node_idx) && memo[node_idx].count(req_dist)) {
            return memo[node_idx][req_dist];
        }
        
        const auto& node = nodes[node_idx];
        DPState best = {1e9, -1, {}};
        
        for (int c = 0; c < node.candidates.size(); ++c) {
            const auto& cand = node.candidates[c];
            if (cand.output == req_dist) {
                double cost = cand.comm_cost + cand.comp_cost;
                bool valid = true;
                std::vector<Dist> actual_ins;
                
                for (size_t i = 0; i < cand.inputs.size(); ++i) {
                    int in_id = node.input_ids[i];
                    Dist in_req = cand.inputs[i];
                    
                    double in_cost = get_min_cost(in_id, in_req);
                    if (in_cost >= 1e8) { valid = false; break; }
                    cost += in_cost;
                    actual_ins.push_back(in_req);
                }
                if (valid && cost < best.cost) {
                    best = {cost, c, actual_ins};
                }
            }
        }
        
        memo[node_idx][req_dist] = best;
        return best;
    }

    double get_min_cost(int node_idx, Dist req_dist) {
        if (min_cost_memo.count(node_idx) && min_cost_memo[node_idx].count(req_dist)) {
            return min_cost_memo[node_idx][req_dist];
        }
        
        double min_cost = 1e9;
        TracebackInfo best_tb;
        best_tb.best_prod_dist = req_dist;
        best_tb.best_cand_idx = -1;
        
        DPState direct_state = solve(node_idx, req_dist);
        if (direct_state.cost < min_cost) {
            min_cost = direct_state.cost;
            best_tb.best_prod_dist = req_dist;
            best_tb.best_cand_idx = direct_state.best_cand_idx;
            best_tb.actual_in_dists = direct_state.actual_in_dists;
        }
        
        std::set<Dist> produced_dists;
        for (const auto& cand : nodes[node_idx].candidates) produced_dists.insert(cand.output);
        
        for (const auto& prod_dist : produced_dists) {
            if (prod_dist == req_dist) continue;
            double comm = get_comm_cost(prod_dist, req_dist);
            if (comm >= 1e8) continue;
            
            DPState prod_state = solve(node_idx, prod_dist);
            if (prod_state.cost >= 1e8) continue;
            
            double cost = prod_state.cost + comm;
            if (cost < min_cost) {
                min_cost = cost;
                best_tb.best_prod_dist = prod_dist;
                best_tb.best_cand_idx = prod_state.best_cand_idx;
                best_tb.actual_in_dists = prod_state.actual_in_dists;
            }
        }
        
        min_cost_memo[node_idx][req_dist] = min_cost;
        tb_memo[node_idx][req_dist] = best_tb;
        return min_cost;
    }

    void traceback(int node_idx, Dist req_dist, std::vector<PlanNode>& plan) {
        if (node_idx < 0 || node_idx >= nodes.size()) return;
        if (min_cost_memo[node_idx][req_dist] >= 1e8) return;
        
        TracebackInfo tb = tb_memo[node_idx][req_dist];
        
        PlanNode pn;
        pn.id = node_idx;
        pn.op_name = nodes[node_idx].op_name;
        pn.final_dist = req_dist;
        pn.produced_dist = tb.best_prod_dist;
        pn.comm_op = (tb.best_prod_dist == req_dist) ? "None" : get_comm_op_name(tb.best_prod_dist, req_dist);
        
        plan.push_back(pn);
        
        if (tb.best_cand_idx != -1) {
            const auto& cand = nodes[node_idx].candidates[tb.best_cand_idx];
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                int in_id = nodes[node_idx].input_ids[i];
                Dist in_req = tb.actual_in_dists[i];
                traceback(in_id, in_req, plan);
            }
        }
    }

public:
    PlannerEngine(int dev_count, double w_c, double w_co, double w_m) 
        : device_count(dev_count), w_comm(w_c), w_comp(w_co), w_mem(w_m) {}

    // -------------------------------------------------------------------------
    // Tensor creation / initialization
    // -------------------------------------------------------------------------
    ggml_tensor* new_tensor(ggml_context* ctx, ggml_type type, int n_dims, const int64_t* ne) override {
        int id = trace_op("param", {}, {
            {{DistType::R, -1}, {}, 0.0, 2.0 * w_mem},
            {{DistType::S, 0}, {}, 0.0, 1.0 * w_mem},
            {{DistType::S, 1}, {}, 0.0, 1.0 * w_mem}
        });
        nodes[id].is_param = true;
        return make_tensor(id);
    }

    ggml_tensor* new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0) override {
        return new_tensor(ctx, type, 1, &ne0);
    }

    void set_input(ggml_tensor* tensor) override {
        int id = get_id(tensor);
        nodes[id].is_input = true;
        nodes[id].is_param = false;
        nodes[id].candidates = {
            {{DistType::R, -1}, {}, 0.0, 0.0}
        };
    }

    ggml_tensor* fill(ggml_tensor* tensor, float value) override {
        return make_tensor(trace_op("fill", {get_id(tensor)}, get_passthrough_candidates()));
    }

    // -------------------------------------------------------------------------
    // Copy / cast
    // -------------------------------------------------------------------------
    ggml_tensor* cont(ggml_tensor* tensor) override { return make_tensor(trace_op("cont", {get_id(tensor)}, get_passthrough_candidates())); }
    ggml_tensor* dup(ggml_tensor* tensor) override { return make_tensor(trace_op("dup", {get_id(tensor)}, get_passthrough_candidates())); }
    ggml_tensor* cast(ggml_tensor* tensor, ggml_type type) override { return make_tensor(trace_op("cast", {get_id(tensor)}, get_passthrough_candidates())); }
    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override { return make_tensor(trace_op("cpy", {get_id(src), get_id(dst)}, get_binary_candidates())); }

    // -------------------------------------------------------------------------
    // Unary arithmetic
    // -------------------------------------------------------------------------
    ggml_tensor* neg(ggml_tensor* t) override { return make_tensor(trace_op("neg", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* abs(ggml_tensor* t) override { return make_tensor(trace_op("abs", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* sqrt(ggml_tensor* t) override { return make_tensor(trace_op("sqrt", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* exp(ggml_tensor* t) override { return make_tensor(trace_op("exp", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* log(ggml_tensor* t) override { return make_tensor(trace_op("log", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* sin(ggml_tensor* t) override { return make_tensor(trace_op("sin", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* cos(ggml_tensor* t) override { return make_tensor(trace_op("cos", {get_id(t)}, get_unary_candidates())); }

    // -------------------------------------------------------------------------
    // Binary arithmetic
    // -------------------------------------------------------------------------
    ggml_tensor* add(ggml_tensor* l, ggml_tensor* r) override { return make_tensor(trace_op("add", {get_id(l), get_id(r)}, get_binary_candidates())); }
    ggml_tensor* sub(ggml_tensor* l, ggml_tensor* r) override { return make_tensor(trace_op("sub", {get_id(l), get_id(r)}, get_binary_candidates())); }
    ggml_tensor* mul(ggml_tensor* l, ggml_tensor* r) override { return make_tensor(trace_op("mul", {get_id(l), get_id(r)}, get_binary_candidates())); }
    ggml_tensor* div(ggml_tensor* l, ggml_tensor* r) override { return make_tensor(trace_op("div", {get_id(l), get_id(r)}, get_binary_candidates())); }

    // -------------------------------------------------------------------------
    // Scalar arithmetic
    // -------------------------------------------------------------------------
    ggml_tensor* scale(ggml_tensor* t, float v) override { return make_tensor(trace_op("scale", {get_id(t)}, get_unary_candidates())); }
    ggml_tensor* clamp(ggml_tensor* t, float min, float max) override { return make_tensor(trace_op("clamp", {get_id(t)}, get_unary_candidates())); }

    // -------------------------------------------------------------------------
    // Matrix operations
    // -------------------------------------------------------------------------
    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        std::vector<Candidate> cands = {
            {{DistType::R, -1}, {{DistType::R, -1}, {DistType::R, -1}}, 0.0, 1.0 * w_comp},
            {{DistType::S, 1}, {{DistType::R, -1}, {DistType::S, 1}}, 0.0, 0.5 * w_comp}, // Column Parallel
            {{DistType::S, 0}, {{DistType::S, 0}, {DistType::R, -1}}, 0.0, 0.5 * w_comp}, // Row Parallel
            {{DistType::P, -1}, {{DistType::S, 1}, {DistType::S, 0}}, 0.0, 0.5 * w_comp}  // Sequence/Partial
        };
        return make_tensor(trace_op("mul_mat", {get_id(l), get_id(r)}, cands));
    }

    // -------------------------------------------------------------------------
    // Reshape
    // -------------------------------------------------------------------------
    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t ne0) override { return make_tensor(trace_op("reshape", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t ne0, int64_t ne1) override { return make_tensor(trace_op("reshape", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2) override { return make_tensor(trace_op("reshape", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) override { return make_tensor(trace_op("reshape", {get_id(t)}, get_passthrough_candidates())); }

    // -------------------------------------------------------------------------
    // Permute / transpose
    // -------------------------------------------------------------------------
    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override { return make_tensor(trace_op("permute", {get_id(t)}, get_passthrough_candidates())); }

    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------
    ggml_tensor* view_1d(ggml_tensor* t, int64_t ne0, size_t off) override { return make_tensor(trace_op("view", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t ne0, int64_t ne1, size_t nb1, size_t off) override { return make_tensor(trace_op("view", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t off) override { return make_tensor(trace_op("view", {get_id(t)}, get_passthrough_candidates())); }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t off) override { return make_tensor(trace_op("view", {get_id(t)}, get_passthrough_candidates())); }

    // -------------------------------------------------------------------------
    // Repeat / broadcast
    // -------------------------------------------------------------------------
    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override { return make_tensor(trace_op("repeat", {get_id(t), get_id(target)}, get_binary_candidates())); }

    // -------------------------------------------------------------------------
    // Concatenation
    // -------------------------------------------------------------------------
    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) override { return make_tensor(trace_op("concat", {get_id(a), get_id(b)}, get_binary_candidates())); }

    // -------------------------------------------------------------------------
    // Reduction
    // -------------------------------------------------------------------------
    ggml_tensor* sum_rows(ggml_tensor* t) override {
        std::vector<Candidate> cands = {
            {{DistType::R, -1}, {{DistType::R, -1}}, 0.0, 1.0 * w_comp},
            {{DistType::R, -1}, {{DistType::S, 0}}, 1.0 * w_comm, 0.5 * w_comp}, 
            {{DistType::P, 0}, {{DistType::S, 1}}, 0.0, 0.5 * w_comp} 
        };
        return make_tensor(trace_op("sum_rows", {get_id(t)}, cands));
    }

    ggml_tensor* silu(ggml_tensor* t) override {
        std::vector<Candidate> cands = {
            {{DistType::R, -1}, {{DistType::R, -1}}, 0.0, 0.1 * w_comp},
            {{DistType::S, 0}, {{DistType::S, 0}}, 0.0, 0.05 * w_comp},
            {{DistType::S, 1}, {{DistType::S, 1}}, 0.0, 0.05 * w_comp}
        };
        return make_tensor(trace_op("silu", {get_id(t)}, cands));
    }

    // -------------------------------------------------------------------------
    // DP Finalization
    // -------------------------------------------------------------------------
    Plan finalize() {
        memo.clear();
        min_cost_memo.clear();
        tb_memo.clear();

        if (nodes.empty()) return Plan();
        int root_id = nodes.size() - 1;
        Dist req_dist = {DistType::R, -1}; // Assume final output must be gathered to Replicated
        
        get_min_cost(root_id, req_dist);
        
        Plan plan;
        traceback(root_id, req_dist, plan.plan_nodes);
        
        for (const auto& pn : plan.plan_nodes) {
            if (nodes[pn.id].is_param) {
                plan.param_distributions[pn.id] = pn.produced_dist;
            }
        }
        
        return plan;
    }
};

// =========================================================================
// Mock Module Framework for Demonstration
// =========================================================================
class Context {
public:
    ggml_context* ctx = nullptr;
};

class Tensor {
public:
    ggml_tensor* ggml_t = nullptr;
    Tensor(ggml_tensor* t) : ggml_t(t) {}
    Tensor() {}
};

class Parameter {
public:
    ggml_tensor* tensor_ = nullptr;
    void set(ggml_tensor* t) { tensor_ = t; }
    ggml_tensor* get() { return tensor_; }
    Tensor get_tensor() { return Tensor(tensor_); }
};

class Scope {
public:
    Context& ctx_;
    Engine& engine_;
    Scope(Context& c, Engine& e) : ctx_(c), engine_(e) {}
    Context& context() { return ctx_; }
    Engine& engine() { return engine_; }
};

class Visitor {
public:
    virtual void visit(Parameter& parameter, std::vector<std::string> path) = 0;
};

class Module {
public:
    std::map<std::string, std::shared_ptr<Module>> modules;
    virtual ~Module() = default;
    virtual Tensor forward(Scope scope, Tensor x) = 0;
    virtual void accept(Visitor& v) {}
};

class Linear : public Module {
public:
    int in_features, out_features;
    Parameter weight;
    
    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {}
    
    Tensor forward(Scope scope, Tensor x) override {
        Tensor w = weight.get_tensor();
        ggml_tensor* out = scope.engine().mul_mat(x.ggml_t, w.ggml_t);
        return Tensor(out);
    }
    
    void accept(Visitor& v) override {
        v.visit(weight, {"weight"});
    }
};

class SiLU : public Module {
public:
    Tensor forward(Scope scope, Tensor x) override {
        ggml_tensor* out = scope.engine().silu(x.ggml_t);
        return Tensor(out);
    }
};

class MLP : public Module {
public:
    MLP() {
        modules["fc1"] = std::make_shared<Linear>(4, 8);
        modules["fc2"] = std::make_shared<Linear>(8, 2);
        modules["silu"] = std::make_shared<SiLU>();
    }

    Tensor forward(Scope scope, Tensor x) {
        auto fc1 = std::static_pointer_cast<Linear>(modules["fc1"]);
        auto fc2 = std::static_pointer_cast<Linear>(modules["fc2"]);
        auto silu = std::static_pointer_cast<SiLU>(modules["silu"]);

        x = fc1->forward(scope, x);
        x = silu->forward(scope, x);
        x = fc2->forward(scope, x);

        return x;
    }
    
    void accept(Visitor& v) override {
        for (auto& [name, mod] : modules) {
            mod->accept(v);
        }
    }
};

class CreateRandomParametersVisitor : public Visitor {
public:
    Scope scope_;
    CreateRandomParametersVisitor(Scope scope) : scope_(scope) {}

    virtual void visit(Parameter& parameter, std::vector<std::string> path) override {
        ggml_tensor* t = scope_.engine().new_tensor(nullptr, 0, 1, nullptr);
        parameter.set(t);
    }
};

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    Context context;
    
    // Weights matching the original prompt
    PlannerEngine planner(2, /*w_comm=*/1.0, /*w_comp=*/1.0, /*w_mem=*/0.0); 

    Scope scope(context, planner);
    
    // Create input FIRST to match expected node IDs (Node 0)
    ggml_tensor* x_t = planner.new_tensor(nullptr, 0, 1, nullptr);
    planner.set_input(x_t);
    Tensor x_tensor(x_t);

    MLP model;
    CreateRandomParametersVisitor visitor(scope);
    model.accept(visitor); // Creates param1 (Node 1) and param2 (Node 2)

    auto y = model.forward(scope, x_tensor);
    
    auto plan = planner.finalize();
    std::cout << plan.to_string() << std::endl;
    
    return 0;
}