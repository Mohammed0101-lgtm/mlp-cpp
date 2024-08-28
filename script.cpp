#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <fstream>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <omp.h>
#include <random>


const int batch_size = 16;
const int context_length = 8;
const int max_iterations = 100000;
const int eval_interval = 300;
const int eval_iters = 200;

std::unordered_map<int, char> global_index_table;

class Value {
private:
    float data = 0.0f;
    float grad = 0.0f;
    std::vector<std::shared_ptr<Value>> _prev;
    char _op = ' ';

public:

    Value() : data(0.0f), grad(0.0f), _op(' ') {}

    Value(Value& val) : data(val.data), grad(val.data), _prev(val._prev), _op(val._op) {}

    Value(float d) : data(d), grad(0.0f), _op(' ') {}

    Value(float d, const std::vector<std::shared_ptr<Value>>& children, char op) 
        : data(d), grad(0.0f), _prev(children), _op(op) {}

    float getData() const { return data; }

    float getGrad() const { return grad; }

    void set_grad(float g) { grad = g; }

    const std::vector<std::shared_ptr<Value>>& getPrev() const { return _prev; }

    friend std::ostream& operator<<(std::ostream& os, const Value& val) {
        os << val.data;
        return os;
    }

    Value operator+(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data + other.data), children, '+');
        return res;
    }

    Value operator-(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data - other.getData()), children, '-');
        return res;
    }

    Value operator*(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data * other.getData()), children, '*');
        return res;
    }

    Value operator*(int scalar) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        Value res((data * scalar), children, '*');
        return res;
    }

    float operator/(const Value& other) const { return data / other.getData(); }

    Value& operator+=(const Value& other) {
        data += other.getData(); 
        return *this; 
    }

    Value _pow(const Value& other) {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        data = pow(data, other.data);
        Value res(data, children, '^');
        return res;
    }

    void _backward() {
        if (_op == ' ' || _prev.empty() || !_prev[0]) {
            std::cerr << "Debug : There is a problem in the backward function" << std::endl;
            return;
        } else if (_op == '+') {
            _prev[0]->grad += grad;
            _prev[1]->grad += grad;
        } else if (_op == '*') {
            _prev[0]->grad += _prev[1]->data * grad;
            _prev[1]->grad += _prev[0]->data * grad;
        } else if (_op == '^') { 
            float exponent = _prev[1]->data;
            _prev[0]->grad += exponent * std::pow(_prev[0]->data, exponent - 1) * grad;
        } else if (_op == 'r') { 
            _prev[0]->grad += (_prev[0]->data > 0) * grad;
        }
    }

    void backward() {
        std::list<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>> visited;
        build_topo(topo, visited, std::make_shared<Value>(*this));
        this->grad = 1.0f;

        topo.reverse();
        for (std::shared_ptr<Value> v : topo) {
            v->_backward();
        }
    }

private:

    void build_topo(std::list<std::shared_ptr<Value>>& topo, 
                    std::set<std::shared_ptr<Value>>& visited, 
                    std::shared_ptr<Value> node) {
                        
        if (visited.find(node) == visited.end()) {
            visited.insert(node);
            for (auto child : node->getPrev()) {
                build_topo(topo, visited, child);
            }

            topo.push_back(node);
        }
    }
};


class Tensor {
private:
    std::vector<int> shape;
    std::vector<int> strides;  
    std::vector<Value> data;

    void computeStrides() {
        int stride = 1;
        strides.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    int getIndex(const std::vector<int>& indices) const {
        int index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return index;
    }

public:
    Tensor() {}

    Tensor(std::initializer_list<int> dims) : shape(dims) {
        int totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }
        data.resize(totalSize);
        computeStrides();
    }

    Tensor(const std::vector<int>& dims) : shape(dims) {
        int totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }
        data.resize(totalSize);
        computeStrides();
    }

    Tensor(std::vector<Value> vals) : data(vals) {}

    class Proxy {
    public: 
        Tensor& tensor;
        std::vector<int> indices;

        Proxy(Tensor& t, std::vector<int> idx) 
        : tensor(t), indices(std::move(idx)) {}

        Proxy operator[](int i) {
            indices.push_back(i);
            return Proxy(tensor, indices);
        }

        Value& operator=(const Value& value) {
            return tensor.data[tensor.getIndex(indices)] = value;
        }

        operator Value&() {
            return tensor.data[tensor.getIndex(indices)];
        }
    };

    class ConstProxy {
    public:
        const Tensor& tensor;
        std::vector<int> indices;

        ConstProxy(const Tensor& t, std::vector<int> idx) 
            : tensor(t), indices(std::move(idx)) {}

        ConstProxy operator[](int i) const {
            std::vector<int> new_indices = indices;
            new_indices.push_back(i);
            return ConstProxy(tensor, new_indices);
        }

        operator const Value&() const {
            return tensor.data[tensor.getIndex(indices)];
        }
    };
    
    Proxy operator[](int i) {
        return Proxy(*this, static_cast<std::vector<int>>(i));
    }

    ConstProxy operator[](int i) const {
        return ConstProxy(*this, static_cast<std::vector<int>>(i));
    }


    Tensor operator+(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    Tensor matmul(const Tensor& other) const {
        assert(shape.size() == 2 && other.shape.size() == 2);
        assert(shape[1] == other.shape[0]);

        Tensor result(static_cast<std::vector<int>>(shape[0], other.shape[1]));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < other.shape[1]; j++) {
                float sum = 0.0f;
                for (int k = 0; k < shape[1]; k++) {
                    sum +=  data[getIndex(std::vector<int>( {i, k} ))].getData() * 
                            other.data[other.getIndex(std::vector<int>( {k, j} )) ].getData();
                }
                result.data[result.getIndex(std::vector<int>( {i, j} ))] = Value(sum);
            }
        }

        return result;
    }

    Tensor relu() const {
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = std::max(0.0f, data[i].getData());
        }
        return result;
    }

    Tensor tanh() const {
        Tensor result(shape);
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = std::tanh(data[i].getData());
        }
        return result;
    }

    Tensor multinomial(int num_samples = 1) const {
        assert(shape.size() == 1); 

        std::vector<float> cumulative_probs(data.size());
        cumulative_probs[0] = data[0].getData();
        for (size_t i = 1; i < data.size(); ++i) {
            cumulative_probs[i] = cumulative_probs[i - 1] + data[i].getData();
        }

        Tensor result(static_cast<std::vector<int>>(num_samples)); 

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, cumulative_probs.back());

        for (int sample = 0; sample < num_samples; ++sample) {
            float rand_prob = dis(gen); 
            auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), rand_prob);
            int sampled_index = std::distance(cumulative_probs.begin(), it);
            result[sample] = Value(static_cast<float>(sampled_index));
        }

        return result;
    }

    Tensor softmax() const {
        assert(shape.size() == 1);
        Tensor result(shape);
        float maxVal = (*std::max_element(data.begin(), data.end(), [](const Value& a, const Value& b) {
            return a.getData() < b.getData();
        })).getData();

        float sumExp = 0.0f;
        #pragma omp parallel for reduction(+:sumExp)
        for (const auto& val : data) {
            sumExp += std::exp(val.getData() - maxVal);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = std::exp(data[i].getData() - maxVal) / sumExp;
        }

        return result;
    }

    size_t size() const { return data.size(); }
    const std::vector<int>& getShape() const { return shape; }
    void set_data(const std::vector<Value>& vals) { data = vals; }
    const std::vector<Value>& get_data() const { return data; }
};


class LinearLayer {
private:
    Tensor weights;
    float bias = 0.0f;

public:

    LinearLayer(Tensor& wei) : weights(wei) {}
    LinearLayer(Tensor& wei, float b) : weights(wei), bias(b) {}

    int get_size() { return weights.size(); }
    Tensor get_weights() { return weights; }
    void set_weights(Tensor& wei) { weights = wei; }

    LinearLayer(size_t size) : weights(std::vector<Value>(size)) {
        weights_init_(size);
    }

    void weights_init_(size_t __size) {
        std::vector<float> wei(__size); 
        srand(static_cast<unsigned>(time(nullptr)));

        for (size_t i = 0; i < __size; i++) {
            wei[i] = (static_cast<float>(rand()) / (float)RAND_MAX * 2 - 1);
        }

        std::vector<Value> vals;
        for (float w : wei) {
            vals.push_back(Value(w));
        }

        weights.set_data(vals);
    }

    Tensor feed_forward(Tensor& data) {
        assert(data.size() == weights.size());
        Tensor output = data.matmul(weights); 
        output = output.relu(); 

        if (bias != 0.0f) {
            std::vector<Value> inter_data = output.get_data();
            for (Value& val : inter_data) {
                val += bias; 
            } 

            output.set_data(inter_data);
        }

        return output;
    }

};


class NeuralNetwork {
private:
    std::vector<Value> parameters;
    LinearLayer input_layer;
    LinearLayer hidden_layer;
    LinearLayer output_layer;
    bool is_training = true; 

public:
    NeuralNetwork(size_t input_size) 
    : input_layer(input_size), hidden_layer(input_size), output_layer(1) {
        for (size_t i = 0; i < input_size * 2 + 1; i++) {
            parameters[i] = Value(0.0f);
        }
    }

    Tensor feed_forward(Tensor& input_data) {
        size_t input_size = input_data.size();

        Tensor logits = input_layer.feed_forward(input_data);
        logits = hidden_layer.feed_forward(logits);
        logits.tanh();
        output_layer.feed_forward(logits);
        logits.softmax();
        return logits;
    }

    void backpropagate(Tensor& output, Tensor& targets, float learning_rate = 0.01) {
        float loss = mse_loss(output, targets);

        Tensor loss_grad = output - targets; 
        loss_grad = loss_grad * (1.0f / output.size()); 

        Tensor d_logits = output.softmax(); 

        Tensor output_layer_weights = output_layer.get_weights();
        std::vector<Value> output_layer_weights_vals = output_layer_weights.get_data();
        for (size_t i = 0; i < output_layer_weights_vals.size(); i++) {
            float grad = d_logits.get_data()[i].getGrad(); 
            output_layer_weights_vals[i] = output_layer_weights_vals[i] - (learning_rate * grad);
            output_layer_weights_vals[i].set_grad(0.0f); 
        }
        Tensor updated_output_weights(output_layer_weights_vals);
        output_layer.set_weights(updated_output_weights);
        
        Tensor hidden_grad = hidden_layer.feed_forward(output); 
        Tensor hidden_layer_weights = hidden_layer.get_weights();
        std::vector<Value> hidden_layer_weights_vals = hidden_layer_weights.get_data();
        for (size_t i = 0; i < hidden_layer_weights_vals.size(); i++) {
            float grad = hidden_grad.get_data()[i].getGrad(); 
            hidden_layer_weights_vals[i] = hidden_layer_weights_vals[i] - (learning_rate * grad);
            hidden_layer_weights_vals[i].set_grad(0.0f);
        }
        Tensor updated_hidden_weights(hidden_layer_weights_vals);
        hidden_layer.set_weights(updated_hidden_weights);
        
        Tensor input_grad = input_layer.feed_forward(hidden_grad); 
        Tensor input_layer_weights = input_layer.get_weights();
        std::vector<Value> input_layer_weights_vals = input_layer_weights.get_data();
        for (size_t i = 0; i < input_layer_weights_vals.size(); i++) {
            float grad = input_grad.get_data()[i].getGrad(); 
            input_layer_weights_vals[i] = input_layer_weights_vals[i] - (learning_rate * grad);
            input_layer_weights_vals[i].set_grad(0.0f); 
        }
        Tensor updated_input_weights(input_layer_weights_vals);
        input_layer.set_weights(updated_input_weights);
    }


    Tensor generate(Tensor idx, size_t max_tokens)  {
        for (size_t i = 0; i < max_tokens; i++) {
            Tensor logits = feed_forward(idx);
            Tensor probs = logits.softmax();
            Tensor next = probs.multinomial();
            idx = idx + next;
        }

        return idx;
    }

    void set_eval() { is_training = false; }
    void set_train() { is_training = true; }

    float mse_loss(const Tensor& predicted, const Tensor& targets) {
        assert(predicted.getShape() == targets.getShape());
        
        float sumSquaredDifferences = 0.0f;
        size_t numElements = predicted.size();
        
        for (size_t i = 0; i < numElements; i++) {
            Value diff = predicted.get_data()[i] - targets.get_data()[i];
            sumSquaredDifferences += diff.getData() * diff.getData();
        }
    
        return sumSquaredDifferences / numElements;
    }
};

std::vector<std::stack<std::vector<int>>> get_batch(std::vector<int>& data) {
    int rand_offset = rand() % (data.size() - (context_length) * batch_size);
    
    std::stack<std::vector<int>> inputs;
    std::stack<std::vector<int>> targets;

    size_t data_size = data.size();
    size_t chunck_begin = rand_offset;
    size_t chunck_end = rand_offset + context_length;
    
    for (int i = 0; i < batch_size; i++) {
        if (chunck_end >= data_size) { break; }
        std::vector<int> input_chunck(data.begin() + chunck_begin, data.begin() + chunck_end);
        std::vector<int> target_chunck(data.begin() + chunck_begin + 1, data.begin() + chunck_end + 1);
        inputs.push(input_chunck);
        targets.push(target_chunck);
        chunck_begin += context_length;
        chunck_end += context_length;
    }

    std::vector<std::stack<std::vector<int>>> result;
    result.push_back(inputs);
    result.push_back(targets);
    return result;
}

std::string read_file(const std::string& _filename) {
    std::ifstream file(_filename, std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading");
    }

    std::string input;
    std::stringstream buffer;

    try {
        buffer << file.rdbuf();  
        input = buffer.str();    
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    
    file.close();
    return input;
}

std::vector<int> encode(const std::string& input) {
    std::set<char> chars;
    std::unordered_map<char, int> table;
    int highest = 0;
    for (char c : input) { chars.emplace(c); }
    for (char c : input) { 
        if (table.find(c) == table.end()) {
            table.emplace(c, highest++);
        }
    }

    std::vector<int> result;
    for (char c : input) {
        result.push_back(table[c]);
    }

    for (std::pair<char, int> c : table) {
        global_index_table[c.second] = c.first;
    }

    return result;
}

std::string decode(const std::vector<int>& tokens) {
    std::string result;
    for (int n : tokens) {
        result.push_back(global_index_table[n]);
    }

    return result;
}

std::unordered_map<std::string, float> estimate_loss(
    NeuralNetwork& model, 
    std::vector<int> train_data, 
    std::vector<int> val_data
) {
    std::unordered_map<std::string, float> out;

    std::vector<std::string> splits;
    splits.push_back("train");
    splits.push_back("val");
    for (const std::string& split : splits) {
        std::vector<float> losses(eval_iters, 0.0f);
        
        for (int k = 0; k < eval_iters; ++k) {
            std::vector<std::stack<std::vector<int>>> batches = get_batch(split == "train" ? train_data:val_data);
            std::stack<std::vector<int>> inputs = batches[0];
            std::stack<std::vector<int>> targets = batches[1];
            Tensor inputs_as_tensor;
            Tensor targets_as_tensor;
            for (int i = 0, n = inputs.size(); i < n; i++) {
                std::vector<Value> vals;
                for (int val : inputs.top()) {
                    vals.push_back(Value(static_cast<float>(val)));
                }
                inputs_as_tensor = inputs_as_tensor + Tensor(vals);
                inputs.pop();
            }

            for (int i = 0, n = targets.size(); i < n; i++) {
                std::vector<Value> vals;
                for (int val : targets.top()) {
                    vals.push_back(Value(static_cast<float>(val)));
                }
                targets_as_tensor = targets_as_tensor + Tensor(vals);
                targets.top();
            }

            Tensor logits = model.feed_forward(inputs_as_tensor);
            float loss = model.mse_loss(logits, targets_as_tensor);
            losses[k] = loss;
        }
        
        float mean_loss = std::accumulate(losses.begin(), losses.end(), 0.0f) / eval_iters;
        out[split] = mean_loss;
    }

    return out;
}

void train_model(NeuralNetwork& model, std::vector<int>& data) {
    size_t data_size = data.size();
    size_t train_size = static_cast<size_t>(data_size * 0.9);
    std::vector<int> train_data(data.begin(), data.begin() + train_size);
    std::vector<int> val_data(data.begin() + train_size, data.end());

    for (int iter = 0; iter < max_iterations; iter++) {
        if (iter % eval_interval == 0) {
            auto losses = estimate_loss(model, train_data, val_data);
            std::cout << "step " << iter << " : train loss " << losses["train"] << ", val loss " << losses["val"] << std::endl;
        }

        std::vector<std::stack<std::vector<int>>> batches = get_batch(train_data);
        std::stack<std::vector<int>> input_batches = batches[0];
        std::stack<std::vector<int>> target_batches = batches[1];
        Tensor input_as_tensor;
        Tensor target_as_tensor;

        for (int i = 0, n = input_batches.size(); i < n; i++) {
            std::vector<Value> vals;
            for (int val : input_batches.top()) {
                vals.push_back(Value(static_cast<float>(val)));
            }
            input_as_tensor = input_as_tensor + Tensor(vals);
            input_batches.pop();
        }

        for (int i = 0, n = target_batches.size(); i < n; i++) {
            std::vector<Value> vals;
            for (int val : target_batches.top()) {
                vals.push_back(Value(static_cast<float>(val)));
            }
            target_as_tensor = target_as_tensor + Tensor(vals);
            target_batches.pop();
        }

        Tensor logits = model.feed_forward(input_as_tensor);
        model.backpropagate(logits, target_as_tensor);
    }
}

int main(void) {
    std::string input = read_file("input_.txt");
    NeuralNetwork model(input.size());
    std::vector<int> encoded = encode(input);
    train_model(model, encoded);
    return 0;
}
