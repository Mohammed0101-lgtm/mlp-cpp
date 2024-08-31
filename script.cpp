#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <fstream>
#include <numeric>
#include <sstream>
#include <random> 
#include <initializer_list>

// hyperparameters
const int batch_size = 16;
const int context_length = 8;
const int max_iterations = 100000;
const int eval_interval = 300;
const int eval_iters = 200;
const int vocab_size = 0;

// thanks to 'micrograd' by 'Andrej Karpathy' which is open on github 
namespace grad {

class Value {
private:
    // actual val
    float data = 0.0f; 
    // gradient of this val in relation to the previous val
    float grad = 0.0f;
    // previous values that produced this value
    std::vector<std::shared_ptr<Value>> _prev;
    // operation by which this value was produced
    char _op = ' ';
    // index for look up
    int index;

public:
    // constructors
    Value() : data(0.0f), grad(0.0f), _op(' ') {}
    Value(const Value& val) : data(val.data), grad(val.grad), _prev(val._prev), _op(val._op) {}
    Value(Value& val) : data(val.data), grad(val.data), _prev(val._prev), _op(val._op) {}
    Value(float d) : data(d), grad(0.0f), _op(' ') {}
    Value(float d, const std::vector<std::shared_ptr<Value>>& children, char op) 
        : data(d), grad(0.0f), _prev(children), _op(op) {}

    // getters
    int getIndex() const { return index; }
    float getData() const { return data; }
    float getGrad() const { return grad; }
    const std::vector<std::shared_ptr<Value>>& getPrev() const { return _prev; }

    // one setting the value just in case we needed direct access to data
    void set_grad(float g) { grad = g; }

    /*---operators---*/
    // for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& val) {
        os << val.data;
        return os;
    }

    // Value + Value
    Value operator+(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data + other.data), children, '+');

        return res;
    }

    // Value(this) - Value(other)
    Value operator-(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data - other.getData()), children, '-');
        return res;
    }

    // Value * Value
    Value operator*(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data * other.getData()), children, '*');
        return res;
    }

    // Value(this) * scalar
    Value operator*(int scalar) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        Value res((data * scalar), children, '*');
        return res;
    }

    // this / Value(other)
    float operator/(const Value& other) const { return data / other.getData(); }

    // meet them outside hehe 
    Value _pow(const Value& other);
    void _backward();
    void backward();
    
private:
    // build the topology of the map you should 
    // travel to get to the beginning aka weights
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

// Value(this) ** Value(other)
Value Value::_pow(const Value& other) {
    // using smart pointer is just simply smarter
    std::vector<std::shared_ptr<Value>> children;
    children.push_back(std::make_shared<Value>(*this));
    data = pow(data, other.data);
    Value res(data, children, '^');
    return res;
}

// how to go backward depends on the operation
void Value::_backward() {
    if (_op == ' ' || _prev.empty() || !_prev[0]) {
        std::cerr << "Debug : There is a problem in the backward function" << std::endl;
        return;
    } else if (_op == '+') {
        // '+' distributes the gradient equally
        // (a + b)' = a' + b'  
        _prev[0]->grad += grad;
        _prev[1]->grad += grad;
    } else if (_op == '*') {
        // (a * b)' = a * b' + a' * b
        _prev[0]->grad += _prev[1]->data * grad;
        _prev[1]->grad += _prev[0]->data * grad;
    } else if (_op == '^') { 
        // (a ^ b)' = b * (a ^ (b - 1))
        float exponent = _prev[1]->data;
        _prev[0]->grad += exponent * std::pow(_prev[0]->data, exponent - 1) * grad;
    } else if (_op == 'r') { 
        // relu'(x) = (x > 0) ? 1 : 0
        _prev[0]->grad += (_prev[0]->data > 0) * grad;
    }
}

// traverse the built map 
void Value::backward() {
    std::list<std::shared_ptr<Value>> topo;
    std::set<std::shared_ptr<Value>> visited;
    build_topo(topo, visited, std::make_shared<Value>(*this));
    this->grad = 1.0f;

    topo.reverse();
    for (std::shared_ptr<Value> v : topo) { v->_backward(); }
}

}; // grad


namespace tensor {

class Tensor {
private:
    // shape of a tensor is the dimensions that describe 
    // it's "shape" in the dimension it exists in
    std::vector<int> shape;
    // go see how tensors are represented in memory
    std::vector<int> strides;  
    // actual linear data
    std::vector<grad::Value> data; 

    void computeStrides() {
        int stride = 1;
        strides.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // convert the vector of indecies to linear 
    // indexing to get the element requested
    int getIndex(const std::vector<int>& indices) const {
        int index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return index;
    }

public:
    // constructors
    Tensor() {}
    Tensor(std::vector<grad::Value> vals) : data(vals) {}
    Tensor(const std::vector<int>& dims) : shape(dims) {
        int totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }
        data.resize(totalSize);
        computeStrides();
    }
    
    // the proxy class is a stretch from me but it works fine
    class Proxy {
    public: 
        Tensor& tensor;
        std::vector<int> indices;

        Proxy(Tensor& t, std::vector<int> idx) : tensor(t), indices(std::move(idx)) {}

        Proxy operator[](int i) {
            indices.push_back(i);
            return Proxy(tensor, indices);
        }

        grad::Value& operator=(const grad::Value& value) { return tensor.data[tensor.getIndex(indices)] = value; }

        operator grad::Value&() { return tensor.data[tensor.getIndex(indices)]; }
    };

    class ConstProxy {
    public:
        const Tensor& tensor;
        std::vector<int> indices;

        ConstProxy(const Tensor& t, std::vector<int> idx) : tensor(t), indices(std::move(idx)) {}

        ConstProxy operator[](int i) const {
            std::vector<int> new_indices = indices;
            new_indices.push_back(i);
            return ConstProxy(tensor, new_indices);
        }

        operator const grad::Value&() const { return tensor.data[tensor.getIndex(indices)]; }
    };
    
    Proxy operator[](int i) { return Proxy(*this, static_cast<std::vector<int>>(i)); }
    
    ConstProxy operator[](int i) const { return ConstProxy(*this, static_cast<std::vector<int>>(i)); }

    Tensor operator+(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        assert(shape == other.shape);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    void set_row(int row_index, const std::vector<grad::Value>& vals) {
        // Ensure the tensor is 2-dimensional
        assert(this->shape.size() == 2); 

        int rows = shape[0];  // Number of rows
        int cols = shape[1];  // Number of columns

        // Check if the provided row index is within bounds
        assert(row_index >= 0 && row_index < rows);
        // Check if the size of the provided values matches the number of columns
        assert(vals.size() == cols);
        // Calculate the starting index in the flattened data vector for the specified row
        int start_idx = row_index * cols;
        // Set the values in the specified row
        for (int i = 0; i < cols; i++) {
            data[start_idx + i] = vals[i];
        }
    }

    // Function to get a row from the tensor
    std::vector<grad::Value> get_row(int row_index) {
        // Ensure the tensor is 2-dimensional
        assert(this->shape.size() == 2); 

        int rows = shape[0];  // Number of rows
        int cols = shape[1];  // Number of columns

        // Check if the provided row index is within bounds
        assert(row_index >= 0 && row_index < rows);

        // Calculate the starting index in the flattened data vector for the specified row
        int start_idx = row_index * cols;

        // Extract the row as a vector of grad::Value
        std::vector<grad::Value> row;
        row.reserve(cols);  // Reserve space for the row elements

        for (int i = 0; i < cols; i++) {
            row.push_back(data[start_idx + i]);  // Add each element of the row
        }

        return row;  // Return the extracted row
    }

    Tensor matmul(const Tensor& other) const;

    Tensor relu() const;

    Tensor tanh() const;

    Tensor multinomial(int num_samples = 1) const;

    Tensor softmax() const;

    size_t size() const { return data.size(); }

    const std::vector<int>& getShape() const { return shape; }

    void set_data(const std::vector<grad::Value>& vals) { data = vals; }

    const std::vector<grad::Value>& get_data() const { return data; }

    std::vector<int> get_shape() { return shape; }
};

Tensor Tensor::matmul(const Tensor& other) const {
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);

    Tensor result(static_cast<std::vector<int>>(shape[0], other.shape[1]));
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < other.shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < shape[1]; k++) {
                int a[2] = {i, k};
                int _a[2] = {k, j};
                sum +=  data[getIndex(std::vector<int>(a, a + 2))].getData() * 
                        other.data[other.getIndex(std::vector<int>(std::vector<int>(_a, _a + 2))) ].getData();
            }
            int a[2] = {i, j};
            result.data[result.getIndex(std::vector<int>(a, a + 2))] = grad::Value(sum);
        }
    }

    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = std::max(0.0f, data[i].getData());
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = std::tanh(data[i].getData());
    }
    return result;
}

Tensor Tensor::multinomial(int num_samples) const {
    assert(shape.size() == 1); // Ensure tensor is one-dimensional

    // Generate cumulative probabilities
    std::vector<float> cumulative_probs(data.size());
    cumulative_probs[0] = data[0].getData();
    for (size_t i = 1; i < data.size(); ++i) {
        cumulative_probs[i] = cumulative_probs[i - 1] + data[i].getData();
    }
    // Create tensor to store the sampled indices
    Tensor result(static_cast<std::vector<int>>(num_samples)); 
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, cumulative_probs.back());

    for (int sample = 0; sample < num_samples; ++sample) {
        float rand_prob = dis(gen); // Sample random probability
        auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), rand_prob);
        int sampled_index = std::distance(cumulative_probs.begin(), it);
        result[sample] = grad::Value(static_cast<float>(sampled_index));
    }

    return result;
}

Tensor Tensor::softmax() const {
    assert(shape.size() == 1);
    Tensor result(shape);
    float maxVal = (*std::max_element(data.begin(), data.end(), [](const grad::Value& a, const grad::Value& b) {
        return a.getData() < b.getData();
    })).getData();

    float sumExp = 0.0f;
    for (const auto& val : data) {
        sumExp += std::exp(val.getData() - maxVal);
    }

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = std::exp(data[i].getData() - maxVal) / sumExp;
    }

    return result;
}


}; // tensor

namespace nn {

class Embedding {
public:
    // Embedding table (V x D)
    tensor::Tensor table;  

    // Initialize the embedding table with random values
    void _init_(const std::vector<int>& dims) {
        table = tensor::Tensor(dims);  // Initialize the tensor with the given dimensions
        size_t size = table.size();
        std::vector<grad::Value> elements;
        for (size_t i = 0; i < size; i++) {
            // Initialize each element with a random value between 0 and 1
            elements.push_back(grad::Value(static_cast<double>(rand()) / RAND_MAX));
        }

        table.set_data(elements);  // Set the data of the tensor
    }

    // Forward pass: lookup embeddings for input indices
    tensor::Tensor forward(tensor::Tensor& input) {
        // Assume input is a 1-D tensor of indices
        std::vector<int> indices;  // Retrieve indices from the input tensor
        std::vector<grad::Value> vals = input.get_data();
        for (grad::Value val : vals) {
            indices.push_back(static_cast<int>(val.getData()));
        }
        
        int d[2] = {static_cast<int>(indices.size()), table.get_shape()[1]};
        std::vector<int> output_dims(d, d + 2);
        // Initialize output tensor with the shape (num_indices, embedding_dim)
        tensor::Tensor output(output_dims);  

        // Retrieve embeddings corresponding to each index in the input
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            std::vector<grad::Value> embedding = table.get_row(idx);  // Get the corresponding row from the table
            output.set_row(i, embedding);  // Set the retrieved row into the output tensor
        }

        return output;  // Return the tensor containing the embedding vectors
    }

    /// Backward pass: update embedding table gradients
    void backward(tensor::Tensor& input, tensor::Tensor& grad_output) {
        // Assume input is a 1-D tensor of indices
        std::vector<int> indices;  // Retrieve indices from the input tensor
        std::vector<grad::Value> vals = input.get_data();
        for (grad::Value val : vals) {
            indices.push_back(static_cast<int>(val.getData()));
        }
        // Retrieve the current data from the table tensor
        std::vector<grad::Value> table_data = table.get_data();
        // Each row of grad_output corresponds to the gradient of the loss with respect to the embedding
        int cols = table.get_shape()[1];  // Number of columns in the embedding table
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            std::vector<grad::Value> grad = grad_output.get_row(i);  // Get the gradient row corresponding to the embedding

            // Accumulate the gradient into the corresponding row of the embedding table
            for (size_t j = 0; j < grad.size(); j++) {
                int table_index = idx * cols + j;
                float g = table_data[table_index].getGrad();
                g += grad[j].getGrad();  // Update the gradient in the local copy of the data
                table_data[table_index].set_grad(g);
            }
        }
        // Set the modified data back into the table tensor
        table.set_data(table_data);
    }
};

class LinearLayer {
private:
    tensor::Tensor weights;
    float bias = 0.0f;

public:

    LinearLayer(tensor::Tensor& wei) : weights(wei) {}
    LinearLayer(tensor::Tensor& wei, float b) : weights(wei), bias(b) {}

    int get_size() { return weights.size(); }
    tensor::Tensor get_weights() { return weights; }
    void set_weights(tensor::Tensor& wei) { weights = wei; }

    LinearLayer(size_t size) : weights(std::vector<grad::Value>(size)) {
        weights_init_(size);
    }

    void weights_init_(size_t __size);
   
    tensor::Tensor feed_forward(tensor::Tensor& data);
};

void LinearLayer::weights_init_(size_t __size) {
    std::vector<float> wei(__size); 
    srand(static_cast<unsigned>(time(nullptr)));

    for (size_t i = 0; i < __size; i++) {
        wei[i] = (static_cast<float>(rand()) / (float)RAND_MAX * 2 - 1);
    }

    std::vector<grad::Value> vals;
    for (float w : wei) {
        vals.push_back(grad::Value(w));
    }

    weights.set_data(vals);
}

tensor::Tensor LinearLayer::feed_forward(tensor::Tensor& data) {
    assert(data.size() == weights.size());
    tensor::Tensor output = data.matmul(weights); 
    output = output.relu(); 

    if (bias != 0.0f) {
        std::vector<grad::Value> inter_data = output.get_data();
        for (grad::Value& val : inter_data) {
            val = val + bias; 
        } 
        output.set_data(inter_data);
    }

    return output;
}

class NeuralNetwork {
private:
    std::vector<grad::Value> parameters;
    Embedding lookup_table;
    LinearLayer hidden_layer;
    LinearLayer output_layer;
    bool is_training = true; 

public:
    NeuralNetwork(size_t input_size) 
    : hidden_layer(input_size), output_layer(1) {
        for (size_t i = 0; i < input_size * 2 + 1; i++) {
            parameters[i] = grad::Value(0.0f);
            int table_dimensions[2] = {vocab_size, 2};
            lookup_table._init_(std::vector<int>(table_dimensions, table_dimensions + 2));
        }
    }

    void set_eval() { is_training = false; }

    void set_train() { is_training = true; }

    tensor::Tensor feed_forward(tensor::Tensor& input_data);
   
    void backpropagate(tensor::Tensor& output, tensor::Tensor& targets, float learning_rate = 0.01);
   
    tensor::Tensor generate(tensor::Tensor idx, size_t max_tokens);
   
    float mse_loss(const tensor::Tensor& predicted, const tensor::Tensor& targets);    
};

// mean squared error loss (for simplicity)
float NeuralNetwork::mse_loss(const tensor::Tensor& predicted, const tensor::Tensor& targets) {
    // targets and logits should have the same shape
    assert(predicted.getShape() == targets.getShape());
    // initialize
    float sumSquaredDifferences = 0.0f;
    size_t numElements = predicted.size();

    for (size_t i = 0; i < numElements; i++) {
        // compare the non absolute difference betweent the raw values
        grad::Value diff = predicted.get_data()[i] - targets.get_data()[i];
        // add the difference to the loss sum
        sumSquaredDifferences += diff.getData() * diff.getData();
    }

    // return the mean average
    return sumSquaredDifferences / numElements;
}

// generate from the model
tensor::Tensor NeuralNetwork::generate(tensor::Tensor idx, size_t max_tokens)  {
    // generate the desired max tokens
    for (size_t i = 0; i < max_tokens; i++) {
        // compute the logits
        tensor::Tensor logits = feed_forward(idx);
        // softmax to get normalized probabilities
        tensor::Tensor probs = logits.softmax();
        // get the indexing 
        tensor::Tensor next = probs.multinomial();
        // append the token
        idx = idx + next;
    }

    return idx;
}

// forward pass of the network 
tensor::Tensor NeuralNetwork::feed_forward(tensor::Tensor& input_data) {
    size_t input_size = input_data.size();
    // look up the tokens indexing 
    tensor::Tensor logits = lookup_table.forward(input_data);
    // push through the dense hidden layer
    logits = hidden_layer.feed_forward(logits);
    // our beautiful activation function
    logits.tanh();
    // get the output layer activations
    output_layer.feed_forward(logits);
    // return the raw logits - softmaxed at the generation phase
    return logits;
}

// backward pass of the neural net
void NeuralNetwork::backpropagate(tensor::Tensor& output, tensor::Tensor& targets, float learning_rate) {
    // Compute Mean Squared Error loss
    float loss = mse_loss(output, targets);
    // Compute the gradient of the loss with respect to the output
    tensor::Tensor loss_grad = output - targets;
    loss_grad = loss_grad * (1.0f / output.size());
    // Gradient of logits after softmax
    tensor::Tensor d_logits = output.softmax();
    // Update weights of the output layer
    tensor::Tensor output_layer_weights = output_layer.get_weights();
    std::vector<grad::Value> output_layer_weights_vals = output_layer_weights.get_data();

    for (size_t i = 0; i < output_layer_weights_vals.size(); i++) {
        float grad = d_logits.get_data()[i].getGrad();
        output_layer_weights_vals[i] = output_layer_weights_vals[i] - (learning_rate * grad);
        output_layer_weights_vals[i].set_grad(0.0f);
    }

    tensor::Tensor updated_output_weights(output_layer_weights_vals);
    output_layer.set_weights(updated_output_weights);

    // Update weights of the hidden layer
    tensor::Tensor hidden_grad = hidden_layer.feed_forward(output);
    tensor::Tensor hidden_layer_weights = hidden_layer.get_weights();
    std::vector<grad::Value> hidden_layer_weights_vals = hidden_layer_weights.get_data();

    for (size_t i = 0; i < hidden_layer_weights_vals.size(); i++) {
        float grad = hidden_grad.get_data()[i].getGrad();
        hidden_layer_weights_vals[i] = hidden_layer_weights_vals[i] - (learning_rate * grad);
        hidden_layer_weights_vals[i].set_grad(0.0f);
    }

    tensor::Tensor updated_hidden_weights(hidden_layer_weights_vals);
    hidden_layer.set_weights(updated_hidden_weights);

    // Backward pass for the lookup table (embedding layer)
    // Compute gradients with respect to the embeddings
    tensor::Tensor lookup_grad = lookup_table.forward(hidden_grad);  // Assume this propagates gradients properly
    std::vector<grad::Value> lookup_grad_vals = lookup_grad.get_data();

    // Retrieve the current embedding data
    std::vector<grad::Value> lookup_table_vals = lookup_table.table.get_data();

    // Update embedding values using the gradients
    for (size_t i = 0; i < lookup_grad_vals.size(); i++) {
        // Retrieve the index of the embedding to update
        int index = hidden_grad.get_data()[i].getIndex(); 
        float grad = lookup_grad_vals[i].getGrad();
        // Update the embedding value at the corresponding index
        lookup_table_vals[index] = lookup_table_vals[index] - (learning_rate * grad);
        lookup_table_vals[index].set_grad(0.0f);  // Reset the gradient after update
    }

    // Set the updated embeddings back into the lookup table
    lookup_table.table.set_data(lookup_table_vals);
}


}; // nn

class Data_loader {
public:
    // to keep track of the backwards mapping of the encoding - to decode
    std::unordered_map<int, char> global_idx;
    std::unordered_map<char, int> char_to_index;
    int vocab_size = 0;

public:

    // sample a batch at random
    std::vector<std::stack<std::vector<int>>> get_batch(std::vector<int>& data) {
        // generate a random index for the offset where the batch begins
        int rand_offset = rand() % (data.size() - (context_length) * batch_size);
        // stack the inputs and targets samples
        std::stack<std::vector<int>> inputs;
        std::stack<std::vector<int>> targets;
        // initialize
        size_t data_size = data.size();
        size_t chunck_begin = rand_offset;
        size_t chunck_end = rand_offset + context_length;

        for (int i = 0; i < batch_size; i++) {
            // if the batch ends beyond the data size we get a seg fault
            if (chunck_end >= data_size) { break; }
            // copy the chunks
            std::vector<int> input_chunck(data.begin() + chunck_begin, data.begin() + chunck_end);
            std::vector<int> target_chunck(data.begin() + chunck_begin + 1, data.begin() + chunck_end + 1);
            // add them to the stack
            inputs.push(input_chunck);
            targets.push(target_chunck);
            // increment the indices
            chunck_begin += context_length;
            chunck_end += context_length;
        }
        // group the batches together 
        std::vector<std::stack<std::vector<int>>> result;
        // maintaining order in the call side is important 
        result.push_back(inputs);
        result.push_back(targets);
    
        return result;
    }

    // get the file data in string form
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

    // encode a string to ints
    std::vector<int> encode(const std::string& str) {
        std::vector<int> result;
        char_to_index.clear();
        int index = 0;

        for (char c : str) {
            if (char_to_index.find(c) == char_to_index.end()) {
                char_to_index[c] = index++;
            }
            result.push_back(char_to_index[c]);
        }

        global_idx.clear(); 
        for (const auto& pair : char_to_index) {
            global_idx[pair.second] = pair.first;
        }

        return result;
    }

    // decode string
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int n : tokens) {
            result.push_back(global_idx[n]);
        }
        return result;
    }
};

// estimate the loss to evaluate the model at a given phase of the training
std::unordered_map<std::string, float> estimate_loss(
    nn::NeuralNetwork& model, 
    std::vector<int> train_data, 
    std::vector<int> val_data
) {
    Data_loader loader;
    std::unordered_map<std::string, float> out;
    std::vector<std::string> splits;
    splits.push_back("train"); // estimate for the training data
    splits.push_back("val"); // ```` for the validation data

    for (const std::string& split : splits) {
        std::vector<float> losses(eval_iters, 0.0f);
        
        for (int k = 0; k < eval_iters; ++k) {
            std::vector<std::stack<std::vector<int>>> batches = loader.get_batch(split == "train" ? train_data:val_data);
            std::stack<std::vector<int>> inputs = batches[0];
            std::stack<std::vector<int>> targets = batches[1];
            tensor::Tensor inputs_as_tensor;
            tensor::Tensor targets_as_tensor;
            for (int i = 0, n = inputs.size(); i < n; i++) {
                std::vector<grad::Value> vals;
                for (int val : inputs.top()) {
                    vals.push_back(grad::Value(static_cast<float>(val)));
                }
                inputs_as_tensor = inputs_as_tensor + tensor::Tensor(vals);
                inputs.pop();
            }

            for (int i = 0, n = targets.size(); i < n; i++) {
                std::vector<grad::Value> vals;
                for (int val : targets.top()) {
                    vals.push_back(grad::Value(static_cast<float>(val)));
                }
                targets_as_tensor = targets_as_tensor + tensor::Tensor(vals);
                targets.top();
            }

            tensor::Tensor logits = model.feed_forward(inputs_as_tensor);
            float loss = model.mse_loss(logits, targets_as_tensor);
            losses[k] = loss;
        }
        
        float mean_loss = std::accumulate(losses.begin(), losses.end(), 0.0f) / eval_iters;
        out[split] = mean_loss;
    }

    return out;
}

void train_model(nn::NeuralNetwork& model, std::vector<int>& data) {
    Data_loader loader;
    size_t data_size = data.size();
    size_t train_size = static_cast<size_t>(data_size * 0.9);
    std::vector<int> train_data(data.begin(), data.begin() + train_size);
    std::vector<int> val_data(data.begin() + train_size, data.end());

    for (int iter = 0; iter < max_iterations; iter++) {
        if (iter % eval_interval == 0) {
            auto losses = estimate_loss(model, train_data, val_data);
            std::cout << "step "       << iter          << " : train loss " << losses["train"] 
                      << ", val loss " << losses["val"] << std::endl;
        }

        std::vector<std::stack<std::vector<int>>> batches = loader.get_batch(train_data);
        std::stack<std::vector<int>> input_batches = batches[0];
        std::stack<std::vector<int>> target_batches = batches[1];
        tensor::Tensor input_as_tensor;
        tensor::Tensor target_as_tensor;

        for (int i = 0, n = input_batches.size(); i < n; i++) {
            std::vector<grad::Value> vals;
            for (int val : input_batches.top()) {
                vals.push_back(grad::Value(static_cast<float>(val)));
            }
            input_as_tensor = input_as_tensor + tensor::Tensor(vals);
            input_batches.pop();
        }

        for (int i = 0, n = target_batches.size(); i < n; i++) {
            std::vector<grad::Value> vals;
            for (int val : target_batches.top()) {
                vals.push_back(grad::Value(static_cast<float>(val)));
            }
            target_as_tensor = target_as_tensor + tensor::Tensor(vals);
            target_batches.pop();
        }

        tensor::Tensor logits = model.feed_forward(input_as_tensor);
        model.backpropagate(logits, target_as_tensor);
    }
}


int main(void) {
    Data_loader loader;
    std::string input = loader.read_file("input.txt");
    nn::NeuralNetwork model(input.size());
    std::vector<int> encoded = loader.encode(input);
    train_model(model, encoded);
    return 0;
}
