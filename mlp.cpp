#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <tuple>
#include <vector>

// hyperparameters
const int batch_size     = 16;
const int eval_iters     = 200;
const int context_length = 8;
const int eval_interval  = 300;
const int max_iterations = 100000;

// helpers
template<typename T>
void append_matrix(std::vector<std::vector<T>>& mat, const std::vector<std::vector<T>>& to_add) {
    size_t __size = mat.size() + to_add.size();
    mat.resize(__size);

    for (unsigned int k = 0, i = mat.size(); i < __size; i++, k++)
    {
        mat[i] = to_add[k];
    }
}

template<typename T>
void append_vector(std::vector<T>& vec, const std::vector<T>& to_add) {
    size_t __size = vec.size() + to_add.size();

    for (unsigned int k = 0, i = vec.size(); i < __size; i++, k++)
    {
        vec[i] = to_add[k];
    }
}

class Data_loader {
   public:
    // to keep track of the backwards mapping of the encoding - to decode
    std::unordered_map<int, char> global_idx;
    std::unordered_map<char, int> char_to_index;
    int                           vocab_size = 0;

   public:
    // sample a batch at random
    Data_loader() = default;

    int get_vocab_size() const { return vocab_size; }

    std::tuple<std::vector<std::vector<int>>, std::vector<int>> get_batch(std::vector<int>& data) {
        // generate a random index for the offset where the batch begins
        int rand_offset = rand() % (data.size() - (context_length) *batch_size);
        // stack the inputs and targets samples
        std::vector<std::vector<int>> inputs;
        std::vector<int>              targets;
        // initialize
        size_t data_size    = data.size();
        size_t chunck_begin = rand_offset;
        size_t chunck_end   = rand_offset + context_length;

        for (int i = 0; i < batch_size; i++)
        {
            // if the batch ends beyond the data size we get a seg fault
            if (chunck_end >= data_size)
            {
                break;
            }
            // copy the chunks
            std::vector<int> input_chunck(data.begin() + chunck_begin, data.begin() + chunck_end);
            // add them to the stack
            inputs.push_back(input_chunck);
            targets.push_back(data[chunck_end + 1]);
            // increment the indices
            chunck_begin += context_length;
            chunck_end += context_length;
        }

        return std::make_tuple(inputs, targets);
    }

    std::string read_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        std::stringstream buf;
        buf << file.rdbuf();
        return buf.str();
    }

    // encode a string to ints
    std::vector<int> encode(const std::string& str) {
        std::vector<int> result;
        char_to_index.clear();
        int index = 0;
        vocab_size = 0;

        for (char c : str)
        {
            if (char_to_index.find(c) == char_to_index.end())
            {
                char_to_index[c] = index++;
                vocab_size++;
            }
            result.push_back(char_to_index[c]);
        }

        global_idx.clear();
        for (const auto& pair : char_to_index)
        {
            global_idx[pair.second] = pair.first;
        }

        return result;
    }

    // decode string
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int n : tokens)
        {
            result.push_back(global_idx[n]);
        }
        return result;
    }
};

// thanks to 'micrograd' by 'Andrej Karpathy' which is open on github
namespace data {

class Value {
   public:
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

    // constructors
    Value() :
        data(0.0f),
        grad(0.0f),
        _op(' ') {}

    Value(const Value* val) :
        data(val->data),
        grad(val->data),
        index(val->index),
        _op(val->_op),
        _prev(val->_prev) {}

    Value(const Value& val) :
        data(val.data),
        grad(val.grad),
        _prev(val._prev),
        _op(val._op) {}

    Value(Value& val) :
        data(val.data),
        grad(val.data),
        _prev(val._prev),
        _op(val._op) {}

    Value(Value* val) :
        data(val->data),
        grad(val->grad),
        _prev(val->_prev),
        _op(val->_op) {}

    Value(float d) :
        data(d),
        grad(0.0f),
        _op(' ') {}

    Value(float d, const std::vector<std::shared_ptr<Value>>& children, char op) :
        data(d),
        grad(0.0f),
        _prev(children),
        _op(op) {}

    Value(Value&& other) noexcept :
        data(other.data),
        grad(other.grad),
        _prev(std::move(other._prev)),
        _op(other._op),
        index(other.index) {
        // Set the moved-from object to a default state
        other.data  = 0.0f;
        other.grad  = 0.0f;
        other._op   = ' ';
        other.index = 0;
    }

    // getters
    int getIndex() const { return index; }

    /*---operators---*/
    // for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& val) {
        os << val.data;
        return os;
    }

    // Copy assignment operator
    Value& operator=(const Value& other) {
        if (this != &other)
        {  // self-assignment check
            this->data  = static_cast<float>(other.data);
            this->grad  = static_cast<float>(other.grad);
            this->_prev = static_cast<std::vector<std::shared_ptr<Value>>>(other._prev);
            this->_op   = static_cast<char>(other._op);
            this->index = static_cast<int>(other.index);
        }

        return *this;
    }

    // Move assignment operator
    Value& operator=(Value&& other) noexcept {
        if (this != &other)
        {
            // Handle assignment
            if (other.data != 0.0f || other.grad != 0.0f || other._op != ' ')
            {
                data  = other.data;
                grad  = other.grad;
                _prev = std::move(other._prev);
                _op   = other._op;
                index = other.index;

                // Set the moved-from object to a valid state
                other.data  = 0.0f;
                other.grad  = 0.0f;
                other._op   = ' ';
                other.index = 0;
            }
            else
            {
                return other;
            }
        }
        return *this;
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
        Value res((data - other.data), children, '-');
        return res;
    }

    // Value * Value
    Value operator*(const Value& other) const {
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        children.push_back(std::make_shared<Value>(other));
        Value res((data * other.data), children, '*');
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
    float operator/(const Value& other) const { return data / other.data; }

    // Value(this) ** Value(other)
    Value _pow(const Value& other) {
        // using smart pointer is just simply smarter
        std::vector<std::shared_ptr<Value>> children;
        children.push_back(std::make_shared<Value>(*this));
        data = pow(data, other.data);
        Value res(data, children, '^');
        return res;
    }

    // how to go backward depends on the operation
    void _backward() {
        if (_op == ' ' || _prev.empty() || !_prev[0])
        {
            std::cerr << "Debug : There is a problem in the backward function" << std::endl;
            return;
        }
        else if (_op == '+')
        {
            // '+' distributes the gradient equally
            // (a + b)' = a' + b'
            _prev[0]->grad += grad;
            _prev[1]->grad += grad;
        }
        else if (_op == '*')
        {
            // (a * b)' = a * b' + a' * b
            _prev[0]->grad += _prev[1]->data * grad;
            _prev[1]->grad += _prev[0]->data * grad;
        }
        else if (_op == '^')
        {
            // (a ^ b)' = b * (a ^ (b - 1))
            float exponent = _prev[1]->data;
            _prev[0]->grad += exponent * std::pow(_prev[0]->data, exponent - 1) * grad;
        }
        else if (_op == 'r')
        {
            // relu'(x) = (x > 0) ? 1 : 0
            _prev[0]->grad += (_prev[0]->data > 0) * grad;
        }
    }

    // traverse the built map
    void backward() {
        std::list<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>>  visited;
        build_topo(topo, visited, std::make_shared<Value>(*this));
        this->grad = 1.0f;

        topo.reverse();
        for (std::shared_ptr<Value> v : topo)
        {
            v->_backward();
        }
    }

   private:
    // build the topology of the map you should
    // travel to get to the beginning aka weights
    void build_topo(std::list<std::shared_ptr<Value>>& topo,
                    std::set<std::shared_ptr<Value>>&  visited,
                    std::shared_ptr<Value>             node) {

        if (visited.find(node) == visited.end())
        {
            visited.insert(node);
            for (auto child : node->_prev)
            {
                build_topo(topo, visited, child);
            }
            topo.push_back(node);
        }
    }
};

std::vector<std::vector<Value>> relu(std::vector<std::vector<Value>>& data) {
    std::vector<std::vector<Value>> result;
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t j = 0; j < data[i].size(); j++)
        {
            result[i][j] = std::max(0.0f, data[i][j].data);
        }
    }

    return result;
}

std::vector<std::vector<Value>> tanh(std::vector<std::vector<Value>>& data) {
    std::vector<std::vector<Value>> result;
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t j = 0; j < data[i].size(); j++)
        {
            result[i][j] = std::tanh(data[i][j].data);
        }
    }

    return result;
}

std::vector<Value> multinomial(std::vector<Value>& data, int num_samples) {
    // Generate cumulative probabilities
    std::vector<float> cumulative_probs(data.size());
    cumulative_probs[0] = data[0].data;

    for (size_t i = 1; i < data.size(); ++i)
    {
        cumulative_probs[i] = cumulative_probs[i - 1] + data[i].data;
    }

    // Create vector to store the sampled indices
    std::vector<Value> result;
    result.reserve(num_samples);

    // Random number generator
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0.0, cumulative_probs.back());

    for (int sample = 0; sample < num_samples; sample++)
    {
        float rand_prob = dis(gen);  // Sample random probability
        auto  it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), rand_prob);
        int   sampled_index = std::distance(cumulative_probs.begin(), it);
        result[sample]      = Value(static_cast<float>(sampled_index));
    }

    return result;
}

std::vector<Value> softmax(std::vector<Value>& data) {
    std::vector<Value> result;
    float              maxVal =
      (*std::max_element(data.begin(), data.end(),
                         [](const Value& a, const Value& b) { return a.data < b.data; }))
        .data;

    float sumExp = 0.0f;
    for (const auto& val : data)
    {
        sumExp += std::exp(val.data - maxVal);
    }

    for (size_t i = 0; i < data.size(); i++)
    {
        result[i] = std::exp(data[i].data - maxVal) / sumExp;
    }

    return result;
}

// Function template to convert a 1D vector into a 2D matrix of specified dimensions
template<typename T>
std::vector<std::vector<T>> vectorToMatrix(const std::vector<T>& vec, int rows, int cols) {
    // Check if the provided dimensions are compatible with the size of the
    // vector
    if (rows * cols != vec.size())
    {
        throw std::invalid_argument("The number of elements in the vector does not "
                                    "match the specified matrix dimensions.");
    }
    // Initialize a 2D vector (matrix) with the specified number of rows and
    // columns
    std::vector<std::vector<T>> matrix(rows, std::vector<T>(cols));

    // Fill the matrix with elements from the 1D vector
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = vec[i * cols + j];  // Mapping 1D index to 2D indices
        }
    }

    return matrix;
}

std::vector<std::vector<Value>> matmul(const std::vector<std::vector<Value>>& A,
                                       const std::vector<std::vector<Value>>& B) {
    // Get dimensions of matrices
    int rowsA = A.size();     // Number of rows in A
    int colsA = A[0].size();  // Number of columns in A
    int colsB = B[0].size();  // Number of columns in B

    // Initialize result matrix C with the appropriate size (rowsA x colsB)
    // filled with zeros
    std::vector<std::vector<Value>> C(rowsA, std::vector<Value>(colsB, 0.0f));

    // Perform matrix multiplication
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            for (int k = 0; k < colsA; ++k)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

};  // namespace data


namespace nn {

class Embedding {
   public:
    // Embedding table (V x D)
    std::vector<std::vector<data::Value>> table;

    Embedding() = default;

    // Initialize the embedding table with random values
    Embedding(const int dims[2]) {
        assert(dims[0] > 0 && dims[1] > 0);

        table.resize(dims[0], std::vector<data::Value>(dims[1]));
        srand(time(0));

        for (size_t i = 0; i < dims[0]; i++)
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
                table[i][j] = data::Value(static_cast<double>(rand()) / RAND_MAX);
            }
        }
    }

    // Forward pass: lookup embeddings for input indices
    std::vector<std::vector<data::Value>> table_forward(const std::vector<int>& input_indices) {
        if (input_indices.empty())
        {
            throw std::runtime_error("Cannot forward pass empty set of indices");
        }

        if (table.empty() || table[0].empty())
        {
            throw std::runtime_error("Cannot forward pass with an empty lookup table");
        }
        int num_indices   = static_cast<int>(input_indices.size());
        int embedding_dim = static_cast<int>(table[0].size());

        // Initialize output tensor with the shape (num_indices, embedding_dim)
        std::vector<std::vector<data::Value>> output(num_indices,
                                                     std::vector<data::Value>(embedding_dim));

        // Retrieve embeddings corresponding to each index in the input
        for (size_t i = 0; i < input_indices.size(); i++)
        {
            int idx = input_indices[i];

            // Ensure the index is within bounds
            if (idx >= table.size() || idx < 0)
            {
                throw std::out_of_range("Index out of range in embedding table.");
            }

            // Get the corresponding row from the table
            std::vector<data::Value> embedding = table[idx];

            // Set the retrieved row into the output tensor
            output[i] = embedding;
        }

        return output;  // Return the tensor containing the embedding vectors
    }

    // Backward pass: update embedding table gradients
    void backward(const std::vector<int>& input_indices,  // Input should be indices
                  const std::vector<std::vector<data::Value>>& grad_output) {

        // Each row of grad_output corresponds to the gradient of the loss with
        // respect to the embedding
        for (size_t i = 0; i < input_indices.size(); i++)
        {
            int idx = input_indices[i];  // Get the index from input

            if (idx >= table.size() || idx < 0)
            {
                throw std::out_of_range(
                  "Index out of range in embedding table during backward pass.");
            }

            const std::vector<data::Value>& grad =
              grad_output[i];  // Get the gradient row corresponding to the embedding
            std::vector<data::Value>& embedding = table[idx];  // Get the embedding to update

            // Accumulate the gradient into the corresponding row of the embedding table
            for (size_t j = 0; j < grad.size(); j++)
            {
                embedding[j].grad +=
                  grad[j].grad;  // Update gradient for each element in the embedding
            }
        }
    }
};

class LinearLayer {
   private:
    std::vector<std::vector<data::Value>> weights;
    float                                 bias = 0.0f;

   public:
    LinearLayer() = default;
    LinearLayer(std::vector<int> dims) :
        weights(std::vector<std::vector<data::Value>>(dims[0], std::vector<data::Value>(dims[1]))) {
    }
    LinearLayer(std::vector<std::vector<data::Value>> wei) :
        weights(wei) {}
    LinearLayer(std::vector<std::vector<data::Value>> wei, float b) :
        weights(wei),
        bias(b) {}
    LinearLayer(size_t size) :
        weights(std::vector<std::vector<data::Value>>(size)) {
        weights_init_(size);
    }

    int                                   get_size() { return weights.size(); }
    std::vector<std::vector<data::Value>> get_weights() { return weights; }
    void set_weights(std::vector<std::vector<data::Value>> wei) { weights = wei; }

    void weights_init_(size_t __size) {
        for (int i = 0; i < weights.size(); i++)
        {
            for (int j = 0; j < weights[i].size(); j++)
            {
                srand(static_cast<unsigned>(time(nullptr)));
                weights[i][j] = (static_cast<float>(rand()) / (float) RAND_MAX * 2 - 1);
            }
        }
    }

    std::vector<std::vector<data::Value>>
    feed_forward(std::vector<std::vector<data::Value>>& data) {
        if (data.size() != weights.size())
        {
            throw std::runtime_error("Weights size should be equal to input size");
        }

        std::vector<std::vector<data::Value>> output = matmul(weights, data);
        output                                       = relu(output);

        if (bias != 0.0f)
        {
            for (size_t i = 0; i < output.size(); i++)
            {
                for (size_t j = 0; j < output[i].size(); j++)
                {
                    output[i][j] = output[i][j] + bias;
                }
            }
        }

        return output;
    }
};

class NeuralNetwork {
   private:
    std::vector<data::Value> parameters;
    Embedding                lookup_table;
    LinearLayer              hidden_layer;
    LinearLayer              output_layer;
    bool                     is_training = true;

   public:
    NeuralNetwork(size_t input_size, int vocab_size) :
        hidden_layer(input_size),
        output_layer(1) {

        if (input_size <= 0)
        {
            throw std::runtime_error(
              "'input_size' cannot be less then or equal to zero for initializing the neural net");
        }

        if (vocab_size <= 0)
        {
            throw std::runtime_error(
              "'vocab_size' cannot be less then or equal to zero for initializing the neural net");
        }

        for (size_t i = 0; i < input_size * 2 + 1; i++)
        {
            parameters.push_back(data::Value(0.0f));
        }


        int table_dimensions[2] = {vocab_size, 2};
        lookup_table            = Embedding(table_dimensions);
    }

    void set_eval() { is_training = false; }

    void set_train() { is_training = true; }

    // forward pass of the network
    std::vector<data::Value> feed_forward(std::vector<int>& input_data) {
        size_t input_size = input_data.size();
        // look up the tokens indexing
        std::vector<std::vector<data::Value>> logits = lookup_table.table_forward(input_data);
        // push through the dense hidden layer
        logits = hidden_layer.feed_forward(logits);
        // our beautiful activation function
        tanh(logits);
        // get the output layer activations
        output_layer.feed_forward(logits);
        // return the raw logits - softmaxed at the generation phase
        std::vector<data::Value> ans;
        for (int i = 0; i < logits.size(); i++)
        {
            for (int j = 0; j < logits[i].size(); j++)
            {
                ans.push_back(logits[i][j]);
            }
        }
        return ans;
    }

    // backward pass of the neural net
    void backpropagate(std::vector<data::Value>& output,
                       std::vector<data::Value>& targets,
                       float                     learning_rate) {

        assert(output.size() == targets.size());
        // Compute Mean Squared Error loss
        float loss = mse_loss(output, targets);
        // Compute the gradient of the loss with respect to the output
        std::vector<data::Value> loss_grad;

        for (int i = 0; i < output.size(); i++)
        {
            loss_grad.push_back(output[i] - targets[i]);
        }

        for (int i = 0; i < loss_grad.size(); i++)
        {
            loss_grad[i].data *= (1.0f / output.size());
        }

        // Gradient of logits after softmax
        std::vector<data::Value> d_logits = data::softmax(output);
        // Update weights of the output layer
        std::vector<std::vector<data::Value>> output_layer_weights = output_layer.get_weights();

        for (size_t i = 0; i < output_layer_weights.size(); i++)
        {
            for (size_t j = 0; j < output_layer_weights[i].size(); j++)
            {
                output_layer_weights[i][j].data -= learning_rate * output_layer_weights[i][j].grad;
                output_layer_weights[i][j].grad = 0.0f;
            }
        }

        // Update weights of the hidden layer
        // TODO : convert the output vector into a matrix of the weights
        // dimensions
        std::vector<std::vector<data::Value>> mat =
          data::vectorToMatrix(output, batch_size, context_length);
        std::vector<std::vector<data::Value>> hidden_grad          = hidden_layer.feed_forward(mat);
        std::vector<std::vector<data::Value>> hidden_layer_weights = hidden_layer.get_weights();

        // update the weights based on the learning rate
        for (size_t i = 0; i < hidden_layer_weights.size(); i++)
        {
            for (size_t j = 0; j < hidden_layer_weights[i].size(); j++)
            {
                hidden_layer_weights[i][j].data -=
                  (learning_rate * hidden_layer_weights[i][j].grad);
                hidden_layer_weights[i][j].grad = 0.0f;
            }
        }

        // Backward pass for the lookup table (embedding layer)
        // Compute gradients with respect to the embeddings
        std::vector<std::vector<data::Value>> lookup_grad;

        for (unsigned int i = 0; i < hidden_grad.size(); i++)
        {
            std::vector<int> indices;
            for (int j = 0; j < hidden_grad[i].size(); j++)
            {
                indices.push_back(static_cast<int>(hidden_grad[i][j].data));
            }

            append_matrix<data::Value>(lookup_grad, lookup_table.table_forward(indices));
        }

        // Update embedding values using the gradients
        for (size_t i = 0; i < lookup_grad.size(); i++)
        {
            for (size_t j = 0; j < lookup_grad[i].size(); j++)
            {
                int index = hidden_grad[i][j].getIndex();
                lookup_table.table[index][j].data -= (learning_rate * lookup_grad[i][j].grad);
            }
        }
    }

    std::vector<int> generate(std::vector<int>& idx, size_t max_tokens) {
        std::vector<int> cat;
        // generate the desired max tokens
        for (size_t i = 0; i < max_tokens; i++)
        {
            // compute the logits
            std::vector<data::Value> logits = feed_forward(idx);
            // softmax to get normalized probabilities
            std::vector<data::Value> probs = data::softmax(logits);
            // get the indexing
            std::vector<data::Value> next = data::multinomial(probs, idx.size());
            // append the token
            cat.reserve(idx.size() + next.size());
            int k = 0;
            for (data::Value val : idx)
            {
                cat[k++] = static_cast<int>(val.data);
            }

            for (data::Value val : next)
            {
                cat[k++] = static_cast<int>(val.data);
            }
        }

        return cat;
    }

    // mean squared error loss (for simplicity)
    float mse_loss(const std::vector<data::Value>& predicted,
                   const std::vector<data::Value>& targets) {
        // initialize
        float  sumSquaredDifferences = 0.0f;
        size_t numElements           = predicted.size();

        for (size_t i = 0; i < numElements; i++)
        {
            // compare the non absolute difference betweent the raw values
            data::Value diff = predicted[i].data - targets[i].data;
            // add the difference to the loss sum
            sumSquaredDifferences += diff.data * diff.data;
        }

        // return the mean average
        return sumSquaredDifferences / numElements;
    }
};

};  // namespace nn

std::unordered_map<std::string, float>
estimate_loss(nn::NeuralNetwork& model, std::vector<int> train_data, std::vector<int> val_data) {
    Data_loader                            loader;
    std::unordered_map<std::string, float> out;

    for (const std::string& split : {"train", "val"})
    {
        std::vector<float> losses(eval_iters, 0.0f);
        for (int k = 0; k < eval_iters; k++)
        {
            auto [inputs, targets] = loader.get_batch(split == "train" ? train_data : val_data);
            std::vector<data::Value> converted_targets;
            for (int val : targets)
            {
                converted_targets.push_back(data::Value(static_cast<float>(val)));
            }

            std::vector<data::Value> logits;
            for (unsigned int i = 0; i < batch_size; i++)
            {
                append_vector<data::Value>(logits, model.feed_forward(inputs[i]));
            }

            float loss = model.mse_loss(logits, converted_targets);
            losses[k]  = loss;
        }

        float mean_loss = std::accumulate(losses.begin(), losses.end(), 0.0f) / eval_iters;
        out[split]      = mean_loss;
    }

    return out;
}

void train_model(nn::NeuralNetwork& model, std::vector<int>& data) {
    Data_loader      loader;
    size_t           data_size  = data.size();
    size_t           train_size = static_cast<size_t>(data_size * 0.9);
    std::vector<int> train_data(data.begin(), data.begin() + train_size);
    std::vector<int> val_data(data.begin() + train_size, data.end());

    for (int iter = 0; iter < max_iterations; iter++)
    {
        if (iter % eval_interval == 0)
        {
            auto losses = estimate_loss(model, train_data, val_data);
            std::cout << "step " << iter << ": train_loss " << losses["train"] << ", val loss"
                      << losses["val"] << std::endl;
        }

        auto [inputs, targets] = loader.get_batch(train_data);

        std::vector<data::Value> logits;
        for (unsigned int i = 0; i < batch_size; i++)
        {
            append_vector<data::Value>(logits, model.feed_forward(inputs[i]));
        }

        std::vector<data::Value> converted_targets;

        for (int val : targets)
        {
            converted_targets.push_back(data::Value(static_cast<float>(val)));
        }

        model.backpropagate(logits, converted_targets, 0.01f);
    }
}

int main(void) {
    Data_loader loader;
    std::string input = loader.read_file("/Users/mac/lmc/input_.txt");

    if (input.empty())
    {
        std::cout << "Cannot initialize with empty input!" << std::endl;
        return 0;
    }

    std::vector<int>  encoded = loader.encode(input);
    nn::NeuralNetwork model(input.length(), loader.get_vocab_size());
    train_model(model, encoded);

    return 0;
}