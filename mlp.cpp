#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

const int batch_size     = 16;
const int context_length = 8;
const int max_iterations = 3000000;
const int eval_interval  = 300;
const int eval_iters     = 200;
const int layer_size     = 100;
const int embd_dim       = 32;

class DataLoader {
   private:
    std::unordered_map<int, char> global_idx;
    std::unordered_map<char, int> char_to_index;
    int                           vocab_size = 0;

   public:
    DataLoader() {}

    int get_vocab_size() { return global_idx.size(); }

    std::vector<int> encode(const std::string& str) {
        std::vector<int> result;
        char_to_index.clear();
        int index = 0;

        for (char c : str)
        {
            if (char_to_index.find(c) == char_to_index.end())
            {
                char_to_index[c] = index++;
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

    std::string decode(torch::Tensor& tensor) const {
        tensor = tensor.to(torch::kInt32);
        std::string result;
        result.reserve(tensor.size(0));

        for (size_t i = 0; i < tensor.size(0); ++i)
        {
            int index = tensor[i].item<int>();
            if (global_idx.find(index) != global_idx.end())
            {
                result.push_back(global_idx.at(index));
            }
            else
            {
                result.push_back('?');
            }
        }

        return result;
    }

    std::string read_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::in);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for read");
        }

        std::stringstream buf;
        buf << file.rdbuf();
        file.close();
        return buf.str();
    }

    std::tuple<torch::Tensor, torch::Tensor> get_batch(const torch::Tensor& data) {
        torch::Tensor inputs    = torch::empty({batch_size, context_length}, torch::kInt32);
        torch::Tensor targets   = torch::empty({batch_size}, torch::kInt64);
        size_t        data_size = data.size(0);

        if (data_size <= context_length)
        {
            throw std::runtime_error("Data size must be greater than context length");
        }

        size_t rand_offset =
          static_cast<size_t>(std::rand()) % (data_size - context_length * batch_size);
        size_t chunk_begin = rand_offset;
        size_t chunk_end   = rand_offset + context_length;

        for (size_t i = 0; i < batch_size; i++)
        {
            if (chunk_end + 1 > data_size)
            {
                throw std::runtime_error("Chunk end exceeds data size.");
            }

            inputs[i]  = data.slice(0, chunk_begin, chunk_end).to(torch::kInt32);
            targets[i] = data[chunk_end].item<int64_t>();

            chunk_begin += context_length;
            chunk_end += context_length;
        }

        return std::make_tuple(inputs, targets);
    }

    std::tuple<std::vector<int>, std::vector<int>> split_data(const std::vector<int>& data) {
        size_t           size = static_cast<size_t>(data.size() * 0.9);
        std::vector<int> train_data(data.begin(), data.begin() + size);
        std::vector<int> val_data(data.begin() + size, data.end());
        return std::make_tuple(train_data, val_data);
    }
};

class LanguageModel: public torch::nn::Module {
   public:
    torch::nn::Embedding lookup_table{nullptr};
    torch::nn::Linear    hidden_layer{nullptr};
    torch::nn::Linear    output_layer{nullptr};

    LanguageModel(int vocab_size) :
        lookup_table(register_module("lookup_table", torch::nn::Embedding(vocab_size, embd_dim))),
        hidden_layer(register_module("hidden_layer",
                                     torch::nn::Linear(context_length * embd_dim, layer_size))),
        output_layer(register_module("output_layer", torch::nn::Linear(layer_size, vocab_size))) {}

    torch::Tensor feed_forward(torch::Tensor input) {
        input = input.to(torch::kLong);
        input = lookup_table->forward(input);
        input = input.view({input.size(0), -1});

        if (input.size(1) != context_length * embd_dim)
        {
            std::cerr << "Error: Mismatch in input dimension after embedding. "
                      << "Expected " << context_length * embd_dim << ", got " << input.size(1)
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        input = hidden_layer->forward(input);
        input = torch::relu(input);
        input = output_layer->forward(input);
        return input;
    }

    std::unordered_map<std::string, float>
    estimate_loss(DataLoader& data_loader, torch::Tensor train_data, torch::Tensor val_data) {
        std::unordered_map<std::string, float> loss_map;

        for (const std::string& split : {"train", "val"})
        {
            float         total_loss = 0.0;
            torch::Tensor data       = (split == "train") ? train_data : val_data;

            for (int k = 0; k < eval_iters; k++)
            {
                auto [inputs, targets]  = data_loader.get_batch(data);
                torch::Tensor predicted = feed_forward(inputs);
                torch::Tensor loss      = torch::nn::functional::cross_entropy(predicted, targets);
                total_loss += loss.item<float>();
            }

            float avg_loss  = total_loss / static_cast<float>(eval_iters);
            loss_map[split] = avg_loss;
        }

        return loss_map;
    }

    void train_model(DataLoader& data_loader, const std::vector<int>& data) {
        size_t data_size            = data.size();
        auto [train_data, val_data] = data_loader.split_data(data);
        torch::Tensor train_tensor  = torch::tensor(train_data, torch::kInt);
        torch::Tensor val_tensor    = torch::tensor(val_data, torch::kInt);

        torch::optim::AdamW optimizer(parameters(), torch::optim::AdamWOptions(0.001));

        for (int iter = 0; iter < max_iterations; iter++)
        {
            if (iter % eval_interval == 0)
            {
                auto losses = estimate_loss(data_loader, train_tensor, val_tensor);
                std::cout << "Step " << iter << ": Train Loss = " << losses["train"]
                          << ", Val Loss = " << losses["val"] << std::endl;
            }

            auto [inputs, targets] = data_loader.get_batch(train_tensor);
            optimizer.zero_grad();
            torch::Tensor logits = feed_forward(inputs);
            torch::Tensor loss   = torch::nn::functional::cross_entropy(logits, targets);
            loss.backward();
            optimizer.step();
        }
    }

    std::string
    generate(DataLoader& data_loader, const std::vector<int>& context, int max_length = 50) {
        std::vector<int> generated = context;
        torch::Tensor    input     = torch::tensor(generated, torch::kInt32).unsqueeze(0);

        if (input.size(1) < context_length)
        {
            std::cerr << "Error: Initial context length is smaller than "
                         "context_length."
                      << std::endl;
            return "";
        }

        for (int i = 0; i < max_length; ++i)
        {
            if (input.size(1) > context_length)
            {
                input = input.slice(1, input.size(1) - context_length, input.size(1));
            }

            torch::Tensor logits        = feed_forward(input);
            torch::Tensor probabilities = torch::softmax(logits, -1);
            torch::Tensor next_token    = probabilities.argmax(1);

            int next_token_id = next_token.item<int>();
            generated.push_back(next_token_id);
            input = torch::cat({input, next_token.unsqueeze(0)}, 1);
            if (input.size(1) > context_length)
            {
                input = input.slice(1, input.size(1) - context_length, input.size(1));
            }
        }
        torch::Tensor as_tensor = torch::tensor(generated, torch::kInt32);
        return data_loader.decode(as_tensor);
    }
};

int main() {
    try
    {
        DataLoader       data_loader;
        std::string      input_data = data_loader.read_file("input.txt");
        std::vector<int> data       = data_loader.encode(input_data);

        int           vocab_size = data_loader.get_vocab_size();
        LanguageModel model(vocab_size);
        model.train_model(data_loader, data);

        std::string      str            = "love is a very thing";
        std::vector<int> context        = data_loader.encode(str);
        std::string      generated_text = model.generate(data_loader, context, 50);
        std::cout << "Generated Text: " << generated_text << std::endl;
    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
