#include <colza/Backend.hpp>
#include <spdlog/spdlog.h>
#include <thallium.hpp>
#include <mona.h>
#include <mona-mpi.h>
#include <mpi.h>
#include <ascent.hpp>
#include <map>
#include <mutex>

namespace tl = thallium;
using json = nlohmann::json;
using namespace std::string_literals;

class AscentPipeline : public colza::Backend {

    enum class CommType {
        MPI,
        MONA
    };

    public:

    AscentPipeline(const colza::PipelineFactoryArgs& args)
    : m_engine(args.engine) {
        if(args.config.contains("ascent_options")) {
            auto options = args.config["ascent_options"].dump();
            m_ascent_options.parse(options);
        }
        if(args.config.contains("comm_type")) {
            if(!args.config["comm_type"].is_string()) {
                throw std::runtime_error("AscentPipeline configuration: comm_type should be a string");
            }
            auto comm_type = args.config["comm_type"].get<std::string>();
            if(comm_type == "mpi") {
                m_comm_type = CommType::MPI;
            } else if(comm_type == "mona") {
                m_comm_type = CommType::MONA;
            } else {
                throw std::runtime_error("AscentPipeline configuration: comm_type should be \"mona\" or \"mpi\"");
            }
        }
        if(args.config.contains("log_wrapped_calls")) {
            if(!args.config["log_wrapped_calls"].is_boolean())
                throw std::runtime_error("AscentPipeline configuration: log_wrapped_calls should be a boolean");
            if(args.config["log_wrapped_calls"].get<bool>()) {
                MPI_Mona_enable_logging();
            }
        }
    }

    void updateMonaAddresses(mona_instance_t mona, const std::vector<na_addr_t>& addresses) override {
        std::lock_guard<tl::mutex> g(m_mona_comm_mtx);
        if(m_mona_comm_latest != nullptr) {
            mona_comm_free(m_mona_comm_latest);
            m_mona_comm_latest = nullptr;
        }
        auto na_ret = mona_comm_create(mona, addresses.size(), addresses.data(), &m_mona_comm_latest);
        if(na_ret != NA_SUCCESS) {
            throw std::runtime_error("mona_comm_create returned error "s + std::to_string((int)na_ret));
        }
    }

    colza::RequestResult<int32_t> start(uint64_t iteration) override {
        auto result      = colza::RequestResult<int32_t>{};
        result.value()   = 0;
        result.success() = true;
        spdlog::trace("AscentPipeline::start() called with iteration {}", iteration);
        if(m_comm_type == CommType::MONA) {
            std::lock_guard<tl::mutex> g(m_mona_comm_mtx);
            auto nret = mona_comm_dup(m_mona_comm_latest, &m_mona_comm);
            if(nret != 0) {
                throw std::runtime_error("mona_comm_dup failed");
            }
            auto ret = MPI_Register_mona_comm(m_mona_comm, &m_mpi_comm);
            if(ret != 0) {
                throw std::runtime_error("MPI_Register_mona_comm failed");
            }
            m_ascent_options["mpi_comm"] = MPI_Comm_c2f(m_mpi_comm);
        } else {
            m_ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
        }
        spdlog::trace("AscentPipeline::updateMonaAddresses succeeded");
        return result;
    }

    void abort(uint64_t iteration) override {
        (void)iteration;
        if(m_comm_type == CommType::MONA) {
            if(m_mona_comm) {
                MPI_Comm_free(&m_mpi_comm);
                mona_comm_free(m_mona_comm);
            }
            m_mona_comm = nullptr;
        }
        spdlog::trace("AscentPipeline::abort() called with iteration {}", iteration);
    }

    colza::RequestResult<int32_t> stage(const std::string& sender_addr,
            const std::string& dataset_name, uint64_t iteration, uint64_t block_id,
            const std::vector<size_t>& dimensions, const std::vector<int64_t>& offsets,
            const colza::Type& type, const thallium::bulk& remote_data) override {
        spdlog::trace("AscentPipeline::stage() called with iteration {}", iteration);
        auto result = colza::RequestResult<int32_t>{};
        (void)offsets;
        (void)type;
        if(dimensions.size() != 1) {
            result.error() = "Unexpected number of dimensions";
            result.success() = false;
            return result;
        }
        // Do RDMA transfer from client
        auto size = dimensions[0];
        std::string data_str(size, '\0');
        std::vector<std::pair<void*, size_t>> segments = {std::make_pair((void*)data_str.data(), size)};
        auto local_bulk = m_engine.expose(segments, tl::bulk_mode::write_only);
        auto origin_ep = m_engine.lookup(sender_addr);
        remote_data.on(origin_ep) >> local_bulk;
        // Convert into conduit Node
        conduit::Node data;
        data.parse(data_str, "conduit_base64_json");
        // Store data
        std::lock_guard<tl::mutex> g(m_data_mtx);
        m_data[iteration][dataset_name].update(std::move(data));
        spdlog::trace("AscentPipeline::stage() completed iteration {}", iteration);
        return result;
    }

    colza::RequestResult<int32_t> execute(uint64_t iteration) override {
        auto result = colza::RequestResult<int32_t>{};
        spdlog::trace("AscentPipeline::execute() called with iteration {}", iteration);
        ascent::Ascent ascent;
        ascent.open(m_ascent_options);
        ascent.publish(m_data[iteration]["mesh"]);
        conduit::Node a; // actions are actually defined in options (via actions_file)
        ascent.execute(a);
        ascent.close();
        spdlog::trace("AscentPipeline::execute() completed iteration {}", iteration);
        return result;
    }

    colza::RequestResult<int32_t> cleanup(uint64_t iteration) override {
        if(m_comm_type == CommType::MONA) {
            if(m_mona_comm) {
                MPI_Comm_free(&m_mpi_comm);
                mona_comm_free(m_mona_comm);
            }
            m_mona_comm = nullptr;
        }
        auto result = colza::RequestResult<int32_t>{};
        spdlog::trace("AscentPipeline::cleanup() called iteration {}", iteration);
        std::lock_guard<tl::mutex> g(m_data_mtx);
        m_data.erase(iteration);
        result.value() = 0;
        return result;
    }

    colza::RequestResult<int32_t> destroy() override {
        auto result = colza::RequestResult<int32_t>{};
        spdlog::trace("AscentPipeline::destroy() called iteration");
        result.value() = 0;
        result.success() = true;
        return result;
    }

    static std::unique_ptr<colza::Backend> create(const colza::PipelineFactoryArgs& args) {
        return std::make_unique<AscentPipeline>(args);
    }

    private:

    // Mochi setup
    tl::engine     m_engine;

    // Collective communication
    CommType m_comm_type           = CommType::MPI;
    MPI_Comm m_mpi_comm            = MPI_COMM_NULL;
    mona_comm_t m_mona_comm        = nullptr;
    mona_comm_t m_mona_comm_latest = nullptr;
    tl::mutex   m_mona_comm_mtx;

    // Data
    std::map<uint64_t,          // iteration
        std::map<std::string,   // dataset name
                conduit::Node>> m_data;
    tl::mutex                   m_data_mtx;

    // Processing
    conduit::Node m_ascent_options;
};

COLZA_REGISTER_BACKEND(ascent, AscentPipeline);
