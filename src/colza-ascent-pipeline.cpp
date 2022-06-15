#include <colza/Backend.hpp>
#include <thallium.hpp>
#include <mona.h>
#include <mona-mpi.h>
#include <mpi.h>
#include <ascent.hpp>
#include <map>

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
    : m_engine(args.engine)
    , m_group_id(args.gid)
    , m_config(args.config) {
        // TODO
    }

    void updateMonaAddresses(mona_instance_t mona, const std::vector<na_addr_t>& addresses) override {
        if(m_comm_type == CommType::MPI) {
            m_mpi_comm = MPI_COMM_WORLD;
            return;
        }
        if(m_mpi_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&m_mpi_comm);
            m_mpi_comm = MPI_COMM_NULL;
        }
        if(m_mona_comm != nullptr) {
            mona_comm_free(m_mona_comm);
            m_mona_comm = nullptr;
        }
        auto na_ret = mona_comm_create(mona, addresses.size(), addresses.data(), &m_mona_comm);
        if(na_ret != NA_SUCCESS) {
            throw std::runtime_error("mona_comm_create returned error "s + std::to_string((int)na_ret));
        }
        auto ret = MPI_Register_mona_comm(m_mona_comm, &m_mpi_comm);
        if(ret != 0) {
            throw std::runtime_error("MPI_Register_mona_comm failed");
        }
    }

    colza::RequestResult<int32_t> start(uint64_t iteration) override {
        auto result      = colza::RequestResult<int32_t>{};
        result.value()   = 0;
        result.success() = true;
        return result;
    }

    void abort(uint64_t iteration) override {
        (void)iteration;
    }

    colza::RequestResult<int32_t> stage(const std::string& sender_addr,
            const std::string& dataset_name, uint64_t iteration, uint64_t block_id,
            const std::vector<size_t>& dimensions, const std::vector<int64_t>& offsets,
            const colza::Type& type, const thallium::bulk& data) override {
        auto result = colza::RequestResult<int32_t>{};
        // TODO
        return result;
    }

    colza::RequestResult<int32_t> execute(uint64_t iteration) override {
        auto result = colza::RequestResult<int32_t>{};
        // TODO
        return result;
    }

    colza::RequestResult<int32_t> cleanup(uint64_t iteration) override {
        auto result = colza::RequestResult<int32_t>{};
        std::lock_guard<tl::mutex> g(m_data_mtx);
        m_data.erase(iteration);
        result.value() = 0;
        return result;
    }

    colza::RequestResult<int32_t> destroy() override {
        auto result = colza::RequestResult<int32_t>{};
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
    ssg_group_id_t m_group_id;
    json           m_config;

    // Collective communication
    CommType m_comm_type    = CommType::MPI;
    MPI_Comm m_mpi_comm     = MPI_COMM_NULL;
    mona_comm_t m_mona_comm = nullptr;

    // Data
    std::map<uint64_t,          // iteration
        std::map<std::string,   // dataset name
            std::map<uint64_t,  // block id
                conduit::Node>>> m_data;
    tl::mutex                    m_data_mtx;
};

COLZA_REGISTER_BACKEND(ascent, AscentPipeline);
