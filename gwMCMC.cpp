#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <array>

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/SyncType.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Pad.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/timer/timer.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poprand;

static const int nIPUs = 2;
static const int nParams = 2;
static const int nWalkers = 100;
static const int nIterations = 100;
static const int nExchange = 1;

std::random_device rd;
std::mt19937 mt(rd());

// From https://github.com/graphcore/examples/blob/master/code_examples/poplar/advanced_example/utils.h
poplar::Device getIpuHwDevice(std::size_t numIpus) {

    auto dm = poplar::DeviceManager::createDeviceManager();
    auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);

    if (hwDevices.size() > 0) {
        for (auto &d : hwDevices) {
            if (d.attach()) {
                return std::move(d);
            }
        }
    }

    throw std::runtime_error("No IPU hardware available.");
}

std::vector<poplar::Device> * getIpuHwDevices(std::size_t numIpus) {

    auto dm = poplar::DeviceManager::createDeviceManager();
    auto hwDevices = dm.getDevices(poplar::TargetType::IPU, 1);

    std::vector<poplar::Device> * devices = new std::vector<poplar::Device>();

    if (hwDevices.size() > 0) {
        for (auto &d : hwDevices) {
            if (d.attach()) {
                devices->push_back(std::move(d));
            }
        }
    }

    return devices;
}

std::vector<float> prepareData(float mu, float sigma, uint n)
{
    std::random_device rd;

    std::mt19937 e(rd());
    std::normal_distribution<> normal(mu, sigma);

    std::vector<float> out(n);
    for (auto &o : out) o = normal(mt);

    return out;
}

std::vector<std::array<float, nParams>> prepareParams(std::array<float, nParams> init,
                                                      int nWalkers,
                                                      float sigma)
{
    std::normal_distribution<> normal(0.0, sigma);

    std::vector<std::array<float, nParams>> params(nWalkers);

    for (auto & w : params) {
        for (int i = 0; i < nParams; i++) {
            w[i] = init[i] + normal(mt);
        }
    }

    return params;

}

std::array<std::array<int, nExchange>, nWalkers> prepareExchangeIndices()
{
    std::array<std::array<int, nExchange>, nWalkers> exchangeIndices;
    for (auto &i : exchangeIndices) for (auto &j : i) j = -1;

    std::uniform_int_distribution<> uniformWalkers(0, nWalkers - 1);

    // This is 'slow', but is fast enough for all reasoanble input sizes

    for (auto &w : exchangeIndices) {

        for (auto &e : w) {

            int idx = uniformWalkers(mt);

            while(std::find(w.begin(), w.end(), idx) != w.end()) idx = uniformWalkers(mt);

            e = idx;
        }
    }

    return exchangeIndices;
}

int main(int argc, char const *argv[]) {

    std::vector<Device> * devices = getIpuHwDevices(2);

    Device dev = std::move((*devices)[0]);

    Graph graph(dev.getTarget());

    popops::addCodelets(graph);
    poprand::addCodelets(graph);
    graph.addCodelets("GWVertex.cpp");

    // Only happens once
    Sequence initSeq;

    // Repeats
    Sequence inSeq;
    Sequence outSeq;

    uint nData = 10000;

    // For selecting a random walker
    std::uniform_int_distribution<> uniformWalkers(0, nWalkers - 1);

    // Generate data set - G(0, 1)
    std::vector<float> dataVec = prepareData(0.0, 1.0, nData);

    // Initialise parameters - G(0.3, 1.4)
    std::array<float, nParams> paramOrigin = {0.3, 1.4};
    std::vector<std::array<float, nParams>> paramInit = prepareParams(paramOrigin, nWalkers, 0.01);
    std::array<std::array<int, nExchange>, nWalkers> exchangeIndices = prepareExchangeIndices();

    for (auto i: exchangeIndices[0]) std::cout << i << ",";
    std::cout << std::endl;
    for (auto i: exchangeIndices[1]) std::cout << i << ",";
    std::cout << std::endl;
    for (auto i: exchangeIndices[2]) std::cout << i << ",";
    std::cout << std::endl;

    // Graph params
    std::vector<Tensor> data(nWalkers);
    std::vector<Tensor> out(nWalkers);
    std::vector<Tensor> randRef(nWalkers);
    std::vector<Tensor> randSeed(nWalkers);
    std::vector<Tensor> rand(nWalkers);
    std::vector<Tensor> randOtherRef(nWalkers);
    std::vector<Tensor> randOther(nWalkers);
    std::vector<Tensor> thisTheta(nWalkers);
    std::vector<Tensor> otherTheta(nWalkers);

    std::vector<DataStream> dataStream(nWalkers);
    std::vector<DataStream> initStream(nWalkers);

    std::vector<ComputeSet> gwCS(nWalkers);
    std::vector<VertexRef> gwVtx(nWalkers);

    for (int tile = 0; tile < nWalkers; tile++) {

        std::string tileStr = std::to_string(tile);

        data[tile] = graph.addVariable(FLOAT, {nData});
        graph.setTileMapping(data[tile], tile);

        dataStream[tile] = graph.addHostToDeviceFIFO("dataFIFO_" + tileStr, FLOAT, nData);
        initSeq.add(Copy(dataStream[tile], data[tile]));

        // Output of GW vertex, but this is where the initialisation goes
        // so the first iteration doesn't need to be a special case
        out[tile] = graph.addVariable(FLOAT, {2});
        graph.setTileMapping(out[tile], tile);

        initStream[tile] = graph.addHostToDeviceFIFO("initFIFO" + tileStr, FLOAT, nParams);
        initSeq.add(Copy(initStream[tile], out[tile]));

        randRef[tile] = graph.addConstant<float>(FLOAT, {2}, {4.2, 4.2});
        graph.setTileMapping(randRef[tile], tile);
        randSeed[tile] = graph.addConstant<float>(UNSIGNED_INT, {2}, {42, 42});
        graph.setTileMapping(randSeed[tile], tile);

        // Without null seed this uses seed on each iteration, can't change that, so use NULL!
        rand[tile] = poprand::uniform(graph, NULL, 0, randRef[tile], FLOAT, 0.0, 1.0, inSeq);
        graph.setTileMapping(rand[tile], tile);

        randOtherRef[tile] = graph.addConstant<int>(INT, {1}, {0});
        graph.setTileMapping(randOtherRef[tile], tile);

        randOther[tile] = poprand::uniform(graph, NULL, 0, randOtherRef[tile], INT, 0, nExchange - 1, inSeq);
        graph.setTileMapping(randOther[tile], tile);

        thisTheta[tile] = graph.addVariable(FLOAT, {2});
        graph.setTileMapping(thisTheta[tile], tile);

        otherTheta[tile] = graph.addVariable(FLOAT, {nExchange, 2});
        graph.setTileMapping(otherTheta[tile], tile);

        gwCS[tile] = graph.addComputeSet("GW");
        gwVtx[tile] = graph.addVertex(gwCS[tile], "GW");
        graph.setCycleEstimate(gwVtx[tile], 100);
        graph.setTileMapping(gwVtx[tile], tile);

        graph.connect(gwVtx[tile]["data"], data[tile]);
        graph.connect(gwVtx[tile]["uniformRand"], rand[tile]);
        graph.connect(gwVtx[tile]["otherRand"], randOther[tile]);
        graph.connect(gwVtx[tile]["thisTheta"], thisTheta[tile]);
        graph.connect(gwVtx[tile]["otherThetas"], otherTheta[tile]);
        graph.connect(gwVtx[tile]["out"], out[tile]);

    }

    for (int tile = 0; tile < nWalkers; tile++) {

        // Separate loop, in case out[rand] location doesn't exist yet
        // Before: generate random uniform
        inSeq.add(Copy(out[tile], thisTheta[tile]));

        // Copy selected 'other' tile parameters to slice of the input array for this tile
        for (int otherIdx = 0; otherIdx < nExchange ; otherIdx++) {
            int fromIndex = exchangeIndices[tile][otherIdx];
            inSeq.add(Copy(out[fromIndex], otherTheta[tile].slice(otherIdx, otherIdx + 1, 0)));
        }

        outSeq.add(Execute(gwCS[tile]));
    }

    Sequence progMain;
    progMain.add(initSeq);

    for (int i = 0; i < nIterations; i++) {
        progMain.add(inSeq);
        progMain.add(Sync(poplar::SyncType::INTERNAL));
        progMain.add(outSeq);
    }

    // Serialization of compiled graph
    std::ofstream exeFile("mcmcTest", std::ios::binary);
    Executable exe = compileGraph(graph, {progMain});
    exe.serialize(exeFile);

    // std::ifstream exeFile("mcmcTest", std::ios::binary);
    // Executable exe = Executable::deserialize(exeFile);

    Engine engine(std::move(exe));

    engine.load(dev);

    for (int tile = 0; tile < nWalkers; tile++) {
        engine.connectStream(dataStream[tile], &(dataVec[0]), &(dataVec[nData]));
        engine.connectStream(initStream[tile], &(paramInit[tile][0]), &(paramInit[tile][nParams]));
    }

    // Warmup run

    engine.run(0);

    // Timing run

    {
    boost::timer::auto_cpu_timer t;
    engine.run(0);
    }

    return 0;
}
