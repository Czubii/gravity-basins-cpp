#ifndef CPU_RENDERER
#define CPU_RENDERER

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <SFML/Graphics.hpp>
#include "bodies.h"

/// @brief the CPU rendere to render gravity basins. make sure to call updateOutput() every frame in main thread for the output texture to update correctly, use getTexture() to get a reference to the render texture
class CpuBasinsRenderer {
private:
    int renderThreadNum;
    std::vector<std::thread> renderThreads;
    std::vector<StaticBody>& staticBodies; 
    int outputWidth;
    int outputHeight;
    float renderScale;

    std::atomic<bool> updateRequiredFlag;
    std::atomic<bool> stopRenderFlag;
    std::atomic<int> threadsRunning;

    sf::Image sharedRenderImage;
    std::mutex imageUpdateMutex;
    sf::Texture outputTexture; 

    void cleanUpRenderThreads();
    void RenderThread(std::vector<StaticBody>& bodies, int threadID, std::vector<int>::iterator pixelIndicesStart, std::vector<int>::iterator pixelIndicesEnd, int updateThreshold);

public:
    bool rendering;
    std::chrono::steady_clock::time_point renderStartTime;

    CpuBasinsRenderer(int _outputWidth, int _outputHeight, std::vector<StaticBody>& _staticBodies, float _renderScale, int _renderThreadNum = 4);

    /// @brief returns a reference to the rendering output texture
    /// @return reference to rendering output texture
    sf::Texture& getTexture();

    /// @brief UNINMPLEMENTED
    /// @return returns true if randering started succesfully, false otherwise
    bool startRendering();

    /// @brief render frame, updates the output every updateOutputThreshold pixels
    /// @param updateOutputThreshold after how many pixels should the output update
    /// @return returns true if randering started succesfully, false otherwise
    bool startLiveRendering(int updateOutputThreshold);

    /// @brief stops active render threads, waits for them to join;
    void stopRendering();

    /// @brief updates the output if any new data is available
    /// @return returns true if the output has been updated and rendering is done, otherwise false
    bool updateOutput();
};

sf::Texture createBodiesTexture(std::vector<StaticBody> bodies);
sf::Texture createTrajectortTexture(Trajectory trajectory, float pointRadius, sf::Color color);

#endif