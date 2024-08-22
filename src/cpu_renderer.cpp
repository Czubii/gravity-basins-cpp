#include "cpu_renderer.h"

#include "bodies.h"
#include <thread>
#include <atomic>
#include <SFML/Graphics.hpp>
#include <random>
#include <iostream>
#include <parameters.h>
#include <mutex>

using namespace std;

vector<int> randomisedIntVector(const int size){
    vector<int> output;
    for(int i =0; i<size; i++)
    output.push_back(i);

    // Create a random number generator
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator

    // Define the range for random numbers
    std::uniform_int_distribution<> distr(0, size); // Range [1, 100]

    for(int i =0; i<size; i++)
    {
        // Generate and a random number
        int random_number = distr(eng);
        swap(output[i], output[random_number]);
    }

    return output;

}


CpuBasinsRenderer::CpuBasinsRenderer(int _outputWidth, int _outputHeight, std::vector<StaticBody>& _staticBodies, float _renderScale, int _renderThreadNum)
    : outputWidth(_outputWidth),
      outputHeight(_outputHeight),
      staticBodies(_staticBodies),
      renderScale(_renderScale),
      renderThreadNum(_renderThreadNum),
      updateRequiredFlag(false),
      stopRenderFlag(false),
      threadsRunning(0),
      rendering(false) {
    sharedRenderImage.create(outputWidth, outputHeight, sf::Color::Black);
    
    outputTexture.loadFromImage(sharedRenderImage);
    outputTexture.setSmooth(true);
}

sf::Texture& CpuBasinsRenderer::getTexture(){
    return outputTexture;
}

    void CpuBasinsRenderer::cleanUpRenderThreads(){
        for(auto & rernderThread: renderThreads){
            if(rernderThread.joinable()){
                rernderThread.join();
            }
        }
        renderThreads.clear();
    }


    void CpuBasinsRenderer::RenderThread(vector<StaticBody>& bodies, int threadID, std::vector<int>::iterator pixelIndicesStart, std::vector<int>::iterator pixelIndicesEnd, int updateThreshold){

    sf::Image renderBufferImage;
    renderBufferImage.create(outputWidth, outputHeight, sf::Color(0, 0, 0, 0));

    int currentPixel = 0;
    for (auto PixelIndexIt = pixelIndicesStart; PixelIndexIt != pixelIndicesEnd; ++PixelIndexIt){
            if (stopRenderFlag.load()){
                cout << endl << "render thread closed prematurely" << endl;
                return;
            }
            
            if(updateThreshold !=0)
            {if(currentPixel % updateThreshold == 0)
            {
                // Lock the mutex to safely update the image 
                lock_guard<std::mutex> lock(imageUpdateMutex);
                //update shared display image  
                sharedRenderImage.copy(renderBufferImage, 0, 0, sf::IntRect(0,0,0,0), true);
                // Flag that the image needs to be updated
                updateRequiredFlag.store(true);
                
            }}

            int y = *PixelIndexIt / outputWidth;
            int x = *PixelIndexIt % outputWidth;

            StaticBody crashingBody = getCrashingBody(bodies, {(float)x / renderScale, (float)y / renderScale}, 15000, 20);
            renderBufferImage.setPixel(x, y, crashingBody.color);

            currentPixel ++;
    }
    // Lock the mutex to safely update the image
    {
            // Lock the mutex to safely update the image        
        lock_guard<std::mutex> lock(imageUpdateMutex);
                    
            //update shared display image  
        sharedRenderImage.copy(renderBufferImage, 0, 0, sf::IntRect(0,0,0,0), true);

        threadsRunning--;
        // Flag that the image needs to be updated
        updateRequiredFlag.store(true);
    }

}




    bool CpuBasinsRenderer::startRendering(){//TODO startRendering() function for non-live render
        return false;
    }



    bool CpuBasinsRenderer::startLiveRendering(int updateOutputThreshold = UPDATE_EVERY){//TODO: implement the threshold

        if(rendering){
            cout << "failed to start rendering: previous renderer still running";
            return false;
        }
        else if (renderThreads.size() != 0)
        {
            cout << "failed to start rendering: render threads not yet cleaned up after previous call";
            return false;
        }

        threadsRunning.store(0);

        int pixelsPerThread = (int)(outputWidth*outputHeight/renderThreadNum);
        static vector<int> randomisedIndices = randomisedIntVector(outputWidth*outputHeight);
        for(int i = 0; i < renderThreadNum; i++){
            vector<int>::iterator start = randomisedIndices.begin() + i*pixelsPerThread;

            renderThreads.emplace_back([this, start, i, pixelsPerThread, updateOutputThreshold]() {
            this->RenderThread(staticBodies, i, start, start + pixelsPerThread, updateOutputThreshold);
            });

            threadsRunning++;
        }

        renderStartTime = chrono::high_resolution_clock::now();
        rendering = true;
        return true;

    }


    void CpuBasinsRenderer::stopRendering(){
        stopRenderFlag.store(true);

        cleanUpRenderThreads();

        rendering = false;
    }


    bool CpuBasinsRenderer::updateOutput(){//TODO: consider returning true when the rendering has finished insead of when the output has been updated
        if(updateRequiredFlag.load()){
            {
                std::lock_guard<std::mutex> lock(imageUpdateMutex);

                outputTexture.update(sharedRenderImage); 
            }

            updateRequiredFlag.store(false);

            //chcek if all threads finished, join them
            if (threadsRunning.load() == 0){

                cleanUpRenderThreads();
                rendering = false;


                chrono::duration<double> elapsed_seconds = chrono::high_resolution_clock::now() - renderStartTime;
                cout << "Rendering finnished. Time elapsed: " << elapsed_seconds.count() << " seconds" << endl;

                return true;
            }

        }
        return false;
    }






sf::Texture createBodiesTexture(vector<StaticBody> bodies){

    // Create a RenderTexture to draw bodies
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH*RENDER_SCALE, WINDOW_HEIGHT*RENDER_SCALE); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 

    for(auto& body: bodies){
        body.render(renderTexture);
    }

    //finalize the render
    renderTexture.display();

    // Create a Texture from the RenderTexture, and return it
    return renderTexture.getTexture();
}

sf::Texture createTrajectortTexture(Trajectory trajectory, float pointRadius = 1, sf::Color color = sf::Color::Red){ //FIXME: name of this abomination, create class Trajectory instead
    // Create a RenderTexture to draw trajectory points
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 

    for(const auto& point: trajectory.points)
    {
        sf::CircleShape circle(pointRadius * RENDER_SCALE);
        circle.setPosition(point * RENDER_SCALE);
        circle.setOrigin(sf::Vector2f(pointRadius * RENDER_SCALE, pointRadius * RENDER_SCALE));
        circle.setFillColor(color);

        renderTexture.draw(circle);
    }

    renderTexture.display();

    return renderTexture.getTexture();

}