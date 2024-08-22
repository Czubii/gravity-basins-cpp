#include <iostream>
#include <vector>
#include <string>
#include <SFML/Graphics.hpp>
#include <array>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>  // For int32_t
#include <random>
#include "bodies.h"
#include "parameters.h"
#include "cpu_renderer.h"

using namespace std;




// float particle_mass = 100;



// atomic<bool> stopFlag(false);

// sf::Image sharedRenderImage;
// sf::Texture sharedRenderTexture;
// sf::Sprite gravityBasinsSprite;
// atomic<bool> updateRequired(false);
// mutex imageMutex;
// atomic<int> finishedThreads(0);

// struct Trajectory
// {
//     vector<sf::Vector2f> points;
// };



// float fastInvSqrt(float number) {
//     if (number <= 0) return 0;  // Handle non-positive input
//     float x = number;
//     float x_half = 0.5f * x;
//     int32_t i = *reinterpret_cast<int32_t*>(&x);
//     i = 0x5f3759df - (i >> 1);  // Initial approximation
//     x = *reinterpret_cast<float*>(&i);
//     x = x * (1.5f - (x_half * x * x));  // Refine approximation with Newton's method
//     return x;
// }

// float fastSqrt(float number) {
//     return 1.0f / fastInvSqrt(number);
// }


// float getDistanceSquared(const sf::Vector2f& point1, const sf::Vector2f& point2) {
//     float dx = point2.x - point1.x;
//     float dy = point2.y - point1.y;
//     return dx * dx + dy * dy;
// }

// float getDistance(const sf::Vector2f& point1, const sf::Vector2f& point2) {
//     return fastSqrt(getDistanceSquared(point1, point2));
// }

// sf::Vector2f gravityForce(StaticBody body, sf::Vector2f pos){
//     float distanceSquared = getDistanceSquared(body.pos, pos);

//     sf::Vector2f direction = (body.pos - pos)/sqrt(distanceSquared);
//     float magnitude = GRAVITY_CONSTANT * body.mass / distanceSquared;

//     return direction * magnitude;
// }

// sf::Vector2f netGravityForce(vector<StaticBody> bodies, sf::Vector2f pos){
//     sf::Vector2f force = {0, 0};


//     for(const auto& body:bodies){
//         force += gravityForce(body, pos);
//     }

//     return force;
// }

// bool colidesWithAny(vector<StaticBody> bodies, sf::Vector2f pos){
//     for(const auto& body: bodies){
//         if (getDistanceSquared(body.pos, pos) <= body.radius * body.radius)
//         return true;
//     }
//     return false;
// }

// Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2f startPos, int maxSize = 15000, 
//                               float stepSize = 1, bool detectCollisions = true){
//     Trajectory traj;

//     traj.points.push_back(startPos);

//     sf::Vector2f pos = startPos;
//     sf::Vector2f vel = {0,0};
//     for (int step = 0; step < maxSize; step++){

//         if (detectCollisions){
//             if(colidesWithAny(bodies, pos))
//             return traj;
//         }

//         vel += netGravityForce(bodies, pos)/particle_mass;
//         pos += vel*stepSize;

//         traj.points.push_back(pos);
//     }

//     return traj;
// }

// Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2i startPos, int maxSize = 15000, 
//                               float stepSize = 1, bool detectCollisions = true){
//                                 sf::Vector2f startPosF = {(float)startPos.x, (float)startPos.y};
//                                 return generateTrajectory(bodies, startPosF, maxSize, stepSize, detectCollisions);
//                               }

// StaticBody getCrashingBody(vector<StaticBody> bodies, sf::Vector2f start_pos, int maxSize = 15000, float stepSize = 1){

//     sf::Vector2f pos = start_pos;
//     sf::Vector2f vel = {0,0};
//     for (int step = 0; step < maxSize; step++){
//         if(colidesWithAny(bodies, pos))

//             for(const auto& body: bodies){
//                 if (getDistanceSquared(body.pos, pos) <= body.radius * body.radius)
//                 return body;
//         }

//         vel += netGravityForce(bodies, pos)/particle_mass;
//         pos += vel*stepSize;
//     }

//     if(!MUTE_STEP_LIMIT_INFO)
//     cout << "no crashes for point: (" << start_pos.x << ", " << start_pos.y << ") after " << maxSize << " steps" << endl;

//     //return a new static body with black color for simplicty
//     return StaticBody({0,0},0, sf::Color::Black);
// }

// sf::Texture createBodiesTexture(vector<StaticBody> bodies){

//     // Create a RenderTexture to draw bodies
//     sf::RenderTexture renderTexture;
//     renderTexture.create(WINDOW_WIDTH*RENDER_SCALE, WINDOW_HEIGHT*RENDER_SCALE); // Use the same size as the image
//     renderTexture.clear(sf::Color::Transparent); 

//     for(auto& body: bodies){
//         body.render(renderTexture);
//     }

//     //finalize the render
//     renderTexture.display();

//     // Create a Texture from the RenderTexture, and return it
//     return renderTexture.getTexture();
// }

// sf::Texture createTrajectortTexture(Trajectory trajectory, float pointRadius = 1, sf::Color color = sf::Color::Red){
//     // Create a RenderTexture to draw trajectory points
//     sf::RenderTexture renderTexture;
//     renderTexture.create(WINDOW_WIDTH * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE); // Use the same size as the image
//     renderTexture.clear(sf::Color::Transparent); 

//     for(const auto& point: trajectory.points)
//     {
//         sf::CircleShape circle(pointRadius * RENDER_SCALE);
//         circle.setPosition(point * RENDER_SCALE);
//         circle.setOrigin(sf::Vector2f(pointRadius * RENDER_SCALE, pointRadius * RENDER_SCALE));
//         circle.setFillColor(color);

//         renderTexture.draw(circle);
//     }

//     renderTexture.display();

//     return renderTexture.getTexture();

// }
// vector<int> randomisedIntVector(const int size){
//     vector<int> output;
//     for(int i =0; i<size; i++)
//     output.push_back(i);

//     // Create a random number generator
//     std::random_device rd; // Obtain a random number from hardware
//     std::mt19937 eng(rd()); // Seed the generator

//     // Define the range for random numbers
//     std::uniform_int_distribution<> distr(0, size); // Range [1, 100]

//     for(int i =0; i<size; i++)
//     {
//         // Generate and a random number
//         int random_number = distr(eng);
//         swap(output[i], output[random_number]);
//     }

//     return output;

// }

// void CPURenderThread(vector<StaticBody>& bodies, int threadID, std::vector<int>::iterator pixelIndicesStart, std::vector<int>::iterator pixelIndicesEnd){
    
//     sf::Image renderBufferImage;
//     renderBufferImage.create(WINDOW_WIDTH * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE, sf::Color(0, 0, 0, 0));

//     int currentPixel = 0;
//     for (auto PixelIndexIt = pixelIndicesStart; PixelIndexIt != pixelIndicesEnd; ++PixelIndexIt){
//             if (stopFlag.load()){
//                 cout << endl << "render thread closed prematurely" << endl;
//                 return;
//             }
            
//             if (currentPixel % UPDATE_EVERY == 0)
//             {
//                 // Lock the mutex to safely update the image
                
//                 lock_guard<std::mutex> lock(imageMutex);
                    
//                     //update shared display image  
//                 sharedRenderImage.copy(renderBufferImage, 0, 0, sf::IntRect(0,0,0,0), true);

//                     // Flag that the image needs to be updated
//                 updateRequired.store(true);
                
//             }

//             int y = *PixelIndexIt / (int)(WINDOW_WIDTH * RENDER_SCALE);
//             int x = *PixelIndexIt % (int)(WINDOW_WIDTH * RENDER_SCALE);

//             StaticBody crashingBody = getCrashingBody(bodies, {(float)x/ RENDER_SCALE, (float)y/RENDER_SCALE}, 15000, 20);
//             renderBufferImage.setPixel(x, y, crashingBody.color);

//             currentPixel ++;
//     }
//     // Lock the mutex to safely update the image
//     {
//             // Lock the mutex to safely update the image        
//         lock_guard<std::mutex> lock(imageMutex);
                    
//             //update shared display image  
//         sharedRenderImage.copy(renderBufferImage, 0, 0, sf::IntRect(0,0,0,0), true);

//         finishedThreads++;
//         // Flag that the image needs to be updated
//         updateRequired.store(true);
//     }

// }
// vector<thread> startMultithreadCPURenderer(vector<StaticBody>& staticBodies){
//     int pixelsPerThread = (WINDOW_HEIGHT*WINDOW_WIDTH*RENDER_SCALE*RENDER_SCALE)/RENDER_THREADS;
//     vector<thread> renderThreads;
//     static vector<int> randomisedIndices = randomisedIntVector(WINDOW_HEIGHT*WINDOW_WIDTH*RENDER_SCALE*RENDER_SCALE);
//     for(int i = 0; i < RENDER_THREADS; i++){
//         vector<int>::iterator start = randomisedIndices.begin() + i*pixelsPerThread;
//         renderThreads.emplace_back(CPURenderThread, ref(staticBodies), i, start, start+pixelsPerThread);
//     }

//     return renderThreads;
// }

int main() {


    vector<StaticBody> static_bodies = {StaticBody({100, 400}, 100, sf::Color(0, 201, 167)),
                                        StaticBody({400, 100}, 100, sf::Color(189, 56, 178)),
                                        StaticBody({700, 400}, 100, sf::Color(212, 55, 37)),
                                        StaticBody({600, 650}, 100, sf::Color(212, 172, 91)),
                                        StaticBody({450, 450}, 10, sf::Color(105, 211, 91))};



    //-----------   Creating test trajectory sprite    -----------
    sf::Texture UserTrajectoryTexture;  
    UserTrajectoryTexture.create(WINDOW_WIDTH*RENDER_SCALE, WINDOW_HEIGHT*RENDER_SCALE); // Use the same size as the image
    UserTrajectoryTexture.setSmooth(true);
    // Create a sprite to display the image
    sf::Sprite userTrajectorySprite(UserTrajectoryTexture);
    userTrajectorySprite.setScale(1.0f/RENDER_SCALE, 1.0f/RENDER_SCALE); //TODO:bring this back with the Trajctory class later

    //-----------   Creating bodies sprite   -----------

    //create and keep in memora a variable for texture so it fucking works after 1000000hours debuging
    sf::Texture bodiesTexture = createBodiesTexture(static_bodies);
    bodiesTexture.setSmooth(true);
    // Create a Sprite to display the bodies
    sf::Sprite bodiesSprite(bodiesTexture);
    bodiesSprite.setScale(1.0f/RENDER_SCALE,1.0f/RENDER_SCALE);
    int traj_steps = 1;



    sf::Texture gravityBasinsTexture;
    gravityBasinsTexture.setSmooth(true);

    CpuBasinsRenderer renderer((int)(WINDOW_WIDTH*RENDER_SCALE), (int)(WINDOW_HEIGHT*RENDER_SCALE), static_bodies, RENDER_SCALE, 8);

    sf::Sprite gravityBasinsSprite;
    gravityBasinsSprite.setTexture(renderer.getTexture());
    gravityBasinsSprite.setScale(1.0f/RENDER_SCALE, 1.0f/RENDER_SCALE);

    renderer.startLiveRendering(1000);



    // Create a window
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Live Rendering");
    window.setFramerateLimit(60);


    // Simulate a render process
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed){// imagine forgeting to put parethesis here so every time you move mouse onto window the app closes and you have no idea why and you remove the stop flag and it starts working but doesntt make any sense untill one hour later you fucking notice....
                renderer.stopRendering();

                //close the app window
                window.close();
            }


            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    traj_steps = 1;
                }
            }
            if(event.type == sf::Event::MouseButtonPressed){

                sf::Vector2i mousePosWindow = sf::Mouse::getPosition(window);
                sf::Vector2u windowSize = window.getSize();
                bool isMouseInWindow =
                    mousePosWindow.x >= 0 &&
                    mousePosWindow.x <= static_cast<int>(windowSize.x) &&
                    mousePosWindow.y >= 0 &&
                    mousePosWindow.y <= static_cast<int>(windowSize.y);
                
                if(isMouseInWindow){
                    if(event.mouseButton.button == sf::Mouse::Left)
                    {
                        Trajectory userTrajectory = generateTrajectory(static_bodies, mousePosWindow, 80000);
                        UserTrajectoryTexture.update(createTrajectortTexture(userTrajectory, 1.0f, sf::Color::White));
                    }
                    else if(event.mouseButton.button == sf::Mouse::Right)// clear trajectories on RMB
                    {
                        // Create an image with the specified width and height
                        sf::Image clearImage;
                        clearImage.create(WINDOW_WIDTH*RENDER_SCALE, WINDOW_HEIGHT*RENDER_SCALE, sf::Color(0, 0, 0, 0)); // Fill the image with transparent color

                        // Update the texture with the new image
                        UserTrajectoryTexture.update(clearImage);
                    }
                }

            }
        }


        renderer.updateOutput();
        // Update the texture if the image has been updated
        // if (updateRequired.load()) {
            
            // std::lock_guard<std::mutex> lock(imageMutex);
            // sharedRenderTexture.update(sharedRenderImage); 
            // updateRequired.store(false);

            // //chcek if all threads finished, join them, and save output image
            // if (finishedThreads == RENDER_THREADS){
            //     for(auto& thread:renderThreads)
            //     {
            //         if (thread.joinable()) {
            //             thread.join();
                        
            //         }
            //     }
            //     if(SAVE_OUTPUT){
            //         sf::Image outputImage(sharedRenderImage);
            //         outputImage.copy(bodiesTexture.copyToImage(), 0, 0, sf::IntRect(0,0,0,0), true);
            //         outputImage.saveToFile(OUTPUT_IMAGE_FILENAME);

                    
            //     }
            //     chrono::duration<double> elapsed_seconds = chrono::high_resolution_clock::now() - renderStartTime;
            //     cout << "Rendering finnished. Time elapsed: " << elapsed_seconds.count() << " seconds" << endl;
            // }
       // }

        window.clear(sf::Color(0,0,0));
    


        
        // Draw the image sprite
        window.draw(gravityBasinsSprite);





        // Draw the circle sprite
        window.draw(bodiesSprite);
        // Display user-s trajectory
        window.draw(userTrajectorySprite);
        window.display();

    }
    
    renderer.stopRendering();

    return 0;
}