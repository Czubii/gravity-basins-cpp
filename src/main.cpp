#include <vector>
#include <SFML/Graphics.hpp>
#include "bodies.h"
#include "kernels.cuh"
#include "parameters.h"
#include "cpu_renderer.h"

using namespace std;

int main() {

    kernel::test();
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

    CpuBasinsRenderer renderer((int)(WINDOW_WIDTH*RENDER_SCALE), (int)(WINDOW_HEIGHT*RENDER_SCALE), static_bodies, RENDER_SCALE, 11);

    sf::Sprite gravityBasinsSprite;
    gravityBasinsSprite.setTexture(renderer.getTexture());
    gravityBasinsSprite.setScale(1.0f/RENDER_SCALE, 1.0f/RENDER_SCALE);

    renderer.startLiveRendering(UPDATE_EVERY);



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

        bool renderFinished;
        renderFinished = renderer.updateOutput();

        if(renderFinished && SAVE_OUTPUT){

            sf::Image outputImg(renderer.getTexture().copyToImage());

            outputImg.saveToFile(OUTPUT_IMAGE_FILENAME);
        }

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