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

using namespace std;


#define M_PI           3.14159265358979323846  /* pi */
#define STATIC_BODY_DENSITY 0.1
#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500
#define RENDER_SCALE 2.0f

#define GRAVITY_CONSTANT 9.81

#define UPDATE_EVERY 30000

#define RENDER_THREADS 12

#define SAVE_TO_FIZE true//TODO
#define SAVE_FILENAME "test.png"


float particle_mass = 100;


atomic<bool> stopFlag(false);

sf::Image sharedRenderImage;
sf::Texture sharedRenderTexture;
sf::Sprite gravityBasinsSprite;
std::atomic<bool> updateRequired(false);
std::mutex imageMutex;

struct Trajectory
{
    vector<sf::Vector2f> points;
};


class StaticBody{
    public:

        sf::Vector2f pos;
        float mass;
        int radius;
        sf::Color color;

        StaticBody(sf::Vector2f start_pos, float _mass, sf::Color _color){
            pos = start_pos;
            mass = _mass;
            color = _color;
            radius = (int)sqrt(mass/(M_PI*STATIC_BODY_DENSITY));
        }

        void render(sf::RenderTexture& renderTexture){
            sf::CircleShape circle(radius*RENDER_SCALE);
            circle.setPosition(pos*RENDER_SCALE);
            circle.setOrigin(sf::Vector2f(radius*RENDER_SCALE, radius*RENDER_SCALE));
            circle.setFillColor(sf::Color::Black);

            renderTexture.draw(circle);

            sf::CircleShape circle2(radius/1.2*RENDER_SCALE);
            circle2.setPosition(pos*RENDER_SCALE);
            circle2.setOrigin(sf::Vector2f(radius/1.2*RENDER_SCALE, radius/1.2*RENDER_SCALE));
            circle2.setFillColor(color);

            renderTexture.draw(circle2);
        }
};

float getDistanceSquared(const sf::Vector2f& point1, const sf::Vector2f& point2) {
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    return dx * dx + dy * dy;
}

float getDistance(const sf::Vector2f& point1, const sf::Vector2f& point2) {
    return sqrt(getDistanceSquared(point1, point2));
}

sf::Vector2f gravityForce(StaticBody body, sf::Vector2f pos){
    float distanceSquared = getDistanceSquared(body.pos, pos);

    sf::Vector2f direction = (body.pos - pos)/sqrt(distanceSquared);
    float magnitude = GRAVITY_CONSTANT * body.mass / distanceSquared;

    return direction * magnitude;
}

sf::Vector2f netGravityForce(vector<StaticBody> bodies, sf::Vector2f pos){
    sf::Vector2f force = {0, 0};

    for(const auto& body:bodies){
        force += gravityForce(body, pos);
    }

    return force;
}

bool colidesWithAny(vector<StaticBody> bodies, sf::Vector2f pos){
    for(const auto& body: bodies){
        if (getDistanceSquared(body.pos, pos) <= body.radius * body.radius)
        return true;
    }
    return false;
}

Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2f start_pos, int maxSize = 15000, 
                              float stepSize = 1, bool detectCollisions = true){
    Trajectory traj;

    traj.points.push_back(start_pos);

    sf::Vector2f pos = start_pos;
    sf::Vector2f vel = {0,0};
    for (int step = 0; step < maxSize; step++){

        if (detectCollisions){
            if(colidesWithAny(bodies, pos))
            return traj;
        }

        vel += netGravityForce(bodies, pos)/particle_mass;
        pos += vel*stepSize;

        traj.points.push_back(pos);
    }

    return traj;
}

StaticBody getCrashingBody(vector<StaticBody> bodies, sf::Vector2f start_pos, int maxSize = 15000, float stepSize = 1){
    sf::Vector2f pos = start_pos;
    sf::Vector2f vel = {0,0};
    for (int step = 0; step < maxSize; step++){

        if(colidesWithAny(bodies, pos))

            for(const auto& body: bodies){
                if (getDistanceSquared(body.pos, pos) <= body.radius * body.radius)
                return body;
        }

        vel += netGravityForce(bodies, pos)/particle_mass;
        pos += vel*stepSize;
    }

    cout << "no crashes for point: (" << start_pos.x << ", " << start_pos.y << ") after " << maxSize << " steps" << endl;

    //return a new static body with black color for simplicty
    return StaticBody({0,0},0, sf::Color::Black);
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

sf::Texture createTrajectortTexture(Trajectory trajectory, float pointRadius = 1, sf::Color color = sf::Color::Red){
    // Create a RenderTexture to draw trajectory points
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH, WINDOW_HEIGHT); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 

    for(const auto& point: trajectory.points)
    {
        sf::CircleShape circle(pointRadius);
        circle.setPosition(point);
        circle.setOrigin(sf::Vector2f(pointRadius, pointRadius));
        circle.setFillColor(color);

        renderTexture.draw(circle);
    }

    renderTexture.display();

    return renderTexture.getTexture();

}

void liveRenderGravityBasin(vector<StaticBody>& bodies, int renderStartX = 0, int renderWidth = WINDOW_WIDTH){
    sf::Image renderBufferImage;
    renderBufferImage.create(renderWidth * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE);

    for (int x = renderStartX * RENDER_SCALE; x - (renderStartX*RENDER_SCALE) < renderWidth * RENDER_SCALE; x++){
        for (int y=0; y<WINDOW_HEIGHT * RENDER_SCALE; y++){
            if (stopFlag.load()){
                cout << endl << "render thread closed prematurely" << endl;
                return;
            }
            
            int currentPixel = (x-renderStartX)*y + y;

            
            if (currentPixel % UPDATE_EVERY == 0)
            {
                // Lock the mutex to safely update the image
                {
                    lock_guard<std::mutex> lock(imageMutex);
                    
                    //update shared display image  
                    sharedRenderImage.copy(renderBufferImage, renderStartX*RENDER_SCALE, 0);

                    // Flag that the image needs to be updated
                    updateRequired.store(true);
                }
            }



            StaticBody crashingBody = getCrashingBody(bodies, {(float)x/ RENDER_SCALE, (float)y/RENDER_SCALE}, 15000, 20);
            renderBufferImage.setPixel(x-(renderStartX*RENDER_SCALE), y, crashingBody.color);


    }}
    // Lock the mutex to safely update the image
    {
        lock_guard<std::mutex> lock(imageMutex);
                    

        //update shared display image 
        sharedRenderImage.copy(renderBufferImage, renderStartX*RENDER_SCALE, 0);

         // Flag that the image needs to be updated
        updateRequired.store(true);
    }

}


int main() {
    vector<StaticBody> static_bodies = {StaticBody({100, 400}, 100, sf::Color(0, 201, 167)),
                                        StaticBody({400, 100}, 100, sf::Color(189, 56, 178)),
                                        StaticBody({700, 400}, 100, sf::Color(212, 55, 37))};




    //-----------   Creating test trajectory sprite    -----------
    Trajectory testTrajectory = generateTrajectory(static_bodies, {100,100});
    sf::Texture trajectoryTexture = createTrajectortTexture(testTrajectory);  

    // Create a sprite to display the image
    sf::Sprite trajectorySprite(trajectoryTexture);


    //-----------   Creating bodies sprite   -----------

    //create and keep in memora a variable for texture so it fucking works after 1000000hours debuging
    sf::Texture bodiesTexture = createBodiesTexture(static_bodies);
    bodiesTexture.setSmooth(true);
    // Create a Sprite to display the bodies
    sf::Sprite bodiesSprite(bodiesTexture);
    bodiesSprite.setScale(1.0f/RENDER_SCALE,1.0f/RENDER_SCALE);

    // Create a window
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Live Rendering");


    int traj_steps = 1;

    sharedRenderImage.create(WINDOW_WIDTH * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE, sf::Color::White);
    sharedRenderTexture.loadFromImage(sharedRenderImage);
    sharedRenderTexture.setSmooth(true);

    gravityBasinsSprite.setTexture(sharedRenderTexture);
    gravityBasinsSprite.setScale(1.0f/RENDER_SCALE, 1.0f/RENDER_SCALE);

    int renderWidth = WINDOW_WIDTH/RENDER_THREADS;
    vector<thread> renderThreads;
    for(int i = 0; i < RENDER_THREADS; i++){
        cout << "thread: "<< i<<" start: " << i*renderWidth<<" end: "<< (i+1) * renderWidth<<endl;
        renderThreads.emplace_back(liveRenderGravityBasin, ref(static_bodies), i*renderWidth, renderWidth);
    }

    // Simulate a render process
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed){// imagine forgeting to put parethesis here so every time you move mouse onto window the app closes and you have no idea why and you remove the stop flag and it starts working but doesntt make any sense untill one hour later you fucking notice....
                // Set the flag to stop the thread
                stopFlag.store(true);

                //close the app window
                window.close();
            }


            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    traj_steps = 1;
                }
            }
        }

        // Update the texture if the image has been updated
        if (updateRequired.load()) {
            
            std::lock_guard<std::mutex> lock(imageMutex);
            sharedRenderTexture.update(sharedRenderImage); 
            updateRequired.store(false);
            
        }

        window.clear();
    


        
        // Draw the image sprite
        window.draw(gravityBasinsSprite);

    





/*

        Trajectory testTrajectory = generateTrajectory(static_bodies, {200,100}, traj_steps);
        sf::Texture trajectoryTexture = createTrajectortTexture(testTrajectory, 2);  

        // Create a sprite to display the image
        sf::Sprite trajectorySprite(trajectoryTexture);



        Trajectory testTrajectory2 = generateTrajectory(static_bodies, {200,100}, traj_steps, 25);
        sf::Texture trajectoryTexture2 = createTrajectortTexture(testTrajectory2, 1.5, sf::Color::Blue);  

        // Create a sprite to display the image
        sf::Sprite trajectorySprite2(trajectoryTexture2);



        window.draw(trajectorySprite);
        window.draw(trajectorySprite2);

*/






        // Draw the circle sprite
        window.draw(bodiesSprite);
        
        window.display();

        traj_steps ++;

    }
    
    for(auto& thread:renderThreads)
    {
        if (thread.joinable()) {
            thread.join();
            
        }
    }
    stopFlag.store(false);

    return 0;
}