#include <iostream>
#include <vector>
#include <string>
#include <SFML/Graphics.hpp>
#include <array>
using namespace std;


#define M_PI           3.14159265358979323846  /* pi */
#define STATIC_BODY_DENSITY 0.1
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


#define GRAVITY_CONSTANT 9.81

float particle_mass = 10;

struct Trajectory
{
    std::vector<sf::Vector2f> points;
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
            sf::CircleShape circle(radius);
            circle.setPosition(pos);
            circle.setOrigin(sf::Vector2f(radius, radius));
            circle.setFillColor(color);

            renderTexture.draw(circle);
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

Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2f start_pos, int maxSize = 10000, bool detectCollisions = true){
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
        pos += vel;

        traj.points.push_back(pos);
    }

    return traj;
}

sf::Texture createBodiesTexture(vector<StaticBody> bodies){

    // Create a RenderTexture to draw bodies
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH, WINDOW_HEIGHT); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 

    for(auto& body: bodies){
        body.render(renderTexture);
    }

    //finalize the render
    renderTexture.display();

    // Create a Texture from the RenderTexture, and return it
    return renderTexture.getTexture();
}

sf::Texture createTrajectortTexture(Trajectory trajectory, int pointRadius = 1){
    // Create a RenderTexture to draw trajectory points
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH, WINDOW_HEIGHT); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 

    for(const auto& point: trajectory.points)
    {
        sf::CircleShape circle(pointRadius);
        circle.setPosition(point);
        circle.setOrigin(sf::Vector2f(pointRadius, pointRadius));
        circle.setFillColor(sf::Color::Red);

        renderTexture.draw(circle);
    }

    renderTexture.display();

    return renderTexture.getTexture();

}

int main() {

    vector<StaticBody> static_bodies = {StaticBody({400, 105}, 100, sf::Color::White),
                                        StaticBody({150, 300}, 100, sf::Color(150, 12, 21)),
                                        StaticBody({600, 415}, 100, sf::Color::White)};




    //-----------   Creating test trajectory sprite    -----------
    Trajectory testTrajectory = generateTrajectory(static_bodies, {100,100});
    sf::Texture trajectoryTexture = createTrajectortTexture(testTrajectory);  

    // Create a sprite to display the image
    sf::Sprite trajectorySprite(trajectoryTexture);


    //-----------   Creating bodies sprite   -----------

    //create a variable for texture so it fucking works after 1000000hours debuging
    sf::Texture bodiesTexture = createBodiesTexture(static_bodies);
    // Create a Sprite to display the bodies
    sf::Sprite bodiesSprite(bodiesTexture);

    // Create a window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Live Rendering");


    int traj_steps = 1;

    // Simulate a render process
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();


        }

        window.clear();


        Trajectory testTrajectory = generateTrajectory(static_bodies, {200,100}, traj_steps);
        sf::Texture trajectoryTexture = createTrajectortTexture(testTrajectory);  

        // Create a sprite to display the image
        sf::Sprite trajectorySprite(trajectoryTexture);

        
        // Draw the image sprite
        window.draw(trajectorySprite);
        
        // Draw the circle sprite
        window.draw(bodiesSprite);
        
        window.display();

        traj_steps ++;

    }

    return 0;
}