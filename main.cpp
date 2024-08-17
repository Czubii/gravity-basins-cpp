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
            circle.setFillColor(color);

            renderTexture.draw(circle);
        }
};

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

sf::Texture createTrajectortTexture(vector<StaticBody> bodies, array<float, 2> starting_pos){
    // Create a RenderTexture to draw trajectory points
    sf::RenderTexture renderTexture;
    renderTexture.create(WINDOW_WIDTH, WINDOW_HEIGHT); // Use the same size as the image
    renderTexture.clear(sf::Color::Transparent); 


    sf::CircleShape circle(3);
    circle.setPosition({100,100});
    circle.setFillColor(sf::Color::Red);

    renderTexture.draw(circle);
    renderTexture.display();

    return renderTexture.getTexture();

}

int main() {

    vector<StaticBody> static_bodies = {StaticBody({0, 15}, 100, sf::Color::White),
                                        StaticBody({150, 300}, 100, sf::Color(150, 12, 21)),
                                        StaticBody({200, 215}, 100, sf::Color::White)};




    //-----------   Creating test trajectory sprite    -----------

    sf::Texture trajectoryTexture = createTrajectortTexture(static_bodies, {100,100});  

    // Create a sprite to display the image
    sf::Sprite trajectorySprite(trajectoryTexture);


    //-----------   Creating bodies sprite   -----------

    //create a variable for texture so it fucking works after 1000000hours debuging
    sf::Texture bodiesTexture = createBodiesTexture(static_bodies);
    // Create a Sprite to display the bodies
    sf::Sprite bodiesSprite(bodiesTexture);

    // Create a window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Live Rendering");

    // Simulate a render process
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();


        }

        window.clear();
        
        // Draw the image sprite
        window.draw(trajectorySprite);
        
        // Draw the circle sprite
        window.draw(bodiesSprite);
        
        window.display();

    }

    return 0;
}