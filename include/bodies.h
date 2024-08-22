#ifndef BODIES_H
#define BODIES_H

#include <SFML/Graphics.hpp>

using namespace std;

struct Trajectory//TODO: make this a class
{
    vector<sf::Vector2f> points;
};

class StaticBody{
    public:

        sf::Vector2f pos;
        float mass;
        int radius;
        sf::Color color;

        StaticBody(sf::Vector2f _pos, float _mass, sf::Color _color);

        /// @brief renders static body on given sf::RenderTexture renderTexture
        /// @param renderTexture sf::RenderTexture to render the body on
        void render(sf::RenderTexture& renderTexture);

        
};

float fastInvSqrt(float number);

float fastSqrt(float number);


float getDistanceSquared(const sf::Vector2f& point1, const sf::Vector2f& point2);

float getDistance(const sf::Vector2f& point1, const sf::Vector2f& point2);

sf::Vector2f gravityForce(StaticBody body, sf::Vector2f pos);

sf::Vector2f netGravityForce(vector<StaticBody> bodies, sf::Vector2f pos);

bool colidesWithAny(vector<StaticBody> bodies, sf::Vector2f pos);



Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2f startPos, int maxSize = 15000, 
                              float stepSize = 1, bool detectCollisions = true);

Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2i startPos, int maxSize = 15000, 
                              float stepSize = 1, bool detectCollisions = true);

StaticBody getCrashingBody(vector<StaticBody> bodies, sf::Vector2f start_pos, int maxSize, float stepSize);

#endif