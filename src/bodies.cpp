#include <bodies.h>

#include <SFML/Graphics.hpp>
#include "parameters.h"
#include <iostream>
#include <cstdint>  // For int32_t
using namespace std;


//TODO: make this file not be a total mess



        StaticBody::StaticBody(sf::Vector2f _pos, float _mass, sf::Color _color)
        : pos(_pos),
          mass(_mass),
          color(_color)
        {

            radius = (int)sqrt(mass/(M_PI*STATIC_BODY_DENSITY));

        }

        void StaticBody::render(sf::RenderTexture& renderTexture){
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


float fastInvSqrt(float number) {
    if (number <= 0) return 0;  // Handle non-positive input
    float x = number;
    float x_half = 0.5f * x;
    int32_t i = *reinterpret_cast<int32_t*>(&x);
    i = 0x5f3759df - (i >> 1);  // Initial approximation
    x = *reinterpret_cast<float*>(&i);
    x = x * (1.5f - (x_half * x * x));  // Refine approximation with Newton's method
    return x;
}

float fastSqrt(float number) {
    return 1.0f / fastInvSqrt(number);
}


float getDistanceSquared(const sf::Vector2f& point1, const sf::Vector2f& point2) {
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    return dx * dx + dy * dy;
}

float getDistance(const sf::Vector2f& point1, const sf::Vector2f& point2) {
    return fastSqrt(getDistanceSquared(point1, point2));
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



Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2f startPos, int maxSize, 
                              float stepSize, bool detectCollisions){
    Trajectory traj;

    traj.points.push_back(startPos);

    sf::Vector2f pos = startPos;
    sf::Vector2f vel = {0,0};
    for (int step = 0; step < maxSize; step++){

        if (detectCollisions){
            if(colidesWithAny(bodies, pos))
            return traj;
        }

        vel += netGravityForce(bodies, pos)/PARTICLE_MASS;
        pos += vel*stepSize;

        traj.points.push_back(pos);
    }

    cout << "hmm";

    return traj;
}

Trajectory generateTrajectory(vector<StaticBody> bodies, sf::Vector2i startPos, int maxSize, 
                              float stepSize, bool detectCollisions){
                                sf::Vector2f startPosF = {(float)startPos.x, (float)startPos.y};
                                return generateTrajectory(bodies, startPosF, maxSize, stepSize, detectCollisions);
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

        vel += netGravityForce(bodies, pos)/PARTICLE_MASS;
        pos += vel*stepSize;
    }

    if(!MUTE_STEP_LIMIT_INFO)
    cout << "no crashes for point: (" << start_pos.x << ", " << start_pos.y << ") after " << maxSize << " steps" << endl;

    //return a new static body with black color for simplicty
    return StaticBody({0,0},0, sf::Color::Black);
}
