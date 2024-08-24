#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H
#include "SFML/Graphics/Texture.hpp"
#include "bodies.h"

class CUDABasinsRenderer{
    private:
    std::vector<StaticBody>& staticBodies; 
    int outputWidth;
    int outputHeight;
    float renderScale;


    sf::Image renderImage;
    sf::Texture outputTexture; 

    
    public:
    bool rendering;

    CUDABasinsRenderer(int _outputWidth, int _outputHeight, std::vector<StaticBody>& _staticBodies, float _renderScale);
    void renderFrame();
    void setScale(float scale);

    sf::Texture& getTexture();

};

#endif