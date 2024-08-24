#include "cuda_renderer.h"
#include "SFML/Graphics/Color.hpp"
#include "SFML/Graphics/Image.hpp"
#include "SFML/Graphics/Texture.hpp"
#include "bodies.h"
#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include "kernels.cuh"

CUDABasinsRenderer::CUDABasinsRenderer(int _outputWidth, int _outputHeight, std::vector<StaticBody>& _staticBodies, float _renderScale)
  : rendering(false),
    outputWidth(_outputWidth),
    outputHeight(_outputHeight),
    staticBodies(_staticBodies),
    renderScale(_renderScale)
{

    renderImage.create(outputWidth*renderScale, outputHeight*renderScale);
    outputTexture.loadFromImage(renderImage);
    outputTexture.setSmooth(true);
}
void CUDABasinsRenderer::renderFrame(){
  unsigned char numBodies = staticBodies.size();
  kernel::StaticBody* kernelBodies = new kernel::StaticBody[numBodies];

  for(int i = 0; i<numBodies; i++){

    kernel::Color color = {staticBodies[i].color.r, staticBodies[i].color.g, staticBodies[i].color.b, staticBodies[i].color.a};
    kernelBodies[i] = kernel::StaticBody{staticBodies[i].pos.x, staticBodies[i].pos.y, staticBodies[i].mass, color, staticBodies[i].radius};
  }

  unsigned char * pixelArray = kernel::testRender(outputWidth, outputHeight, kernelBodies, numBodies, renderScale);


  outputTexture.update(pixelArray);

}
void CUDABasinsRenderer::setScale(float scale){
  renderScale = scale;
}

sf::Texture & CUDABasinsRenderer::getTexture(){
  return outputTexture;
}
