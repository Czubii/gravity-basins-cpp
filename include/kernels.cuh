
#ifndef KERNELS_CUH
#define KERNELS_CUH



namespace kernel {

struct Color {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
};

struct vector2f {
  float x;
  float y;

  vector2f operator+(const vector2f &other) const {
    return {x + other.x, y + other.y};
  }
  vector2f operator-(const vector2f &other) const {
    return {x - other.x, y - other.y};
  }
  vector2f operator*(const float &other) const {
    return {x * other, y * other};
  }
  vector2f operator/(const float &other) const {
    return {x / other, y / other};
  }
  vector2f &operator+=(const vector2f &other) {
    x += other.x;
    y += other.y;
    return *this; // Return the modified object
  }
};

struct particleData {
  kernel::vector2f pos;
  kernel::vector2f vel;
  unsigned int terminated;
};

struct StaticBody {
  float x;
  float y;
  float mass;
  kernel::Color color;
  int radius;
};

unsigned char *testRender(int outputWidth, int outputHeight,
                          kernel::StaticBody *bodies,
                          unsigned char numStaticBodies, float renderScale);

}

#endif
