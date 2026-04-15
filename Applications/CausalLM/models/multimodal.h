#include <utility>

#include "tensor_dim.h"

// TODO nntrainer should have ways to deal with this
typedef std::pair<void *, size_t> multimodal_pointer;

class ImageModel {
  virtual multimodal_pointer runImage(const multimodal_pointer &image,
                                      int image_height, int image_width) = 0;
};

class VoiceModel {
  virtual multimodal_pointer runVoice(multimodal_pointer voice) = 0;
};
