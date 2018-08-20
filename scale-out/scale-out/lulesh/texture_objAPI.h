#pragma once 

template <class ScalarType> class TextureObj;

template<class ScalarType> 
class TextureObj_Base
{
  public: 

  void initialize(ScalarType* ptr, int size)
  {

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = ptr;
    resDesc.res.linear.sizeInBytes= size*sizeof(ScalarType);

    cudaCheckError();

    // Channel Format Desc
    cudaChannelFormatDesc channelDesc;
    this->getChannelDesc(channelDesc);
    resDesc.res.linear.desc = channelDesc;
   
    cudaCheckError();

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
 
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    cudaCheckError();

  }

  virtual void getChannelDesc(cudaChannelFormatDesc& channelDesc) = 0;

  ~TextureObj_Base()
  {
    cudaDestroyTextureObject(texObj);
  }

  protected:

  cudaTextureObject_t texObj;

};

template <>
class TextureObj<double> : public TextureObj_Base<double> 
{
  public:

  void getChannelDesc(cudaChannelFormatDesc& channelDesc) 
  { 
    channelDesc = cudaCreateChannelDesc<int2>();
  }

  __device__ __forceinline__ double operator[]( int idx ) const 
  { 
    int2 texel = tex1Dfetch<int2>( texObj, idx ); 
    return __hiloint2double( texel.y, texel.x ); 
  } 
};

template <>
class TextureObj<float> : public TextureObj_Base<float>
{

  public:
  void getChannelDesc(cudaChannelFormatDesc& channelDesc) 
  { 
    channelDesc = cudaCreateChannelDesc<float>();
  };

  __device__ __forceinline__ float operator[]( int idx ) const 
  { 
    return tex1Dfetch<float>( texObj, idx ); 
  } 
};

template <>
class TextureObj<int> : public TextureObj_Base<int>
{

  public:
  void getChannelDesc(cudaChannelFormatDesc& channelDesc) 
  { 
    channelDesc = cudaCreateChannelDesc<int>();
  };

  __device__ __forceinline__ int operator[]( int idx ) const 
  { 
    return tex1Dfetch<int>( texObj, idx ); 
  } 
};



