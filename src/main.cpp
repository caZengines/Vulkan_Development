#include "vulkan/vulkan.hpp"
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <algorithm>
#include <fstream>
#include <map>
#include <unordered_map>
#include <chrono>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "../models/container.obj";
const std::string TEXTURE_PATH = "../textures/container.png";

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription description;
        description.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::eVertex);

        return description;
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription() {
        vk::VertexInputAttributeDescription posAttribute;
        posAttribute.setLocation(0).setFormat(vk::Format::eR32G32B32Sfloat).setOffset(offsetof(Vertex, pos));
        vk::VertexInputAttributeDescription uvAttribute;
        uvAttribute.setLocation(1).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, texCoord));

        return {posAttribute, uvAttribute};
    }
    bool operator==(const Vertex& other) const {
        return pos == other.pos && texCoord == other.texCoord;
    }
};
namespace std{
    template<> struct hash<Vertex>{
        size_t operator()(Vertex const& vertex) const {
            return (hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec2>()(vertex.texCoord) << 1));
        }
    };
}

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }
    private:
        GLFWwindow*                              window = nullptr;
        vk::raii::Context                        context;
        vk::raii::Instance                       instance       = nullptr;
        vk::raii::DebugUtilsMessengerEXT         debugMessenger = nullptr;
        vk::raii::SurfaceKHR                     surface        = nullptr;
        vk::raii::PhysicalDevice                 physicalDevice = nullptr;
        vk::raii::Device                         device         = nullptr;
        vk::raii::Queue                          graphicsQueue  = nullptr;
        vk::raii::Queue                          transferQueue  = nullptr;
        vk::raii::SwapchainKHR                   swapChain      = nullptr;
        std::vector<vk::Image>                   swapchainImages;
        vk::SurfaceFormatKHR                     swapchainSurfaceFormat;
        vk::Extent2D                             swapchainExtent;
        std::vector<vk::raii::ImageView>         swapchainImageViews;

        uint32_t                                 graphicsQueueIndex      = ~0;
        uint32_t                                 transferQueueIndex      = ~0;

        vk::raii::DescriptorSetLayout            descriptorSetLayout  = nullptr;
        vk::raii::PipelineLayout                 pipelineLayout       = nullptr;
        vk::raii::Pipeline                       graphicsPipeline     = nullptr;
        vk::raii::CommandPool                    graphicsCommandPool      = nullptr;
        vk::raii::CommandPool                    transientCommandPool     = nullptr;
        std::vector<vk::raii::CommandBuffer>     graphicsCommandBuffers;

        std::vector<Vertex>                      vertices;
        std::vector<uint32_t>                    indices;
        vk::raii::Buffer                         vertexBuffer         = nullptr;
        vk::raii::DeviceMemory                   vertexBufferMemory   = nullptr;
        vk::raii::Buffer                         indexBuffer          = nullptr;
        vk::raii::DeviceMemory                   indexBufferMemory    = nullptr;
        std::unordered_map<Vertex, uint32_t>     uniqueVertices{};
        
        uint32_t                                 mipLevels;
        vk::raii::Image                          textureImage         = nullptr;
        vk::raii::DeviceMemory                   textureImageMemory   = nullptr;
        vk::raii::ImageView                      textureImageView     = nullptr;
        vk::raii::Sampler                        textureSampler       = nullptr;
        vk::raii::Image                          depthImage           = nullptr;
        vk::raii::DeviceMemory                   depthImageMemory     = nullptr;
        vk::raii::ImageView                      depthImageView       = nullptr;

        std::vector<vk::raii::Buffer>            uniformBuffers;
        std::vector<vk::raii::DeviceMemory>      uniformBuffersMemory;
        std::vector<void *>                      uniformBuffersMapped;
        uint32_t                                 frameIndex       = 0;

        vk::raii::DescriptorPool                 descriptorPool      = nullptr;
        std::vector<vk::raii::DescriptorSet>     descriptorSets;

        std::vector<vk::raii::Semaphore>         presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore>         presentWaitSemaphores;
        vk::raii::Semaphore                      renderFinishedTimelineSemaphore = nullptr;
        std::vector<vk::raii::Fence>             inFlightFences;
        uint64_t                                 frameCount       = 0;

        bool                                     framebufferResized = false;
 
        std::vector<const char *> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};
        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "C' Vulkan", nullptr, nullptr);
            glfwSetWindowUserPointer(window, this);
            glfwSetFramebufferSizeCallback(window, glfwFramebufferResizeCallback);
        }

        static void glfwFramebufferResizeCallback(GLFWwindow* window, int width, int height){
            auto app                = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
            app->framebufferResized = true;
        }

        void initVulkan() {
            createInstance();
            setupDebugMessenger();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createCommandPools();
            createDepthResources();
            createTextureImage();
            createTextureImageView();
            createTextureSampler();
            loadModel();
            createVertexBuffer();
            createIndexBuffer();
            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
        }

        void createImageViews(){
            assert(swapchainImageViews.empty());
            swapchainImageViews.reserve(swapchainImages.size());
            for(const auto& image : swapchainImages){         
                swapchainImageViews.emplace_back(createImageView(image, swapchainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
            }
        }

        void createDescriptorSetLayout(){
            std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
                vk::DescriptorSetLayoutBinding().setBinding(0).setDescriptorType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eVertex),
                vk::DescriptorSetLayoutBinding().setBinding(1).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eFragment)
            };
            vk::DescriptorSetLayoutCreateInfo layoutInfo;
            layoutInfo.setBindings(bindings);

            descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        }

        void createGraphicsPipeline(){
            //shader modules
            vk::raii::ShaderModule shaderModule = createShaderModule(ReadFile("../shaders/slang.spv"));

            vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
            vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex)
                               .setModule(shaderModule)
                               .setPName("vertMain");

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
            fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment)
                               .setModule(shaderModule)
                               .setPName("fragMain");

            vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            //dynamic state
            std::vector<vk::DynamicState> dynamicStates = {
                vk::DynamicState::eViewport, vk::DynamicState::eScissor
            };
            vk::PipelineDynamicStateCreateInfo dynamicState;
            dynamicState.setDynamicStates(dynamicStates);
            vk::PipelineViewportStateCreateInfo viewportState;
            viewportState.setViewportCount(1).setScissorCount(1);

            //fixed function
            auto                                   bindingDescription = Vertex::getBindingDescription();
            auto                                   attributeDescription = Vertex::getAttributeDescription();
            vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
            vertexInputInfo.setVertexBindingDescriptions(bindingDescription)
                           .setVertexAttributeDescriptions(attributeDescription);
            vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
            inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
            //rasterizer
            vk::PipelineRasterizationStateCreateInfo rasterizer;
            rasterizer.setDepthClampEnable(vk::False)
                      .setRasterizerDiscardEnable(vk::False)
                      .setPolygonMode(vk::PolygonMode::eFill)
                      .setCullMode(vk::CullModeFlagBits::eBack)
                      .setFrontFace(vk::FrontFace::eCounterClockwise)
                      .setDepthBiasEnable(vk::False)
                      .setLineWidth(1.0f);
            //multisampling
            vk::PipelineMultisampleStateCreateInfo multisampling;
            multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1)
                         .setSampleShadingEnable(vk::False);
            //depth and stencil testing
            vk::PipelineDepthStencilStateCreateInfo depthStencil;
            depthStencil.setDepthTestEnable(vk::True)
                        .setDepthWriteEnable(vk::True)
                        .setDepthCompareOp(vk::CompareOp::eLess)
                        .setDepthBoundsTestEnable(vk::False)
                        .setStencilTestEnable(vk::False);
            //color blending
            vk::PipelineColorBlendAttachmentState colorBlendAttachment;
            colorBlendAttachment.setBlendEnable(vk::True)
                                .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
                                .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
                                .setColorBlendOp(vk::BlendOp::eAdd)
                                .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
                                .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
                                .setAlphaBlendOp(vk::BlendOp::eAdd)
                                .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
            vk::PipelineColorBlendStateCreateInfo colorBlending;
            colorBlending.setLogicOpEnable(vk::False)
                         .setLogicOp(vk::LogicOp::eCopy)
                         .setAttachmentCount(1)
                         .setPAttachments(&colorBlendAttachment);
            //pipeline layout
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
            pipelineLayoutInfo.setSetLayouts(*descriptorSetLayout)
                              .setPushConstantRangeCount(0);
            pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

            //dynamic rendering
            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain;
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().setStages(shaderStages)
                                                                         .setPVertexInputState(&vertexInputInfo)
                                                                         .setPInputAssemblyState(&inputAssembly)
                                                                         .setPDepthStencilState(&depthStencil)
                                                                         .setPViewportState(&viewportState)
                                                                         .setPRasterizationState(&rasterizer)
                                                                         .setPMultisampleState(&multisampling)
                                                                         .setPColorBlendState(&colorBlending)
                                                                         .setPDynamicState(&dynamicState)
                                                                         .setLayout(pipelineLayout)
                                                                         .setRenderPass(nullptr);
            
            pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>().setColorAttachmentCount(1)
                                                                          .setPColorAttachmentFormats(&swapchainSurfaceFormat.format)
                                                                          .setDepthAttachmentFormat(findDepthFormat());
                                                                          

            //createGraphicsPipeline
            graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        }

        void createCommandPools(){
            vk::CommandPoolCreateInfo graphicsPoolInfo;
            graphicsPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                            .setQueueFamilyIndex(graphicsQueueIndex);
            graphicsCommandPool = vk::raii::CommandPool(device, graphicsPoolInfo);
            vk::CommandPoolCreateInfo transientPoolInfo;
            transientPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient)
                            .setQueueFamilyIndex(transferQueueIndex);
            transientCommandPool = vk::raii::CommandPool(device, transientPoolInfo);
        }

        void createDescriptorPool(){
            std::array<vk::DescriptorPoolSize, 2> poolSize{
                vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(MAX_FRAMES_IN_FLIGHT),
                vk::DescriptorPoolSize().setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(MAX_FRAMES_IN_FLIGHT)
            };
            vk::DescriptorPoolCreateInfo poolInfo;
            poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet).setMaxSets(MAX_FRAMES_IN_FLIGHT).setPoolSizes(poolSize);

            descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        }

        void createDescriptorSets(){
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
            vk::DescriptorSetAllocateInfo allocInfo;
            allocInfo.setDescriptorPool(descriptorPool).setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
                     .setSetLayouts(layouts);

            descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
            for(size_t i = 0 ; i < MAX_FRAMES_IN_FLIGHT ; ++i){
                vk::DescriptorBufferInfo bufferInfo;
                bufferInfo.setBuffer(uniformBuffers[i]).setOffset(0).setRange(sizeof(UniformBufferObject));
                vk::DescriptorImageInfo imageInfo;
                imageInfo.setSampler(textureSampler).setImageView(textureImageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

                std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
                    vk::WriteDescriptorSet().setDstSet(descriptorSets[i]).setDstBinding(0).setDstArrayElement(0).setDescriptorType(vk::DescriptorType::eUniformBuffer).setBufferInfo(bufferInfo),
                    vk::WriteDescriptorSet().setDstSet(descriptorSets[i]).setDstBinding(1).setDstArrayElement(0).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setImageInfo(imageInfo)
                };

                device.updateDescriptorSets(descriptorWrites, {});
            }
        }

        std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::SharingMode mode = vk::SharingMode::eExclusive, const std::vector<uint32_t>& queueFamilyIndicies = {}){
            vk::BufferCreateInfo bufferInfo;
            bufferInfo.setSize(size).setUsage(usage).setSharingMode(mode);
            if(mode == vk::SharingMode::eConcurrent){
                bufferInfo.setQueueFamilyIndices(queueFamilyIndicies);
            }
                vk::raii::Buffer       buffer = vk::raii::Buffer(device, bufferInfo);
            
            vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
            vk::MemoryAllocateInfo memAllocateInfo;
            memAllocateInfo.setAllocationSize(memRequirements.size)
                           .setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));
                vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, memAllocateInfo);
            buffer.bindMemory(*bufferMemory, 0);
            
            return {std::move(buffer), std::move(bufferMemory)};
        }

        std::pair<vk::raii::Image, vk::raii::DeviceMemory>  createImage(uint32_t width, uint32_t height, uint32_t _mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::SharingMode mode = vk::SharingMode::eExclusive, const std::vector<uint32_t>& queueFamilies = {}) const{
            vk::ImageCreateInfo imageInfo;
            imageInfo.setExtent({width, height, 1}).setFormat(format).setTiling(tiling).setUsage(usage).setSharingMode(mode)
                     .setImageType(vk::ImageType::e2D).setMipLevels(_mipLevels).setArrayLayers(1);
            if(mode == vk::SharingMode::eConcurrent){
                imageInfo.setQueueFamilyIndices(queueFamilies);
            }
                vk::raii::Image      image = vk::raii::Image(device, imageInfo);

            vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
            vk::MemoryAllocateInfo memAllocateInfo;
            memAllocateInfo.setAllocationSize(memRequirements.size)
                           .setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));
                vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, memAllocateInfo);
            image.bindMemory(*imageMemory, 0);

            return {std::move(image), std::move(imageMemory)};
        }

        vk::raii::CommandBuffer beginSingleTimeCommands(vk::raii::CommandPool &commandPool){
            vk::CommandBufferAllocateInfo allocInfo;
            allocInfo.setCommandPool(commandPool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
            vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
            //record command
            vk::CommandBufferBeginInfo beginInfo; beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer.begin(beginInfo);

            return std::move(commandBuffer);
        }

        void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer, vk::raii::Queue &queue){
            commandBuffer.end();
            //wait for submit
            vk::FenceCreateInfo fenceInfo;
            vk::raii::Fence commandFence(device, fenceInfo);

            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(*commandBuffer);
            queue.submit(submitInfo, commandFence);
            (void)device.waitForFences({commandFence}, VK_TRUE, UINT64_MAX);

        }

        void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & desBuffer, vk::DeviceSize size){
            vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands(transientCommandPool);
            commandCopyBuffer.copyBuffer(*srcBuffer, *desBuffer, vk::BufferCopy(0, 0, size));
            endSingleTimeCommands(std::move(commandCopyBuffer), transferQueue);
        }

        void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height) const{
            vk::BufferImageCopy region;
            vk::ImageSubresourceLayers imageSubresource;imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor).setBaseArrayLayer(0).setLayerCount(1).setMipLevel(0);
            region.setBufferOffset(0).setBufferImageHeight(0).setBufferRowLength(0)
                  .setImageOffset({0,0,0})
                  .setImageExtent({width, height, 1})
                  .setImageSubresource(imageSubresource);

            commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
        }

        void createDepthResources(){
            vk::Format depthFormat = findDepthFormat();

            std::tie(depthImage, depthImageMemory) = createImage(swapchainExtent.width, swapchainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal);
            depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        }

        void createTextureImage(){
            int            texWidth, texHeight, texChannel;
            stbi_uc       *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannel, STBI_rgb_alpha);
            vk::DeviceSize imageSize = texWidth * texHeight * 4;
            mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) +1;
            if(!pixels){
                throw std::runtime_error("failed to load texture image! ");
            }

            auto [stagingBuffer, stagingBufferMemory] = 
                    createBuffer(
                        imageSize,
                        vk::BufferUsageFlagBits::eTransferSrc,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                        vk::SharingMode::eExclusive
            );
            void *data = stagingBufferMemory.mapMemory(0, imageSize);
            memcpy(data, pixels, imageSize);
            stagingBufferMemory.unmapMemory();
            stbi_image_free(pixels);

            std::tie(textureImage, textureImageMemory) = 
                        createImage(
                            texWidth, texHeight, mipLevels,
                            vk::Format::eR8G8B8A8Srgb,
                            vk::ImageTiling::eOptimal,
                            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                            vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(graphicsCommandPool);
            transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
            copyBufferToImage(commandBuffer, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            endSingleTimeCommands(std::move(commandBuffer), graphicsQueue);
            //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps
            generateMipMaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
        }

        void generateMipMaps(vk::raii::Image &image, vk::Format imageFormat, uint32_t texWidth_, uint32_t texHeight_, uint32_t mipLevels_){
            vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
            if(!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)){
                throw std::runtime_error("texture image format does not support linear blitting!");
            }

            vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(graphicsCommandPool);

            vk::ImageMemoryBarrier barrier;
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead)
                   .setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                   .setDstQueueFamilyIndex(vk::QueueFamilyIgnored).setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                   .setImage(image);
            barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor).setBaseArrayLayer(0).setLayerCount(1).setLevelCount(1);

            uint32_t mipWidth  = texWidth_;
            uint32_t mipHeight = texHeight_;
            for(uint32_t i = 1 ; i < mipLevels_ ;++i){
                barrier.subresourceRange.setBaseMipLevel(i - 1);
                barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
                barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);

                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

                vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
                offsets[0] = vk::Offset3D(0, 0, 0);
                offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
                dstOffsets[0] = vk::Offset3D(0, 0, 0);
                dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth/2 : 1, mipHeight > 1 ? mipHeight/2 : 1, 1);
                vk::ImageBlit blit; blit.setSrcSubresource({vk::ImageAspectFlagBits::eColor, i-1, 0, 1}).setSrcOffsets(offsets)
                                        .setDstSubresource({vk::ImageAspectFlagBits::eColor, i, 0, 1}).setDstOffsets(dstOffsets);
                commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, {blit}, vk::Filter::eLinear);

                //barrier.subresourceRange.setBaseMipLevel(i - 1);
                barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal).setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
                barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
                
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

                if(mipWidth  > 1) mipWidth  /= 2;
                if(mipHeight > 1) mipHeight /= 2;
            }
            barrier.subresourceRange.setBaseMipLevel(mipLevels_ - 1);
            barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
            
            endSingleTimeCommands(std::move(commandBuffer), graphicsQueue);
        }

        void createTextureImageView(){
            textureImageView = createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
        }

        vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels_) const{
            vk::ImageViewCreateInfo viewInfo;
            viewInfo.setImage(image)
                    .setFormat(format)
                    .setViewType(vk::ImageViewType::e2D)
                    .setSubresourceRange({aspectFlags, 0, mipLevels_, 0, 1});
            return vk::raii::ImageView(device, viewInfo);
        }

        void createTextureSampler(){
            vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
            vk::SamplerCreateInfo samplerInfo;
            samplerInfo.setMagFilter(vk::Filter::eLinear).setMinFilter(vk::Filter::eLinear)
                       .setAddressModeU(vk::SamplerAddressMode::eRepeat).setAddressModeV(vk::SamplerAddressMode::eRepeat).setAddressModeW(vk::SamplerAddressMode::eRepeat)
                       .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                       .setMipLodBias(0.0f).setMaxLod(vk::LodClampNone).setMinLod(0.0f)
                       .setAnisotropyEnable(vk::True)
                       .setMaxAnisotropy(properties.limits.maxSamplerAnisotropy)
                       .setCompareEnable(false).setCompareOp(vk::CompareOp::eAlways)
                       .setUnnormalizedCoordinates(vk::False)
                       .setBorderColor(vk::BorderColor::eIntOpaqueBlack);
            textureSampler = vk::raii::Sampler(device, samplerInfo);
        }

        void loadModel(){
            tinyobj::attrib_t                attrib;
            std::vector<tinyobj::shape_t>    shapes;
            std::vector<tinyobj::material_t> materials;
            std::string                      err;
            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str()))
            {
                throw std::runtime_error(err);
            }

            for(const auto& shape : shapes){
                for(const auto& index : shape.mesh.indices){
                    Vertex vertex{};
                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                    };
                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                    auto [it, inserted] = uniqueVertices.insert({vertex, static_cast<uint32_t>(vertices.size())});
                    if(inserted){
                        vertices.emplace_back(vertex);
                    }
                    indices.emplace_back(it->second);
                }
            }
        }

        void createVertexBuffer(){
            std::cout <<"Number of vertices: " <<vertices.size() <<"\n";
            vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
            auto [stagingBuffer, stagingBufferMemory] = 
                        createBuffer(bufferSize, 
                                    vk::BufferUsageFlagBits::eTransferSrc, 
                                    vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible
            );
            void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
            memcpy(dataStaging, vertices.data(), bufferSize);
            stagingBufferMemory.unmapMemory();

            std::tie(vertexBuffer, vertexBufferMemory) = 
                        createBuffer(bufferSize,
                                     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        }

        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const{
            vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
            for(int i = 0 ; i < memoryProperties.memoryTypeCount ;++i){
                if(( typeFilter & (1 << i) ) && ( (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) ){
                    return i;
                }
            }
            throw std::runtime_error("failed to find suitable memory type!");
        }

        vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features){
            for(const auto& format : candidates){
                vk::FormatProperties props = physicalDevice.getFormatProperties(format);
                if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features){
                    return format;
                }
                if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features){
                    return format;
                }
            }
            throw std::runtime_error("failed to find supported format! ");
        }
        vk::Format findDepthFormat(){
            return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
            );
        }
        bool hasStencilComponent(vk::Format format){
            return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
        }

        void createIndexBuffer(){
            vk::DeviceSize BufferSize = sizeof(indices[0]) * indices.size();
            auto [stagingBuffer, stagingBufferMemory] = 
                        createBuffer(
                                     BufferSize,
                                     vk::BufferUsageFlagBits::eTransferSrc,
                                     vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible
            );
            void* dataStaging = stagingBufferMemory.mapMemory(0, BufferSize);
            memcpy(dataStaging, indices.data(), (size_t)BufferSize);
            stagingBufferMemory.unmapMemory();

            std::tie(indexBuffer, indexBufferMemory) = 
                        createBuffer(
                            BufferSize,
                            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                            vk::MemoryPropertyFlagBits::eDeviceLocal,
                            vk::SharingMode::eConcurrent,
                            {graphicsQueueIndex, transferQueueIndex}
            );
            copyBuffer(stagingBuffer, indexBuffer, BufferSize);
        }

        void createUniformBuffers(){
            for(size_t i = 0 ; i < MAX_FRAMES_IN_FLIGHT; ++i){
                vk::DeviceSize BufferSize = sizeof(UniformBufferObject);
                auto [buffer, bufferMem] = 
                        createBuffer(
                            BufferSize,
                            vk::BufferUsageFlagBits::eUniformBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                );
                uniformBuffers.emplace_back(std::move(buffer));
                uniformBuffersMemory.emplace_back(std::move(bufferMem));
                uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, BufferSize));
            }
        }

        void createCommandBuffers(){
            vk::CommandBufferAllocateInfo allocInfo;
             allocInfo.setCommandPool(*graphicsCommandPool)
                      .setLevel(vk::CommandBufferLevel::ePrimary)
                      .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
            
                graphicsCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        }

        void recordCommandBuffer(uint32_t ImageIndex){
            graphicsCommandBuffers[frameIndex].begin({});

            // Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
            transition_image_layout(
                swapchainImages[ImageIndex],
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},                                                        //srcAccessMask(no need to wait for previous operation)
                vk::AccessFlagBits2::eColorAttachmentWrite,                //dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,         //srcStage
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,         //dstStage
                vk::ImageAspectFlagBits::eColor
            );
            transition_image_layout(
                *depthImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::ImageAspectFlagBits::eDepth
            );
            vk::ClearValue              clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
            vk::ClearValue              clearDepth = vk::ClearDepthStencilValue(1.0f, 0); 
            vk::RenderingAttachmentInfo attachmentInfo;
            attachmentInfo.setImageView(swapchainImageViews[ImageIndex])
                          .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                          .setLoadOp(vk::AttachmentLoadOp::eClear)
                          .setStoreOp(vk::AttachmentStoreOp::eStore)
                          .setClearValue(clearColor);
            vk::RenderingAttachmentInfo depthInfo;
            depthInfo.setImageView(depthImageView)
                     .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                     .setLoadOp(vk::AttachmentLoadOp::eClear)
                     .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                     .setClearValue(clearDepth);
            //renderingInfo
            vk::RenderingInfo renderingInfo;
            vk::Rect2D renderArea;
            renderArea.setOffset({0,0})
                      .setExtent(swapchainExtent);
            renderingInfo.setRenderArea(renderArea)
                         .setLayerCount(1)
                         .setColorAttachments(attachmentInfo)
                         .setPDepthAttachment(&depthInfo);

            //start rendering
            graphicsCommandBuffers[frameIndex].beginRendering(renderingInfo);
                //binding the graphics pipeline
                graphicsCommandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
                //binding the vertexbuffer
                graphicsCommandBuffers[frameIndex].bindVertexBuffers(0, *vertexBuffer, {0});
                graphicsCommandBuffers[frameIndex].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
                //command buffer dynamic state
                graphicsCommandBuffers[frameIndex].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
                graphicsCommandBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0,0), swapchainExtent));

                graphicsCommandBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);
                graphicsCommandBuffers[frameIndex].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            //end rendering
            graphicsCommandBuffers[frameIndex].endRendering();
            // After rendering, transition the swapchain image to vk::ImageLayout::ePresentSrcKHR
            transition_image_layout(
                swapchainImages[ImageIndex],
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,        // srcAccessMask
                {},                                                // dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                vk::PipelineStageFlagBits2::eBottomOfPipe,          // dstStage
                vk::ImageAspectFlagBits::eColor
            );
            
            graphicsCommandBuffers[frameIndex].end();
        }
        void transition_image_layout(vk::Image                image,
                                     vk::ImageLayout         old_layout,
                                     vk::ImageLayout         new_layout,
                                     vk::AccessFlags2        src_access_mask,
                                     vk::AccessFlags2        dst_access_mask,
                                     vk::PipelineStageFlags2 src_stage_mask,
                                     vk::PipelineStageFlags2 dst_stage_mask,
                                     vk::ImageAspectFlags    image_aspect_flags) 
                                {
                                    vk::ImageMemoryBarrier2 barrier;
                                    vk::ImageSubresourceRange subresource;
                                    subresource.setAspectMask(image_aspect_flags)
                                               .setBaseMipLevel(0)
                                               .setLevelCount(1)
                                               .setBaseArrayLayer(0)
                                               .setLayerCount(1);
                                    barrier.setSrcStageMask(src_stage_mask)
                                           .setSrcAccessMask(src_access_mask)
                                           .setDstStageMask(dst_stage_mask)
                                           .setDstAccessMask(dst_access_mask)
                                           .setOldLayout(old_layout)
                                           .setNewLayout(new_layout)
                                           .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                           .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                           .setImage(image)
                                           .setSubresourceRange(subresource);

                                    vk::DependencyInfo dependency_info;
                                    dependency_info.setDependencyFlags({})
                                                   .setImageMemoryBarriers(barrier);
                                    graphicsCommandBuffers[frameIndex].pipelineBarrier2(dependency_info);
        }
        void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels_) const{
            vk::ImageMemoryBarrier barrier;
            vk::ImageSubresourceRange subresourceRange;
            subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1).setLevelCount(mipLevels_);
            barrier.setOldLayout(oldLayout).setNewLayout(newLayout)
                   .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored).setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                   .setImage(image).setSubresourceRange(subresourceRange);
            vk::PipelineStageFlags sourceStage;
            vk::PipelineStageFlags destinationStage;
            if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal){
                barrier.srcAccessMask = {};
                barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

                sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
                destinationStage = vk::PipelineStageFlagBits::eTransfer;
            }
            else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal){
                barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite; //wait for "transferWrite opearation" to complete.
                barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead; // make the shader able to read visible data.

                sourceStage      = vk::PipelineStageFlagBits::eTransfer; //wait for transfer stage to complete all access.
                destinationStage = vk::PipelineStageFlagBits::eFragmentShader; //if the transfer stage completes, the process can move to the fragment shader stage.
            }
            else{
                throw std::invalid_argument("unsupported layout transition!");
            }
            commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
        }

        void createSyncObjects(){
            assert(presentWaitSemaphores.empty() && presentCompleteSemaphores.empty() && inFlightFences.empty());
            vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> SemaphoreType;
            SemaphoreType.get<vk::SemaphoreTypeCreateInfo>().setSemaphoreType(vk::SemaphoreType::eTimeline)
                                                            .setInitialValue(0);
            renderFinishedTimelineSemaphore = vk::raii::Semaphore(device, SemaphoreType.get<vk::SemaphoreCreateInfo>());
            for(int i = 0 ; i < MAX_FRAMES_IN_FLIGHT ; ++i){
                inFlightFences.emplace_back(device, vk::FenceCreateInfo(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)));
                presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            }
            for(int i = 0 ; i < swapchainImages.size() ; ++i){
                presentWaitSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            }
        }

        bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
        {
            // Check if the physicalDevice supports the Vulkan 1.3 API version
            bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

            // Check if any of the queue families support graphics operations
            auto queueFamilies = physicalDevice.getQueueFamilyProperties();
            bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp)
                                                        { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

            // Check if all required physicalDevice extensions are available
            auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
            bool supportsAllRequiredExtensions =
                std::ranges::all_of(requiredDeviceExtensions,
                                    [&availableDeviceExtensions](auto const &requiredDeviceExtension)
                                    {
                                        return std::ranges::any_of(availableDeviceExtensions,
                                                                   [requiredDeviceExtension](auto const &availableDeviceExtension)
                                                                   { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                                    });

            // Check if the physicalDevice supports the required features (dynamic rendering and extended dynamic state)
            auto features =
                physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                                            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

            // Return true if the physicalDevice meets all the criteria
            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
        }

        void pickPhysicalDevice()
        {
            std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
            std::multimap<int, vk::raii::PhysicalDevice> candidates;
            for(auto const &pd : physicalDevices) {
                if(isDeviceSuitable(pd)) {
                    int score = 0;
                    auto properties = pd.getProperties();
                    if(properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score += 1000;
                    else if(properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) score += 500;

                    score += properties.limits.maxImageDimension2D;

                    candidates.insert(std::make_pair(score, pd));
                }
            }
            if(!candidates.empty() && candidates.rbegin()-> first > 0) {
                physicalDevice = candidates.rbegin()-> second;
                if(enableValidationLayers) std::cout << "GPU Information: " << physicalDevice.getProperties().deviceName << std::endl;
            }
            else{
                throw std::runtime_error("failed to find a suitable GPU!");
            }
            
        }

        void setupDebugMessenger() {
            if(!enableValidationLayers) return;

        vk::DebugUtilsMessengerCreateInfoEXT createInfo;
        createInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                  .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
                  .setPfnUserCallback(&debugCallback);

            debugMessenger = instance.createDebugUtilsMessengerEXT( createInfo );
        }

        void createSwapChain(){
            vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR( *surface );
            swapchainExtent                                = chooseSwapExtent(surfaceCapabilities);
            uint32_t minImageCount                         = chooseSwapMinImageCount(surfaceCapabilities);

            std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR( *surface );
            swapchainSurfaceFormat                             = chooseSwapSurfaceFormat(availableFormats);

            std::vector<vk::PresentModeKHR> availablePresent = physicalDevice.getSurfacePresentModesKHR( *surface );
            vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresent);

        vk::SwapchainCreateInfoKHR swapChainCreateInfo;
        swapChainCreateInfo.setSurface(*surface)
                           .setMinImageCount(minImageCount)
                           .setImageFormat(swapchainSurfaceFormat.format)
                           .setImageColorSpace(swapchainSurfaceFormat.colorSpace)
                           .setImageExtent(swapchainExtent)
                           .setImageArrayLayers(1)
                           .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                           .setImageSharingMode(vk::SharingMode::eExclusive)
                           .setPreTransform(surfaceCapabilities.currentTransform)
                           .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                           .setPresentMode(presentMode)
                           .setClipped(true)
                           .setOldSwapchain(nullptr);

        swapChain       = vk::raii::SwapchainKHR( device, swapChainCreateInfo );
        swapchainImages = swapChain.getImages();
}

        vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats){
            const auto formatIt = std::ranges::find_if(
                availableFormats,
                [](const auto &format)
                { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
            return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
        }

        vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities){
            if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
                return capabilities.currentExtent;
            }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
                };
        }

        uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities){
            auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
            if((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)){
                minImageCount = surfaceCapabilities.maxImageCount;
            }
            return minImageCount;
        }

        vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes){
            assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
            return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
               vk::PresentModeKHR::eMailbox :
               vk::PresentModeKHR::eFifo;
        }

        void mainLoop() {
            while(!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
            }
            device.waitIdle();
        }

        void drawFrame(){
            auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
            if(fenceResult != vk::Result::eSuccess){
                throw std::runtime_error("failed to wait for fence!");
            }
            
            auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
            if(result == vk::Result::eErrorOutOfDateKHR) {
                recreateSwapChain();
                return;
            }
            else if(result  != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
              throw std::runtime_error("failed to acquire swap chain image!");
            }

            updateUniformBuffer(frameIndex);

            //Only reset the fence if we are submitting work
            device.resetFences(*inFlightFences[frameIndex]);
            graphicsCommandBuffers[frameIndex].reset();
            recordCommandBuffer(imageIndex);

            uint64_t signalValue = ++frameCount;
            std::array<vk::Semaphore, 2> signalSemaphores = {
                *renderFinishedTimelineSemaphore,
                *presentWaitSemaphores[imageIndex]
            };
            std::array<uint64_t, 2> signalValues = { signalValue, 0 };

            vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo;
            timelineSubmitInfo.setSignalSemaphoreValueCount(2)
                              .setPSignalSemaphoreValues(signalValues.data());

            //submitting the commandBuffer
            vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
            const vk::SubmitInfo submitInfo = [&]() {
              vk::SubmitInfo info;
              info.setWaitSemaphores(*presentCompleteSemaphores[frameIndex])
                  .setWaitDstStageMask(waitDestinationStageMask)
                  .setCommandBuffers(*graphicsCommandBuffers[frameIndex])
                  .setSignalSemaphores(signalSemaphores)
                  .setPNext(&timelineSubmitInfo);
              return info;
            }();

            graphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

            const vk::PresentInfoKHR presentInfoKHR = [this, &imageIndex]() {
                vk::PresentInfoKHR info;
                info.setWaitSemaphores(*presentWaitSemaphores[imageIndex])
                    .setSwapchains(*swapChain)
                    .setImageIndices(imageIndex);
                return info;
            }();
            result = graphicsQueue.presentKHR(presentInfoKHR);
            if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            }
            else {
                assert(result == vk::Result::eSuccess);
            }
            frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void updateUniformBuffer(uint32_t currentImage){
            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

            UniformBufferObject ubo{};
            ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj =
                    glm::perspective(glm::radians(45.0f), static_cast<float>(swapchainExtent.width) /static_cast<float>(swapchainExtent.height) , 0.1f, 100.0f);
            ubo.proj[1][1] *= -1;

            memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
        }

        void cleanupSwapChain(){
            swapchainImageViews.clear();
            swapChain = nullptr;
        }

        void recreateSwapChain(){
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            while(width == 0 && height == 0){
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }
            device.waitIdle();

            cleanupSwapChain();
            createSwapChain();
            createImageViews();
            createDepthResources();
        }

        void cleanup() {
            cleanupSwapChain();

            glfwDestroyWindow(window);
            glfwTerminate();
        }

        void createInstance() {
            vk::ApplicationInfo appInfo;
            appInfo.setPApplicationName("Hello Triangle")
                   .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
                   .setPEngineName("No Engine")
                   .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
                   .setApiVersion(vk::ApiVersion14);
            //get required layers
            std::vector<const char*> RequiredLayers;
            if(enableValidationLayers){
                RequiredLayers.assign(validationLayers.begin(), validationLayers.end());
            }
            auto LayerProperties = vk::enumerateInstanceLayerProperties();
                 auto unsupportedLayersIt = std::ranges::find_if(RequiredLayers,
                                                               [&LayerProperties](auto const &RequiredLayer){
                                                                    return std::ranges::none_of(LayerProperties, [RequiredLayer](auto const &LayerProperty){
                                                                        return strcmp(RequiredLayer, LayerProperty.layerName) == 0;});
                                                                });
            if(unsupportedLayersIt != RequiredLayers.end()){
                throw std::runtime_error(std::string("validation layer requested, but not available: ") + *unsupportedLayersIt);
            }
            //get required extensions
            auto RequiredExtensions = GetRequiredExtension();

            //check if all required extensions are supported
            auto ExtensionProperties = vk::enumerateInstanceExtensionProperties();
                auto unsupportedExtensionsIt = std::ranges::find_if(RequiredExtensions,
                                                               [&ExtensionProperties](auto const &RequiredExtension){
                                                                    return std::ranges::none_of(ExtensionProperties, [RequiredExtension](auto const &ExtensionProperty){
                                                                        return strcmp(RequiredExtension, ExtensionProperty.extensionName) == 0;});
                                                               });
            if(unsupportedExtensionsIt != RequiredExtensions.end()) {
                throw std::runtime_error(std::string("extension requested, but not available: ") + *unsupportedExtensionsIt);
            }
                                                              
            vk::InstanceCreateInfo createInfo;
            createInfo.setPApplicationInfo(&appInfo)
                      .setEnabledLayerCount(static_cast<uint32_t>(RequiredLayers.size()))
                      .setPpEnabledLayerNames(RequiredLayers.data())
                      .setEnabledExtensionCount(static_cast<uint32_t>(RequiredExtensions.size()))
                      .setPpEnabledExtensionNames(RequiredExtensions.data());

            instance = vk::raii::Instance(context, createInfo);
        }

        void createLogicalDevice() {
            // find the index of the first queue family that supports graphics
            std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
            auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](const auto &qfp)
                                                                    { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); });
            if(graphicsQueueFamilyProperty != queueFamilyProperties.end()){
                auto graphicIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
                graphicsQueueIndex = graphicIndex;
            }
            if(graphicsQueueIndex == ~0) throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
            //find the index of the first queue family that supports transfer queue
            auto transferQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [&](const auto& qfp){ 
                                                                        if(&qfp == &queueFamilyProperties[graphicsQueueIndex]) return false;
                                                                        return (qfp.queueFlags & vk::QueueFlagBits::eTransfer) != static_cast<vk::QueueFlags>(0); 
                                                                    });
            if(transferQueueFamilyProperty != queueFamilyProperties.end()){
                auto transferIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), transferQueueFamilyProperty));
                transferQueueIndex = transferIndex;
            }
            if(transferQueueIndex == ~0) throw std::runtime_error("Could not find a queue for transfer");

            //enabledPhysicalDeviceFeatures
            vk::StructureChain<vk::PhysicalDeviceFeatures2,
                               vk::PhysicalDeviceVulkan13Features,
                               vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                               vk::PhysicalDeviceShaderDrawParametersFeatures,
                               vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> featureChain;
            auto& deviceFeatures2 = featureChain.get<vk::PhysicalDeviceFeatures2>();
            deviceFeatures2.features.setFillModeNonSolid(true);
            deviceFeatures2.features.setGeometryShader(false);
            deviceFeatures2.features.setSamplerAnisotropy(true);

            featureChain.get<vk::PhysicalDeviceVulkan13Features>().setDynamicRendering(true);
            featureChain.get<vk::PhysicalDeviceVulkan13Features>().setSynchronization2(true);
            featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().setExtendedDynamicState(true);
            featureChain.get<vk::PhysicalDeviceShaderDrawParametersFeatures>().setShaderDrawParameters(true);
            featureChain.get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().setTimelineSemaphore(true);
            //Create graphic & transfer queue
            std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
            float queuePriority = 0.5f;
            vk::DeviceQueueCreateInfo graphicQueueCreateInfo;
            graphicQueueCreateInfo.setQueueFamilyIndex(graphicsQueueIndex)
                                 .setQueueCount(1)
                                 .setPQueuePriorities(&queuePriority);
                queueCreateInfos.emplace_back(graphicQueueCreateInfo);
            vk::DeviceQueueCreateInfo transferQueueCreateInfo;
            transferQueueCreateInfo.setQueueFamilyIndex(transferQueueIndex)
                                   .setQueueCount(1)
                                   .setPQueuePriorities(&queuePriority);
                queueCreateInfos.emplace_back(transferQueueCreateInfo);

            vk::DeviceCreateInfo deviceCreateInfo;
            deviceCreateInfo.setPNext(&featureChain.get<vk::PhysicalDeviceFeatures2>())
                            .setQueueCreateInfos(queueCreateInfos)
                            .setEnabledExtensionCount(static_cast<uint32_t>(requiredDeviceExtension.size()))
                            .setPpEnabledExtensionNames(requiredDeviceExtension.data());

            device = vk::raii::Device(physicalDevice, deviceCreateInfo);

            graphicsQueue = vk::raii::Queue(device, graphicsQueueIndex, 0);

            transferQueue = vk::raii::Queue(device, transferQueueIndex, 0);
        }

        void createSurface()
        {
            VkSurfaceKHR _surface;
            if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
            {
                throw std::runtime_error("failed to create window surface!");
            }
            surface = vk::raii::SurfaceKHR(instance, _surface);
        }

        std::vector<const char*> GetRequiredExtension(){
            uint32_t glfwExtensionsCount = 0;
            auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);
            
            std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionsCount);
            if(enableValidationLayers){
                extensions.push_back(vk::EXTDebugUtilsExtensionName);
            }

            return extensions;
        }

        vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
            vk::ShaderModuleCreateInfo createInfo;
            createInfo.setCodeSize(code.size() * sizeof(char))
                      .setPCode(reinterpret_cast<const uint32_t *>(code.data()));
            vk::raii::ShaderModule shaderModule{device, createInfo};

            return shaderModule;
        }

        static std::vector<char> ReadFile(const std::string &filename) {
          std::ifstream file(filename, std::ios::ate | std::ios::binary);
          if (!file.is_open()) {
            throw std::runtime_error("Failed to load file: " + filename);
          }

          const auto endPos = file.tellg();
          if(endPos == -1){
            throw std::runtime_error("Failed to get file size: " + filename);
          }
          std::vector<char> buffer(endPos);
          file.seekg(0, std::ios::beg);
          file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
          if (!file) {
            throw std::runtime_error("Failed to read complete file content.");
          }
          file.close();

          return buffer;
        }

        static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT Severity,
            vk::DebugUtilsMessageTypeFlagsEXT Type,
            const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
            void *)
        {
            if (Severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || Severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            {
                std::cerr << "validation layer: type " << to_string(Type) << " msg: " << pCallbackData->pMessage << std::endl;
            }
            return VK_FALSE;
        }
};


int main(){
    HelloTriangleApplication app;
    try{
        app.run();
    } catch (const std::exception& e) {
        std::cerr <<e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
