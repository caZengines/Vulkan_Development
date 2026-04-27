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

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> RequiredDeviceExtension = {
    vk::KHRSwapchainExtensionName
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec2 position;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription description;
        description.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::eVertex);

        return description;
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription() {
        vk::VertexInputAttributeDescription posAttribute;
        posAttribute.setLocation(0).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, position));
        vk::VertexInputAttributeDescription colorAttribute;
        colorAttribute.setLocation(1).setFormat(vk::Format::eR32G32B32Sfloat).setOffset(offsetof(Vertex, color));

        return std::array<vk::VertexInputAttributeDescription, 2>{posAttribute, colorAttribute};
    }
};

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

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
        vk::raii::SwapchainKHR                   swapChain      = nullptr;
        std::vector<vk::Image>                   swapchainImages;
        vk::SurfaceFormatKHR                     swapchainSurfaceFormat;
        vk::Extent2D                             swapchainExtent;
        std::vector<vk::raii::ImageView>         swapchainImageViews;

        uint32_t                                 queueIndex       = ~0;

        vk::raii::PipelineLayout                 pipelineLayout   = nullptr;
        vk::raii::Pipeline                       graphicsPipeline = nullptr;
        vk::raii::CommandPool                    commandPool      = nullptr;
        std::vector<vk::raii::CommandBuffer>     commandBuffers;
        vk::raii::Buffer                         vertexBuffer       = nullptr;
        vk::raii::DeviceMemory                   vertexBufferMemory = nullptr;
        uint32_t                                 frameIndex       = 0;

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
            createGraphicsPipeline();
            createCommandPool();
            createVertexBuffers();
            createCommandBuffers();
            createSyncObjects();
        }

        void createImageViews(){
            assert(swapchainImageViews.empty());
            for(auto image : swapchainImages){
                vk::ImageViewCreateInfo imageViewCreateInfo;
                imageViewCreateInfo.setViewType(vk::ImageViewType::e2D)
                                   .setFormat(swapchainSurfaceFormat.format)
                                   .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
                                   .setImage(image);
                swapchainImageViews.emplace_back(device, imageViewCreateInfo);
            }
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
            vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height));
            vk::Rect2D scissor {vk::Offset2D{0, 0}, swapchainExtent};
            vk::PipelineViewportStateCreateInfo viewportState;
            viewportState.setViewportCount(1).setScissorCount(1);

            std::vector<vk::DynamicState> dynamicStates = {
                vk::DynamicState::eViewport, vk::DynamicState::eScissor};
            vk::PipelineDynamicStateCreateInfo dynamicState;
            dynamicState.setDynamicStateCount(static_cast<uint32_t>(dynamicStates.size()))
                        .setPDynamicStates(dynamicStates.data());

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
                      .setFrontFace(vk::FrontFace::eClockwise)
                      .setDepthBiasEnable(vk::False)
                      .setLineWidth(1.0f);
            //multisampling
            vk::PipelineMultisampleStateCreateInfo multisampling;
            multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1)
                         .setSampleShadingEnable(vk::False);
            //depth and stencil testing
            vk::PipelineDepthStencilStateCreateInfo depthStenciltesing;
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
            pipelineLayoutInfo.setSetLayoutCount(0)
                              .setPushConstantRangeCount(0);
            pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

            //dynamic rendering
            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain;
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().setStages(shaderStages)
                                                                         .setPVertexInputState(&vertexInputInfo)
                                                                         .setPInputAssemblyState(&inputAssembly)
                                                                         .setPViewportState(&viewportState)
                                                                         .setPRasterizationState(&rasterizer)
                                                                         .setPMultisampleState(&multisampling)
                                                                         .setPDepthStencilState(&depthStenciltesing)
                                                                         .setPColorBlendState(&colorBlending)
                                                                         .setPDynamicState(&dynamicState)
                                                                         .setLayout(pipelineLayout)
                                                                         .setRenderPass(nullptr);
            
            pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>().setColorAttachmentCount(1)
                                                                          .setPColorAttachmentFormats(&swapchainSurfaceFormat.format);
                                                                          

            //createGraphicsPipeline
            graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        }

        void createCommandPool(){
            vk::CommandPoolCreateInfo poolInfo;
            poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                    .setQueueFamilyIndex(queueIndex);
            commandPool = vk::raii::CommandPool(device, poolInfo);
        }

        void createVertexBuffers(){
            vk::BufferCreateInfo bufferInfo;
            bufferInfo.setSize(sizeof(vertices[0]) * vertices.size())
                      .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
                      .setSharingMode(vk::SharingMode::eExclusive);
            vertexBuffer = vk::raii::Buffer(device, bufferInfo);
            //allocate memory
            vk::MemoryRequirements memoryRequirements = vertexBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo memoryAllocateInfo;
            memoryAllocateInfo.setAllocationSize(memoryRequirements.size)
                              .setMemoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | 
                                                                                                                                               vk::MemoryPropertyFlagBits::eHostCoherent));
            vertexBufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);
            vertexBuffer.bindMemory(*vertexBufferMemory, 0);
            
            void* data = vertexBufferMemory.mapMemory(0, bufferInfo.size);
            memcpy(data, vertices.data(), bufferInfo.size);
            vertexBufferMemory.unmapMemory();
        }

        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties){
            vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
            for(int i = 0 ; i < memoryProperties.memoryTypeCount ;++i){
                if((typeFilter & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)){
                    return i;
                }
            }
            throw std::runtime_error("failed to find suitable memory type!");
        }

        void createCommandBuffers(){
            vk::CommandBufferAllocateInfo allocInfo;
             allocInfo.setCommandPool(commandPool)
                      .setLevel(vk::CommandBufferLevel::ePrimary)
                      .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
            
            commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        }

        void recordCommandBuffer(uint32_t ImageIndex){
            commandBuffers[frameIndex].begin({});

            // Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
            transition_image_layout(
                ImageIndex,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},                                                        //srcAccessMask(no need to wait for previous operation)
                vk::AccessFlagBits2::eColorAttachmentWrite,                //dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,         //srcStage
                vk::PipelineStageFlagBits2::eColorAttachmentOutput          //dstStage
            );
            vk::ClearValue              clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
            vk::RenderingAttachmentInfo attachmentInfo;
            attachmentInfo.setImageView(swapchainImageViews[ImageIndex])
                          .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                          .setLoadOp(vk::AttachmentLoadOp::eClear)
                          .setStoreOp(vk::AttachmentStoreOp::eStore)
                          .setClearValue(clearColor);
            //renderingInfo
            vk::RenderingInfo renderingInfo;
            vk::Rect2D renderArea;
            renderArea.setOffset({0,0})
                      .setExtent(swapchainExtent);
            renderingInfo.setRenderArea(renderArea)
                         .setLayerCount(1)
                         .setColorAttachments(attachmentInfo);

            //start rendering
            commandBuffers[frameIndex].beginRendering(renderingInfo);
            //binding the graphics pipeline
            commandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
            //binding the vertexbuffer
            commandBuffers[frameIndex].bindVertexBuffers(0, *vertexBuffer, {0});
            //command buffer dynamic state
            commandBuffers[frameIndex].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),static_cast<float>(swapchainExtent.height), 0.0f, 0.0f));
            commandBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0,0), swapchainExtent));
            commandBuffers[frameIndex].draw(vertices.size(), 1, 0, 0);
            //end rendering
            commandBuffers[frameIndex].endRendering();
            // After rendering, transition the swapchain image to vk::ImageLayout::ePresentSrcKHR
            transition_image_layout(
                ImageIndex,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,        // srcAccessMask
                {},                                                // dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
            );
            
            commandBuffers[frameIndex].end();
        }
        void transition_image_layout(uint32_t                imageIndex,
                                     vk::ImageLayout         old_layout,
                                     vk::ImageLayout         new_layout,
                                     vk::AccessFlags2        src_access_mask,
                                     vk::AccessFlags2        dst_access_mask,
                                     vk::PipelineStageFlags2 src_stage_mask,
                                     vk::PipelineStageFlags2 dst_stage_mask) 
                                {
                                    vk::ImageMemoryBarrier2 barrier;
                                    vk::ImageSubresourceRange subresource;
                                    subresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
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
                                           .setImage(swapchainImages[imageIndex])
                                           .setSubresourceRange(subresource);

                                    vk::DependencyInfo dependency_info;
                                    dependency_info.setDependencyFlags({})
                                                   .setImageMemoryBarriers(barrier);
                                    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
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
            bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion12;

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
                physicalDevice
                    .template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
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
                std::cout << "GPU Information: " << physicalDevice.getProperties().deviceName << std::endl;
            }
            else{
                throw std::runtime_error("failed to find a suitable GPU!");
            }
            
        }

        void setupDebugMessenger() {
            if(!enableValidationLayers) return;

            vk::DebugUtilsMessageSeverityFlagsEXT SeverityFlags(
                                                                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            );
            vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation 
                | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            );

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
            else if(result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
              assert(result == vk::Result::eTimeout ||
                     result == vk::Result::eNotReady);
              throw std::runtime_error("failed to acquire swap chain image!");
            }
            //Only reset the fence if we are submitting work
            device.resetFences(*inFlightFences[frameIndex]);
            commandBuffers[frameIndex].reset();
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
                  .setCommandBuffers(*commandBuffers[frameIndex])
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
            auto graphicIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
            queueIndex = graphicIndex;

            //enabledPhysicalDeviceFeatures
            vk::PhysicalDeviceFeatures deviceFeatures;
            deviceFeatures.setFillModeNonSolid(true);
            vk::StructureChain<vk::PhysicalDeviceFeatures2,
                               vk::PhysicalDeviceVulkan13Features,
                               vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                               vk::PhysicalDeviceShaderDrawParametersFeatures,
                               vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> featureChain;
            auto& deviceFeatures2 = featureChain.get<vk::PhysicalDeviceFeatures2>();
            deviceFeatures2.features.setFillModeNonSolid(true);

            featureChain.get<vk::PhysicalDeviceVulkan13Features>().setDynamicRendering(true);
            featureChain.get<vk::PhysicalDeviceVulkan13Features>().setSynchronization2(true);
            featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().setExtendedDynamicState(true);
            featureChain.get<vk::PhysicalDeviceShaderDrawParametersFeatures>().setShaderDrawParameters(true);
            featureChain.get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().setTimelineSemaphore(true);

            std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
            float queuePriority = 0.5f;
            vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
            deviceQueueCreateInfo.setQueueFamilyIndex(graphicIndex)
                                 .setQueueCount(1)
                                 .setPQueuePriorities(&queuePriority);

            vk::DeviceCreateInfo deviceCreateInfo;
            deviceCreateInfo.setPNext(&featureChain.get<vk::PhysicalDeviceFeatures2>())
                            .setQueueCreateInfos(deviceQueueCreateInfo)
                            .setEnabledExtensionCount(static_cast<uint32_t>(requiredDeviceExtension.size()))
                            .setPpEnabledExtensionNames(requiredDeviceExtension.data());

            device = vk::raii::Device(physicalDevice, deviceCreateInfo);

            graphicsQueue = vk::raii::Queue(device, graphicIndex, 0);
        }

        void createSurface()
        {
            VkSurfaceKHR _surface;
            if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
            {
                throw std::runtime_error("failed to create window curface!");
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

          std::vector<char> buffer(file.tellg());
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
