struct VertexInPut {
    float2 inPosition;
    float2 inTexCoord;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

struct VertexOutput {
    float4 pos : SV_Position;
    float2 fragTexCoord;
};

[shader("vertex")]
VertexOutput vertMain(VertexInPut InPut) {
    VertexOutput Output;
    Output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(InPut.inPosition, 0.0, 1.0))));
    Output.fragTexCoord = InPut.inTexCoord;

    return Output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {

    return texture.Sample(inVert.fragTexCoord);
}
