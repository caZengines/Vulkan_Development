struct VertexInPut {
    float2 inPosition;
    float3 inColor;
}

struct VertexOutput {
    float3 color;
    float4 pos : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(VertexInPut InPut) {
    VertexOutput Output;
    Output.color = InPut.inColor;
    Output.pos = float4(InPut.inPosition, 0.0, 1.0);

    return Output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float3 color = inVert.color;

    return float4(color, 1.0);
}
