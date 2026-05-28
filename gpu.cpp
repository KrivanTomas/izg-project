/*
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 *         Tomáš Křivan, xkrivat00@stud.fit.vut.cz
 */

#include <studentSolution/gpu.hpp>
#include <cstring>

#include "Tracy.hpp"

void bind_framebuffer(GPUMemory& mem, BindFramebufferCommand& cm) {
    mem.activatedFramebuffer = cm.id;
}

void bind_program(GPUMemory& mem, BindProgramCommand& cm) {
    mem.activatedProgram = cm.id;
}

void bind_vertex_array(GPUMemory& mem, BindVertexArrayCommand& cm) {
    mem.activatedVertexArray = cm.id;
}



void block_writes(GPUMemory& mem, BlockWritesCommand& cm) {
    mem.blockWrites.color = cm.blockWrites.color;
    mem.blockWrites.depth = cm.blockWrites.depth;
    mem.blockWrites.stencil = cm.blockWrites.stencil;
}

void set_backface_culling(GPUMemory& mem, SetBackfaceCullingCommand& cm) {
    mem.backfaceCulling.enabled = cm.enabled;
}

void set_front_face(GPUMemory& mem, SetFrontFaceCommand& cm) {
    mem.backfaceCulling.frontFaceIsCounterClockWise = cm.frontFaceIsCounterClockWise;
}

void set_stencil(GPUMemory& mem, SetStencilCommand& cm) {
    mem.stencilSettings = cm.settings;
}

void set_blending(GPUMemory& mem, SetBlendingCommand& cm) {
    mem.blendingSettings = cm.settings;
}

void set_draw_id(GPUMemory& mem, SetDrawIdCommand& cm) {
    mem.gl_DrawID = cm.id;
}

void write_frag_color(Framebuffer& fbo, const glm::uvec2 pos, const glm::vec4 color) {
    void *pixel_start;
    pixel_start = getPixel(fbo.color, pos.x, fbo.yReversed ? fbo.height - 1 - pos.y : pos.y);

    if(fbo.color.format == Image::U8) {
        glm::vec4 out_color = glm::clamp(color, 0.0f, 1.0f) * 255.0f;
        std::uint8_t *pixel = reinterpret_cast<std::uint8_t*>(pixel_start);
        switch(fbo.color.channels) {
            case 4:
                pixel[3] = out_color[fbo.color.channelTypes[3]];
            case 3:
                pixel[2] = out_color[fbo.color.channelTypes[2]];
            case 2:
                pixel[1] = out_color[fbo.color.channelTypes[1]];
            case 1:
                pixel[0] = out_color[fbo.color.channelTypes[0]];
            case 0:
            default:
                break;
        }
        //for(uint32_t i = 0; i < fbo.color.channels; i++) {
        //    pixel[i] = 255.0f * clamped_color[fbo.color.channelTypes[i]];
        //}
    }
    else {
        float *pixel = reinterpret_cast<float*>(pixel_start);
        switch(fbo.color.channels) {
            case 4:
                pixel[3] = color[fbo.color.channelTypes[3]];
            case 3:
                pixel[2] = color[fbo.color.channelTypes[2]];
            case 2:
                pixel[1] = color[fbo.color.channelTypes[1]];
            case 1:
                pixel[0] = color[fbo.color.channelTypes[0]];
            case 0:
            default:
                break;
        }
        //for(uint32_t i = 0; i < fbo.color.channels; i++) {
        //    pixel[i] = color[fbo.color.channelTypes[i]];
        //}
    }
}

glm::vec4 read_frag_color(Framebuffer& fbo, const glm::uvec2& pos) {
    glm::vec4 color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    void *pixel_start;
    pixel_start = getPixel(fbo.color, pos.x, fbo.yReversed ? fbo.height - 1 - pos.y : pos.y);

    constexpr float max_inverted = 1.0f / 255.0f;

    if(fbo.color.format == Image::U8) {
        std::uint8_t *pixel = reinterpret_cast<std::uint8_t*>(pixel_start);
        for(uint32_t i = 0; i < fbo.color.channels; i++) {
            color[fbo.color.channelTypes[i]] = pixel[i] * max_inverted;
        }
    }
    else {
        float *pixel = reinterpret_cast<float*>(pixel_start);
        for(uint32_t i = 0; i < fbo.color.channels; i++) {
            color[fbo.color.channelTypes[i]] = pixel[i];
        }
    }
    return color;
}

inline void write_frag_depth(Framebuffer& fbo, const glm::uvec2& pos, const float depth) {
    float *frag_depth = reinterpret_cast<float*>(getPixel(fbo.depth, pos.x, fbo.yReversed ? fbo.height - 1 - pos.y : pos.y));
    *frag_depth = depth;
}

inline void write_frag_stencil(Framebuffer& fbo, const glm::uvec2& pos, const std::uint8_t stencil) {
    uint8_t *pixel_stencil = reinterpret_cast<uint8_t*>(getPixel(fbo.stencil, pos.x, fbo.yReversed ? fbo.height - 1 - pos.y : pos.y));
    *pixel_stencil = stencil;
}

void clear_color(GPUMemory& mem, ClearColorCommand& cm) {
    Framebuffer& fbo = mem.framebuffers[mem.activatedFramebuffer];

    if(fbo.color.data == nullptr) return;

    // doesn't care about yReversed
    void *pixel_start = fbo.color.data;

    // write to first pixel
    if(fbo.color.format == Image::U8) {
        glm::vec4 clamped_color = glm::clamp(cm.value, 0.0f, 1.0f);
        std::uint8_t *pixel = reinterpret_cast<std::uint8_t*>(pixel_start);
        for(uint32_t i = 0; i < fbo.color.channels; i++) {
            pixel[i] = 255.0f * clamped_color[fbo.color.channelTypes[i]];
        }
    }
    else if(fbo.color.format == Image::F32) {
        float *pixel = reinterpret_cast<float*>(pixel_start);
        for(uint32_t i = 0; i < fbo.color.channels; i++) {
            pixel[i] = cm.value[fbo.color.channelTypes[i]];
        }
    }

    const uint8_t *copy_end = reinterpret_cast<uint8_t*>(fbo.color.data) + fbo.height * fbo.width * fbo.color.bytesPerPixel;
    uint8_t *insert_start = reinterpret_cast<uint8_t*>(fbo.color.data) + fbo.color.bytesPerPixel;
    size_t copy_bytes = fbo.color.bytesPerPixel;

    // exponentialy copy the pixels
    while(insert_start + copy_bytes < copy_end) {
        std::memcpy(insert_start, fbo.color.data, copy_bytes);
        insert_start += copy_bytes;
        copy_bytes <<= 1;
    }

    // copy the remainder
    std::memcpy(insert_start, fbo.color.data, copy_end - insert_start);
}

void clear_depth(GPUMemory& mem, ClearDepthCommand& cm) {
    Framebuffer& fbo = mem.framebuffers[mem.activatedFramebuffer];

    if(fbo.depth.data == nullptr) return;

    // write to first pixel
    float *first = reinterpret_cast<float*>(fbo.depth.data);
    *first = cm.value;

    const uint8_t *copy_end = reinterpret_cast<uint8_t*>(fbo.depth.data) + fbo.height * fbo.width * fbo.depth.bytesPerPixel;
    uint8_t *insert_start = reinterpret_cast<uint8_t*>(fbo.depth.data) + fbo.depth.bytesPerPixel;
    size_t copy_bytes = fbo.depth.bytesPerPixel;

    // exponentialy copy the pixels
    while(insert_start + copy_bytes < copy_end) {
        std::memcpy(insert_start, fbo.depth.data, copy_bytes);
        insert_start += copy_bytes;
        copy_bytes <<= 1;
    }

    // copy the remainder
    std::memcpy(insert_start, fbo.depth.data, copy_end - insert_start);
}

void clear_stencil(GPUMemory& mem, ClearStencilCommand& cm) {
    Framebuffer& fbo = mem.framebuffers[mem.activatedFramebuffer];

    if(fbo.stencil.data == nullptr) return;

    // write to first pixel
    uint8_t *first = reinterpret_cast<uint8_t*>(fbo.stencil.data);
    *first = cm.value;

    const uint8_t *copy_end = reinterpret_cast<uint8_t*>(fbo.stencil.data) + fbo.height * fbo.width * fbo.stencil.bytesPerPixel;
    uint8_t *insert_start = reinterpret_cast<uint8_t*>(fbo.stencil.data) + fbo.stencil.bytesPerPixel;
    size_t copy_bytes = fbo.stencil.bytesPerPixel;

    // exponentialy copy the pixels
    while(insert_start + copy_bytes < copy_end) {
        std::memcpy(insert_start, fbo.stencil.data, copy_bytes);
        insert_start += copy_bytes;
        copy_bytes <<= 1;
    }

    // copy the remainder
    std::memcpy(insert_start, fbo.stencil.data, copy_end - insert_start);
}

void user(GPUMemory& mem, UserCommand& cm) {
    if(cm.callback == nullptr) return;

    cm.callback(cm.data);
}

void compute_vertex_id(GPUMemory &mem, InVertex& in, std::uint32_t index) {
    VertexArray *array = mem.vertexArrays + mem.activatedVertexArray;
    const std::uint8_t *indexing_start = nullptr;
    if(array->indexBufferID >= 0) {
        // indexing on
        indexing_start = reinterpret_cast<const std::uint8_t*>((mem.buffers + array->indexBufferID)->data) + array->indexOffset;
    }
    if(array->indexBufferID >= 0) {
        // indexing on
        if(array->indexType == IndexType::U8) {
            in.gl_VertexID = indexing_start[index];
        }
        else if(array->indexType == IndexType::U16) {
            in.gl_VertexID = reinterpret_cast<const std::uint16_t*>(indexing_start)[index];
        }
        else if(array->indexType == IndexType::U32) {
            in.gl_VertexID = reinterpret_cast<const std::uint32_t*>(indexing_start)[index];
        }
    }
    else {
        in.gl_VertexID = index;
    }
}

void vertex_assembly(GPUMemory& mem, VertexArray *array, InVertex& vertex) {
    for(std::uint32_t attrib_idx = 0; attrib_idx < maxAttribs; attrib_idx++) {
        VertexAttrib& attrib = array->vertexAttrib[attrib_idx];
        if(attrib.bufferID < 0) continue;
        Buffer& buffer = mem.buffers[attrib.bufferID];
        

        const void *value = reinterpret_cast<const char*>(buffer.data) + attrib.offset + attrib.stride * vertex.gl_VertexID;
        const float *float_value = reinterpret_cast<const float*>(value);
        const std::uint32_t *uint_value = reinterpret_cast<const std::uint32_t*>(value);
        switch(attrib.type) {
            case AttribType::FLOAT:
                vertex.attributes[attrib_idx].v1 = float_value[0];
                break;
            case AttribType::VEC2:
                vertex.attributes[attrib_idx].v2 = 
                    glm::vec2(
                            float_value[0],
                            float_value[1]);
                break;
            case AttribType::VEC3:
                vertex.attributes[attrib_idx].v3 = 
                    glm::vec3(
                            float_value[0],
                            float_value[1],
                            float_value[2]);
                break;
            case AttribType::VEC4:
                vertex.attributes[attrib_idx].v4 = 
                    glm::vec4(
                            float_value[0],
                            float_value[1],
                            float_value[2],
                            float_value[3]);
                break;
            case AttribType::UINT:
                vertex.attributes[attrib_idx].u1 = uint_value[0];
                break;
            case AttribType::UVEC2:
                vertex.attributes[attrib_idx].u2 =
                    glm::uvec2(
                            uint_value[0],
                            uint_value[1]);
                break;
            case AttribType::UVEC3:
                vertex.attributes[attrib_idx].u3 =
                    glm::uvec3(
                            uint_value[0],
                            uint_value[1],
                            uint_value[2]);
                break;
            case AttribType::UVEC4:
                vertex.attributes[attrib_idx].u4 =
                    glm::uvec4(
                            uint_value[0],
                            uint_value[1],
                            uint_value[2],
                            uint_value[3]);
                break;
            case AttribType::EMPTY:
                break;
        }
    }
}

struct Primitive {
    OutVertex verts[3];
    AttribType type[maxAttribs];
};

void primitive_assembly(GPUMemory& mem, Primitive& t, std::uint32_t offset) {
    InVertex in;

    VertexArray *vertex_array = mem.vertexArrays + mem.activatedVertexArray;
    Program *prog = mem.programs + mem.activatedProgram;

    ShaderInterface interface;
    interface.gl_DrawID = mem.gl_DrawID;
    interface.textures = mem.textures;
    interface.uniforms = mem.uniforms;


    // per vertex in triangle
    for(unsigned i = 0; i < 3; i++) {
        compute_vertex_id(mem, in, offset + i);
        vertex_assembly(mem, vertex_array, in);
        if(prog->vertexShader) {
            prog->vertexShader(t.verts[i], in, interface);
        }
    }
}

void clip_point(Primitive &tri, int clip_idx, int to_idx) {
    OutVertex& clipped_vert = tri.verts[clip_idx];
    OutVertex& to_vert = tri.verts[to_idx];
    glm::vec4 clip = clipped_vert.gl_Position; 
    glm::vec4 to = to_vert.gl_Position; 
    float t = (-to.w - to.z) / (clip.w - to.w + clip.z - to.z);

    for(std::uint32_t attrib_idx = 0; attrib_idx < maxAttribs; attrib_idx++) {
        Attrib& clipped_attrib = clipped_vert.attributes[attrib_idx];
        Attrib& to_attrib = to_vert.attributes[attrib_idx];
        switch(tri.type[attrib_idx]) {
            case AttribType::FLOAT:
            case AttribType::VEC2:
            case AttribType::VEC3:
            case AttribType::VEC4:
                clipped_attrib.v4 = to_attrib.v4 + t * (clipped_attrib.v4 - to_attrib.v4);
                break;
            case AttribType::UINT:
            case AttribType::UVEC2:
            case AttribType::UVEC3:
            case AttribType::UVEC4:
                clipped_attrib.u4 = glm::round(glm::vec4(to_attrib.u4) + t * glm::vec4(clipped_attrib.u4 - to_attrib.u4));
                break;
            case AttribType::EMPTY:
                break;
        }
        
    }
  
    clipped_vert.gl_Position = to + t * (clip - to);

}

int clipping(Primitive& t, Primitive& u) {
    int clipped_count = 0;
    int not_clipped;
    int clipped;
    for(int i = 0; i < 3; i++) {
        glm::vec4 point = t.verts[i].gl_Position;
        // clip triangles with near-plane
        if(!(-point.w <= point.z)) {
            clipped_count++;
            clipped = i;
        }
        else {
            not_clipped = i;
        }

    }

    if(clipped_count == 3) return 0;
    if(clipped_count == 0) return 1;

    if(clipped_count == 2) {
        for(int i = 0; i < 3; i++) {
            if(i == not_clipped) continue;
            clip_point(t, i, not_clipped);
        }
        return 1;
    }
    
    // clipped_count == 1
    u = t;
    clip_point(t, clipped, (clipped + 1) % 3);
    u.verts[(clipped + 1) % 3] = t.verts[clipped];
    clip_point(u, clipped, (clipped + 2) % 3);

    return 2;
}

void perspective_division(GPUMemory& mem, Primitive& t) {
    // per vertex in triangle (clip-space to ndc)
    for(unsigned i = 0; i < 3; i++) {
        auto pos = t.verts[i].gl_Position;
        auto w_inv = 1.0f / pos.w;
        pos.x *= w_inv;
        pos.y *= w_inv;
        pos.z *= w_inv;
        t.verts[i].gl_Position = pos;
    }
}

void viewport_transform(GPUMemory& mem, Framebuffer& framebuffer, Primitive& t) {
    std::uint32_t width = framebuffer.width;
    std::uint32_t height = framebuffer.height;

    // per vertex in triangle (ndc to screen-space)
    for(unsigned i = 0; i < 3; i++) {
        glm::vec4 pos = t.verts[i].gl_Position;
        pos.x = (pos.x * 0.5f + 0.5f) * width;
        pos.y = (pos.y * 0.5f + 0.5f) * height;
        t.verts[i].gl_Position = pos;
    }
}

bool should_cull(GPUMemory const& mem, Primitive const& t, bool& front_facing) {

    glm::mat3 mat{
        t.verts[0].gl_Position,
        t.verts[1].gl_Position,
        t.verts[2].gl_Position
    };

    mat[0][2] = 1.0f;
    mat[1][2] = 1.0f;
    mat[2][2] = 1.0f;

    float det = glm::determinant(mat);
    bool is_clockwise = det < 0.0f;

    // always cull lines
    float eps = 1e-9f;
    if(det < eps && det > -eps) {
        return true;
    }

    if(mem.backfaceCulling.enabled == false) {
        front_facing = mem.backfaceCulling.frontFaceIsCounterClockWise ? !is_clockwise : is_clockwise;
        return false;
    }
    if(mem.backfaceCulling.frontFaceIsCounterClockWise) {
        front_facing = !is_clockwise;
        return is_clockwise;
    }
    else {
        front_facing = is_clockwise;
        return !is_clockwise;
    }
}

struct Bounds {
    float right = -INFINITY;
    float up = -INFINITY;
    float left = INFINITY;
    float down = INFINITY;
};

struct Barycentrics {
    // hl3 confirmed
    glm::vec3 lambda;
};

// also computes barycentrics
inline bool is_inside_triangle(glm::vec2 pos, Primitive const& t, Barycentrics& bary) {
    ZoneScoped;
    //glm::dvec2 a = t.verts[0].gl_Position;
    //glm::dvec2 b = t.verts[1].gl_Position;
    //glm::dvec2 c = t.verts[2].gl_Position;
    //glm::dvec2 p = pos;

    //glm::dvec2 v0 = b - a, v1 = c - a, v2 = p - a;
    //if(bary.inv_denom_cache == 0) {
    //    bary.inv_denom_cache = 1.0f / (v0.x * v1.y - v1.x * v0.y);
    //}

    //bary.lambda[1] = (v2.x * v1.y - v1.x * v2.y) * bary.inv_denom_cache;
    //bary.lambda[2] = (v0.x * v2.y - v2.x * v0.y) * bary.inv_denom_cache;
    ////bary.lambda[0] = ((v0.x - v2.x) * (v1.y - v2.y) - (v1.x - v2.x) * (v0.y - v2.y)) * inv_denom;
    //// this should really be 1 - (u + v), but then the some tests wouldn't pass
    //bary.lambda[0] = 1.0 - (bary.lambda[1] + bary.lambda[2] + 1e-10);

    bool inside = bary.lambda[0] >= 0.0f && bary.lambda[1] >= 0.0f && bary.lambda[2] >= 0.0f;

    return inside;
}

void stencil_operation(GPUMemory const& mem, std::uint8_t *frag_stencil, StencilOp op) {
    switch(op) {
        case StencilOp::KEEP:
            break;
        case StencilOp::ZERO:
            *frag_stencil = 0;
            break;
        case StencilOp::REPLACE:
            *frag_stencil = mem.stencilSettings.refValue;
            break;
        case StencilOp::INCR:
            if(*frag_stencil != 255)
                (*frag_stencil)++; 
            break;
        case StencilOp::INCR_WRAP:
            (*frag_stencil)++;
            break;
        case StencilOp::DECR:
            if(*frag_stencil != 0)
                (*frag_stencil)--; 
            break;
        case StencilOp::DECR_WRAP:
            (*frag_stencil)--;
            break;
        case StencilOp::INVERT:
            *frag_stencil = 255 - *frag_stencil;
            break;
    }
}

bool stencil_test_pass(GPUMemory const& mem, Framebuffer& framebuffer, const glm::uvec2 pos, const bool front_facing) {
    if(!mem.stencilSettings.enabled || framebuffer.stencil.data == nullptr) return true;

    std::uint8_t *stencil;
    stencil = reinterpret_cast<std::uint8_t*>(getPixel(framebuffer.stencil, pos.x, framebuffer.yReversed ? framebuffer.height - 1 - pos.y : pos.y));

    bool test_result;
    auto ref = mem.stencilSettings.refValue;

    // Stencil test by operation
    switch(mem.stencilSettings.func) {
        case StencilFunc::ALWAYS:
            test_result = true;
            break;
        case StencilFunc::NEVER:
            test_result = false;
            break;
        case StencilFunc::EQUAL:
            test_result = *stencil == ref;
            break;
        case StencilFunc::NOTEQUAL:
            test_result = *stencil != ref;
            break;
        case StencilFunc::LESS:
            test_result = *stencil < ref;
            break;
        case StencilFunc::LEQUAL:
            test_result = *stencil <= ref;
            break;
        case StencilFunc::GREATER:
            test_result = *stencil > ref;
            break;
        case StencilFunc::GEQUAL:
            test_result = *stencil >= ref;
            break;
    }

    if(!mem.blockWrites.stencil && !test_result) { 
        // sfail
        stencil_operation(mem, stencil, 
                front_facing ?  mem.stencilSettings.frontOps.sfail :
                                mem.stencilSettings.backOps.sfail);
    }
    return test_result;
}

bool depth_test_pass(GPUMemory& mem, Framebuffer& framebuffer, InFragment const& in, const glm::uvec2 pos, const bool front_facing) {
    if(framebuffer.depth.data == nullptr) return true;

    float *depth;
    depth = reinterpret_cast<float*>(getPixel(framebuffer.depth, pos.x, framebuffer.yReversed ? framebuffer.height - 1 - pos.y : pos.y));
    bool result = *depth > in.gl_FragCoord.z;
    if(result) return true;

    // try dpfail write to stencil
    if(!mem.stencilSettings.enabled || mem.blockWrites.stencil || framebuffer.stencil.data == nullptr) return false;

    // write to stencil
    std::uint8_t *stencil;
    stencil = reinterpret_cast<std::uint8_t*>(getPixel(framebuffer.stencil, pos.x, framebuffer.yReversed ? framebuffer.height - 1 - pos.y : pos.y));

    stencil_operation(mem, stencil, 
            front_facing ?  mem.stencilSettings.frontOps.dpfail :
                            mem.stencilSettings.backOps.dpfail);

    return false;
}

void get_blended_frag_color(GPUMemory const& mem, Framebuffer& framebuffer, const glm::uvec2 pos, glm::vec4& src_color) {
    const BlendingSettings settings = mem.blendingSettings;
    if(!settings.enabled) return;

    glm::vec4 dst_color = read_frag_color(framebuffer, pos);

    glm::vec4 src_weight;
    switch(settings.sFactor) {
        case BlendFunc::ZERO:
            src_weight = glm::vec4(0.0f);
            break;
        case BlendFunc::ONE:
            src_weight = glm::vec4(1.0f);
            break;
        case BlendFunc::SRC_COLOR:
            src_weight = src_color;
            break;
        case BlendFunc::ONE_MINUS_SRC_COLOR:
            src_weight = glm::vec4(1.0f) - src_color;
            break;
        case BlendFunc::SRC_ALPHA:
            src_weight = glm::vec4(src_color.a);
            break;
        case BlendFunc::ONE_MINUS_SRC_ALPHA:
            src_weight = glm::vec4(1.0f) - glm::vec4(src_color.a);
            break;
        case BlendFunc::DST_COLOR:
            src_weight = dst_color;
            break;
        case BlendFunc::ONE_MINUS_DST_COLOR:
            src_weight = glm::vec4(1.0f) - dst_color;
            break;
        case BlendFunc::DST_ALPHA:
            src_weight = glm::vec4(dst_color.a);
            break;
        case BlendFunc::ONE_MINUS_DST_ALPHA:
            src_weight = glm::vec4(1.0f) - glm::vec4(dst_color.a);
            break;
    }

    glm::vec4 dst_weight;
    switch(settings.dFactor) {
        case BlendFunc::ZERO:
            dst_weight = glm::vec4(0.0f);
            break;
        case BlendFunc::ONE:
            dst_weight = glm::vec4(1.0f);
            break;
        case BlendFunc::SRC_COLOR:
            dst_weight = src_color;
            break;
        case BlendFunc::ONE_MINUS_SRC_COLOR:
            dst_weight = glm::vec4(1.0f) - src_color;
            break;
        case BlendFunc::SRC_ALPHA:
            dst_weight = glm::vec4(src_color.a);
            break;
        case BlendFunc::ONE_MINUS_SRC_ALPHA:
            dst_weight = glm::vec4(1.0f) - glm::vec4(src_color.a);
            break;
        case BlendFunc::DST_COLOR:
            dst_weight = dst_color;
            break;
        case BlendFunc::ONE_MINUS_DST_COLOR:
            dst_weight = glm::vec4(1.0f) - dst_color;
            break;
        case BlendFunc::DST_ALPHA:
            dst_weight = glm::vec4(dst_color.a);
            break;
        case BlendFunc::ONE_MINUS_DST_ALPHA:
            dst_weight = glm::vec4(1.0f) - glm::vec4(dst_color.a);
            break;
    }

    switch(settings.equation) {
        case BlendEquation::ADD:
            src_color = src_color * src_weight + dst_color * dst_weight;
            break;
        case BlendEquation::SUBTRACT:
            src_color = src_color * src_weight - dst_color * dst_weight;
            break;
        case BlendEquation::REVERSE_SUBTRACT:
            src_color = dst_color * dst_weight - src_color * src_weight;
            break;
        case BlendEquation::MIN:
            src_color = glm::min(src_color, dst_color);
            break;
        case BlendEquation::MAX:
            src_color = glm::max(src_color, dst_color);
            break;
    }
}

void create_fragment(GPUMemory& mem, glm::vec2& pos, Barycentrics& bary, Primitive const& t, InFragment& in) {
    ZoneScoped;
    in.gl_FragCoord.x = pos.x;
    in.gl_FragCoord.y = pos.y;

    // depth interpolation
    in.gl_FragCoord.z = 
        bary.lambda[0] * t.verts[0].gl_Position.z +
        bary.lambda[1] * t.verts[1].gl_Position.z +
        bary.lambda[2] * t.verts[2].gl_Position.z;

    // perspective correct barycentric coords
    float s0 = bary.lambda[0] / t.verts[0].gl_Position.w;
    float s1 = bary.lambda[1] / t.verts[1].gl_Position.w;
    float s2 = bary.lambda[2] / t.verts[2].gl_Position.w;
    float s_inv = 1.0f / (s0 + s1 + s2);
    bary.lambda[0] = s0 * s_inv;
    bary.lambda[1] = s1 * s_inv;
    bary.lambda[2] = s2 * s_inv;

    
    float a;
    // interpolate attributes
    for(std::uint32_t attrib_idx = 0; attrib_idx < maxAttribs; attrib_idx++) {
        switch(t.type[attrib_idx]) {
            case AttribType::FLOAT:
            case AttribType::VEC2:
            case AttribType::VEC3:
            case AttribType::VEC4:
                in.attributes[attrib_idx].v4 = 
                    bary.lambda[0] * t.verts[0].attributes[attrib_idx].v4 +
                    bary.lambda[1] * t.verts[1].attributes[attrib_idx].v4 +
                    bary.lambda[2] * t.verts[2].attributes[attrib_idx].v4;
                break;
            case AttribType::UINT:
            case AttribType::UVEC2:
            case AttribType::UVEC3:
            case AttribType::UVEC4:
                in.attributes[attrib_idx].u4 = 
                    glm::round(
                    bary.lambda[0] * static_cast<glm::vec4>(t.verts[0].attributes[attrib_idx].u4) +
                    bary.lambda[1] * static_cast<glm::vec4>(t.verts[1].attributes[attrib_idx].u4) +
                    bary.lambda[2] * static_cast<glm::vec4>(t.verts[2].attributes[attrib_idx].u4));
                break;
            case AttribType::EMPTY:
                break;
        }
    }
}

void rasterize(GPUMemory& mem, Framebuffer& framebuffer, Primitive const& t, const bool front_facing) {
    ZoneScoped;
    Bounds bounds;

    // per vertex in triangle (calculate bounds)
    for(unsigned i = 0; i < 3; i++) {
        glm::vec4 pos = t.verts[i].gl_Position;
        bounds.right = glm::max(bounds.right, pos.x);
        bounds.up = glm::max(bounds.up, pos.y);
        bounds.left = glm::min(bounds.left, pos.x);
        bounds.down = glm::min(bounds.down, pos.y);
    }
    // clip bounds to framebuffer
    std::uint32_t width = framebuffer.width;
    std::uint32_t height = framebuffer.height;
    bounds.right = glm::min<float>(bounds.right + 0.5f, width - 1);
    bounds.up = glm::min<float>(bounds.up + 0.5f, height - 1);
    bounds.left = glm::max<float>(bounds.left - 0.5f, 0.0f);
    bounds.down = glm::max<float>(bounds.down - 0.5f, 0.0f);

    Program *prog = mem.programs + mem.activatedProgram;
    InFragment in;
    OutFragment out;
    Barycentrics bary;

    ShaderInterface interface;
    interface.gl_DrawID = mem.gl_DrawID;
    interface.textures = mem.textures;
    interface.uniforms = mem.uniforms;


    glm::dvec2 v0 = t.verts[0].gl_Position, v1 = t.verts[1].gl_Position, v2 = t.verts[2].gl_Position; // triangle vertices

    float area2 = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);

    // Coefficients for each lambda
    double a0 = v1.x * v2.y - v2.x * v1.y;
    double b0 = v1.y - v2.y;
    double c0 = v2.x - v1.x;

    double a1 = v2.x * v0.y - v0.x * v2.y;
    double b1 = v2.y - v0.y;
    double c1 = v0.x - v2.x;

    double a2 = v0.x * v1.y - v1.x * v0.y;
    double b2 = v0.y - v1.y;
    double c2 = v1.x - v0.x;

    double inv_area2 = 1.0 / area2;

    // Deltas (per-unit steps)
    glm::dvec3 dlambda_dx = glm::dvec3(b0, b1, b2) * inv_area2;

    // per fragment in bounds
    for(std::uint32_t y = bounds.down; y <= bounds.up; y++) {
        bool line_drawn_to = false;
        // scanline rendering

        std::uint32_t x_left = glm::floor(bounds.left);
        std::uint32_t x_right = glm::ceil(bounds.right);
        bary.lambda = glm::vec3(
                (a0 + b0 * x_left + c0 * y) * inv_area2,
                (a1 + b1 * x_left + c1 * y) * inv_area2,
                (a2 + b2 * x_left + c2 * y) * inv_area2
        );
        for(std::uint32_t x = x_left; x <= x_right; x++) {
            glm::uvec2 data_pos(x, y);
            glm::vec2 pos(x, y);
            pos += 0.5f; // the center of a pixel

            if(!is_inside_triangle(pos, t, bary)) {
                if(line_drawn_to) break;
                bary.lambda += dlambda_dx;
                continue;
            };


            line_drawn_to = true;
            out.discard = false;
            out.gl_FragColor = glm::vec4();
            // draw fragment
            
            Barycentrics fake = bary;
            create_fragment(mem, pos, fake, t, in);
    
            // stencil test
            if(!stencil_test_pass(mem, framebuffer, data_pos, front_facing)) { 
                bary.lambda += dlambda_dx;
                continue;
            }

            // depth test
            if(!depth_test_pass(mem, framebuffer, in, data_pos, front_facing)) {
                bary.lambda += dlambda_dx;
                continue;
            }

            // fragment shader
            if(prog->fragmentShader) {
                prog->fragmentShader(out, in, interface);
                if(out.discard) {
                    bary.lambda += dlambda_dx;
                    continue;
                }
            }

            // write dppass stencil
            if(framebuffer.stencil.data != nullptr && !mem.blockWrites.stencil && mem.stencilSettings.enabled) {
                std::uint8_t *stencil;
                stencil = reinterpret_cast<std::uint8_t*>(getPixel(framebuffer.stencil, pos.x, framebuffer.yReversed ? framebuffer.height - 1 - pos.y : pos.y));
                stencil_operation(mem, stencil, 
                        front_facing ?  mem.stencilSettings.frontOps.dppass :
                                        mem.stencilSettings.backOps.dppass);
            }
            // write depth
            if(framebuffer.depth.data != nullptr && !mem.blockWrites.depth) {
                write_frag_depth(framebuffer, data_pos, in.gl_FragCoord.z);
            }
            // write color
            if(framebuffer.color.data != nullptr && !mem.blockWrites.color) {
                get_blended_frag_color(mem, framebuffer, data_pos, out.gl_FragColor);
                write_frag_color(framebuffer, data_pos, out.gl_FragColor);
            }
            bary.lambda += dlambda_dx;
        }
    }
}

void draw(GPUMemory& mem, DrawCommand& cm) {
    ZoneScoped;
    Framebuffer& framebuffer = mem.framebuffers[mem.activatedFramebuffer];
    Primitive t;
    Primitive u;
    bool front_facing;

    Program *prog = mem.programs + mem.activatedProgram;
    for(std::int32_t attrib_idx = 0; attrib_idx < maxAttribs; attrib_idx++) {
        t.type[attrib_idx] = prog->vs2fs[attrib_idx];
        u.type[attrib_idx] = prog->vs2fs[attrib_idx];
    }

    // per triangle
    for(std::uint32_t vert_idx = 0; vert_idx < cm.nofVertices; vert_idx += 3) {
        primitive_assembly(mem, t, vert_idx);

        int primitives = clipping(t, u);
        if(primitives == 0) continue;

        // first triangle
        perspective_division(mem, t);
        viewport_transform(mem, framebuffer, t);
        if(should_cull(mem, t, front_facing)) continue;
        rasterize(mem, framebuffer, t, front_facing);

        if(primitives != 2) continue;
        // second triangle created while clipping
        perspective_division(mem, u);
        viewport_transform(mem, framebuffer, u);
        if(should_cull(mem, u, front_facing)) continue;
        rasterize(mem, framebuffer, u, front_facing);
    }

    mem.gl_DrawID++;
}

void decode_command(GPUMemory& mem, Command& command);

void sub(GPUMemory& mem, SubCommand& cm) {
    auto cb = cm.commandBuffer;
    for(std::uint32_t idx = 0; idx < cb->nofCommands; idx++) {
        auto command = cb->commands[idx];
        decode_command(mem, command);
    }
}

void decode_command(GPUMemory& mem, Command& command) {
    switch(command.type) {
        case CommandType::BIND_FRAMEBUFFER:
            bind_framebuffer(mem, command.data.bindFramebufferCommand);
            break;
        case CommandType::BIND_PROGRAM:
            bind_program(mem, command.data.bindProgramCommand);
            break;
        case CommandType::BIND_VERTEXARRAY:
            bind_vertex_array(mem, command.data.bindVertexArrayCommand);
            break;
        case CommandType::BLOCK_WRITES_COMMAND:
            block_writes(mem, command.data.blockWritesCommand);
            break;
        case CommandType::SET_BACKFACE_CULLING_COMMAND:
            set_backface_culling(mem, command.data.setBackfaceCullingCommand);
            break;
        case CommandType::SET_FRONT_FACE_COMMAND:
            set_front_face(mem, command.data.setFrontFaceCommand);
            break;
        case CommandType::SET_STENCIL_COMMAND:
            set_stencil(mem, command.data.setStencilCommand);
            break;
        case CommandType::SET_BLENDING_COMMAND:
            set_blending(mem, command.data.setBlendingCommand);
            break;
        case CommandType::SET_DRAW_ID:
            set_draw_id(mem, command.data.setDrawIdCommand);
            break;
        case CommandType::CLEAR_COLOR:
            clear_color(mem, command.data.clearColorCommand);
            break;
        case CommandType::CLEAR_DEPTH:
            clear_depth(mem, command.data.clearDepthCommand);
            break;
        case CommandType::CLEAR_STENCIL:
            clear_stencil(mem, command.data.clearStencilCommand);
            break;
        case CommandType::USER_COMMAND:
            user(mem, command.data.userCommand);
            break;
        case CommandType::DRAW:
            draw(mem, command.data.drawCommand);
            break;
        case CommandType::SUB_COMMAND:
            sub(mem, command.data.subCommand);
            break;
        case CommandType::EMPTY:
            break;
    }
}

//! [student_GPU_run]
void student_GPU_run(GPUMemory& mem, CommandBuffer const& cb) {
    /// \todo Tato funkce reprezentuje funkcionalitu grafické karty.<br>
    /// Měla by umět zpracovat command buffer, čistit framebuffer a kreslit.<br>
    /// mem obsahuje paměť grafické karty.
    /// cb obsahuje command buffer pro zpracování.
    /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.
    ///
    /// V základu jde o to, že cb obsahuje příkazy, které se musí provést nad pamětí mem.
    /// Správně fungující grafická karta dobře interpretuje příkazy v cb a správně změní obsah paměti mem.
    
    ZoneScoped;

    mem.gl_DrawID = 0;

    for(std::uint32_t idx = 0; idx < cb.nofCommands; idx++) {
        auto command = cb.commands[idx];
        decode_command(mem, command);
    }
}
//! [student_GPU_run]

