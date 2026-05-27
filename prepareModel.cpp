/*!
 * @file
 * @brief This file contains functions for model rendering
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#include <studentSolution/prepareModel.hpp>
#include <studentSolution/gpu.hpp>
#include <solutionInterface/uniformLocations.hpp>
#include <iostream>
void process_node(GPUMemory& mem, CommandBuffer& cb, Node const& node, Model const& model, glm::mat4 const& parent_model, size_t& draw_counter) {

    glm::mat4 model_matrix = parent_model * node.modelMatrix;

    if(node.mesh >= 0) {
        Mesh mesh = model.meshes[node.mesh];

        VertexArray vao;
        vao.indexBufferID = mesh.indexBufferID;
        vao.indexOffset = mesh.indexOffset;
        vao.indexType = mesh.indexType;

        vao.vertexAttrib[0] = mesh.position;
        vao.vertexAttrib[1] = mesh.normal;
        vao.vertexAttrib[2] = mesh.texCoord;

        mem.vertexArrays[draw_counter] = vao;

        pushBindVertexArrayCommand(cb, draw_counter);
        pushSetBackfaceCullingCommand(cb, !mesh.doubleSided);
        pushDrawCommand(cb, mesh.nofIndices);

        glm::mat4 inverse_transpose_model = glm::inverse(glm::transpose(model_matrix));

        mem.uniforms[getUniformLocation(draw_counter, MODEL_MATRIX                  )].m4 = model_matrix;
        mem.uniforms[getUniformLocation(draw_counter, INVERSE_TRANSPOSE_MODEL_MATRIX)].m4 = inverse_transpose_model;
        mem.uniforms[getUniformLocation(draw_counter, DIFFUSE_COLOR                 )].v4 = mesh.diffuseColor;
        mem.uniforms[getUniformLocation(draw_counter, TEXTURE_ID                    )].i1 = mesh.diffuseTexture;
        mem.uniforms[getUniformLocation(draw_counter, DOUBLE_SIDED                  )].v1 = mesh.doubleSided;

        draw_counter++;
    }


    for(size_t child_idx = 0; child_idx < node.nofChildren; child_idx++) {
        process_node(mem, cb, node.children[child_idx], model, model_matrix, draw_counter);
    }
}

/**
 * @brief This function prepares model into memory and creates command buffer
 *
 * @param mem gpu memory
 * @param commandBuffer command buffer
 * @param model model structure
 */
//! [drawModel]
void student_prepareModel(GPUMemory& mem, CommandBuffer& commandBuffer, Model const& model){
    // Set buffers
    for(size_t buffer_idx = 0; buffer_idx < model.nofBuffers; buffer_idx++) {
        mem.buffers[buffer_idx] = model.buffers[buffer_idx];
    }

    // Set textures
    for(size_t texture_idx = 0; texture_idx < model.nofTextures; texture_idx++) {
        mem.textures[texture_idx] = model.textures[texture_idx];
    }


    // Iterate trough root nodes
    size_t draw_counter = 0;
    for(size_t root_idx = 0; root_idx < model.nofRoots; root_idx++) {
        Node& root = model.roots[root_idx];

        // Walk through the tree in preorder
        process_node(mem, commandBuffer, root, model, glm::mat4(1.0f), draw_counter);
    }
}
//! [drawModel]

/**
 * @brief This function represents vertex shader of texture rendering method.
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param si shader interface
 */
//! [drawModel_vs]
void student_drawModel_vertexShader(OutVertex& outVertex, InVertex const& inVertex, ShaderInterface const& si){
    const glm::mat4& model = si.uniforms[getUniformLocation(si.gl_DrawID, MODEL_MATRIX)].m4;
    const glm::mat4& inverse_transpose = si.uniforms[getUniformLocation(si.gl_DrawID, INVERSE_TRANSPOSE_MODEL_MATRIX)].m4;
    const glm::mat4& projection = si.uniforms[getUniformLocation(si.gl_DrawID, PROJECTION_VIEW_MATRIX)].m4;
    const glm::mat4& light = si.uniforms[getUniformLocation(si.gl_DrawID, USE_SHADOW_MAP_MATRIX)].m4;
    
    const glm::vec3& position = inVertex.attributes[0].v3;
    const glm::vec3& normal = inVertex.attributes[1].v3;
    const glm::vec2& uv = inVertex.attributes[2].v2;

    // position in world-space
    outVertex.attributes[0].v3 = model * glm::vec4(position, 1.0f);
    // normal in world-space
    outVertex.attributes[1].v3 = inverse_transpose * glm::vec4(normal, 0.0f);
    // texture coords (uv)
    outVertex.attributes[2].v2 = uv;
    // position in light clip-space
    outVertex.attributes[3].v4 = light * glm::vec4(outVertex.attributes[0].v3, 1.0f);

    // position
    outVertex.gl_Position = projection * model * glm::vec4(position, 1.0f);

}
//! [drawModel_vs]

#include<iostream>
#include<studentSolution/shaderFunctions.hpp>

/**
 * @brief This functionrepresents fragment shader outexture rendering method.
 *
 * @param outFragment output fragment
 * @param inFragment input fragment
 * @param si shader interface
 */
//! [drawModel_fs]
void student_drawModel_fragmentShader(OutFragment& outFragment, InFragment const& inFragment, ShaderInterface const& si){
    
    std::int32_t texture_id = si.uniforms[getUniformLocation(si.gl_DrawID, TEXTURE_ID)].i1;
    std::int32_t shadowmap_id = si.uniforms[getUniformLocation(si.gl_DrawID, SHADOWMAP_ID)].i1;

    std::int32_t double_sided = si.uniforms[getUniformLocation(si.gl_DrawID, DOUBLE_SIDED)].i1;

    glm::vec3 light_position = si.uniforms[getUniformLocation(si.gl_DrawID, LIGHT_POSITION)].v3;
    glm::vec3 camera_position = si.uniforms[getUniformLocation(si.gl_DrawID, CAMERA_POSITION)].v3;

    glm::vec3 position = inFragment.attributes[0].v3;
    glm::vec3 normal = inFragment.attributes[1].v3;
    glm::vec2 uv = inFragment.attributes[2].v2;
    glm::vec4 shadow_position = inFragment.attributes[3].v4;

    if(double_sided > 0 && glm::dot(camera_position - position, normal) < 0.0f) {
        // flip normal if looking from the back
        normal = -normal;
    }

    glm::vec3 ambient_color = si.uniforms[getUniformLocation(si.gl_DrawID, AMBIENT_LIGHT_COLOR)].v3;
    glm::vec3 light_color = si.uniforms[getUniformLocation(si.gl_DrawID, LIGHT_COLOR)].v3;
    glm::vec3 diffuse_color;
    glm::vec4 material_color;

    if(texture_id >= 0) {
        // has texture
        const Texture& tex = si.textures[texture_id];
        material_color = student_read_texture(tex, uv);
    }
    else {
        // no texture
        material_color = si.uniforms[getUniformLocation(si.gl_DrawID, DIFFUSE_COLOR)].v4;
    }

    outFragment.discard = material_color.a < 0.5;
    // discard fragment if it is too much transparent
    if(outFragment.discard) return;

    diffuse_color = material_color;

    normal = glm::normalize(normal);
        
    glm::vec3 ambient_light = diffuse_color * ambient_color;
    glm::vec3 diffuse_light = diffuse_color * light_color * glm::clamp(glm::dot(normal, glm::normalize(light_position - position)), 0.0f, 1.0f);
    if(shadowmap_id < 0) {
        // shadowmapping disabled
        outFragment.gl_FragColor = glm::vec4(ambient_light + diffuse_light, material_color.a); 
    }
    else {
        // shadowmapping enabled
        shadow_position /= shadow_position.w;
        if(shadow_position.x < 0 || shadow_position.y < 0 ||
           shadow_position.x > 1 || shadow_position.y > 1) {
            // not in shadowmap, render normaly
            outFragment.gl_FragColor = glm::vec4(ambient_light + diffuse_light, material_color.a); 
        }
        else {
            float shadow_depth = student_read_texture(si.textures[shadowmap_id], glm::vec2(shadow_position)).r;
            if(shadow_position.z > shadow_depth) {
                // in a shadow
                outFragment.gl_FragColor = glm::vec4(ambient_light, material_color.a); 
            }
            else {
                // not in a shadow
                outFragment.gl_FragColor = glm::vec4(ambient_light + diffuse_light, material_color.a); 
            } 
        }
    }
}
//! [drawModel_fs]

