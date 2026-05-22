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

        glm::mat4 inverse_transpose_model = glm::transpose(glm::inverse(model_matrix));

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

    // position world-space
    outVertex.attributes[0].v3 = model * glm::vec4(position, 1.0f);
    // normal world-space
    outVertex.attributes[1].v3 = inverse_transpose * glm::vec4(normal, 0.0f);
    // texture coords (uv)
    outVertex.attributes[2].v2 = uv;
    // position in light clip-space
    outVertex.attributes[3].v4 = light * glm::vec4(outVertex.attributes[0].v3, 1.0f);

    // position
    outVertex.gl_Position = projection * glm::vec4(outVertex.attributes[0].v3, 1.0f);

}
//! [drawModel_vs]

#include<iostream>
#include<studentSolution/shaderFunctions.hpp>

/**
 * @brief This functionrepresents fragment shader of texture rendering method.
 *
 * @param outFragment output fragment
 * @param inFragment input fragment
 * @param si shader interface
 */
//! [drawModel_fs]
void student_drawModel_fragmentShader(OutFragment& outFragment, InFragment const& inFragment, ShaderInterface const& si){
    (void)outFragment;
    (void)inFragment;
    (void)si;
    /// \todo Tato funkce reprezentujte fragment shader.<br>
    /// Vašim úkolem je správně obarvit fragmenty a osvětlit je pomocí lambertova osvětlovacího modelu.
    /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.
    
    std::uint32_t texture_id = si.uniforms[getUniformLocation(si.gl_DrawID, TEXTURE_ID)].i1;
    glm::vec2 uv = inFragment.attributes[2].v2;

    if(texture_id >= 0) {
        const Texture& tex = si.textures[texture_id];
        outFragment.gl_FragColor = student_read_texture(tex, uv);
    }
    else {
        outFragment.gl_FragColor = si.uniforms[getUniformLocation(si.gl_DrawID, DIFFUSE_COLOR)].v4;
    }
}
//! [drawModel_fs]

