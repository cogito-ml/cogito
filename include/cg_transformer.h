/**
 * Unified Transformer Block Primitive
 * 
 * Implements a complete Transformer block:
 * - Pre/Post-RMSNorm
 * - Multi-Head Attention (with Flash Attention)
 * - SwiGLU Feed Forward Network
 * - Residual Connections
 */

#ifndef CG_TRANSFORMER_H
#define CG_TRANSFORMER_H

#include "cg_layers.h"
#include "cg_tensor.h"
#include <stdbool.h>

/* Configuration for Transformer */
typedef struct {
    int hidden_dim;
    int num_heads;
    int num_kv_heads;       /* For GQA/MQA (if < num_heads) */
    int head_dim;
    int intermediate_dim;   /* MLP hidden dimension (e.g. 4*d or custom) */
    
    float norm_eps;
    float dropout_prob;
    bool use_bias;          /* Usually false for LLaMA */
    
    /* Architecture flags */
    bool pre_norm;           /* True = Norm->Attn->Res, False = Attn->Norm->Res (legacy) */
    bool use_rms_norm;       /* True = RMSNorm, False = LayerNorm */
    bool use_swiglu;         /* True = SwiGLU, False = Standard ReLU/GELU */
} cg_transformer_config;

typedef struct {
    cg_layer base;
    cg_transformer_config config;
    
    /* Sub-layers */
    cg_layer* norm1;        /* Attention Norm */
    cg_layer* attn_q_proj;
    cg_layer* attn_k_proj;
    cg_layer* attn_v_proj;
    cg_layer* attn_o_proj;
    
    cg_layer* norm2;        /* MLP Norm */
    cg_layer* mlp_gate;     /* SwiGLU Gate (if active) */
    cg_layer* mlp_up;       /* SwiGLU Value / MLP In */
    cg_layer* mlp_down;     /* MLP Out */
    cg_layer* act_fn;       /* Activation (if separate layer) */
    
    /* Dropout layers */
    cg_layer* drop1;
    cg_layer* drop2;
    
} cg_transformer_block;

cg_transformer_block* cg_transformer_block_new(cg_transformer_config config);

/* Standard forward pass */
cg_tensor* cg_transformer_forward(cg_layer* layer, cg_tensor* input, cg_tensor* mask);

#endif /* CG_TRANSFORMER_H */
