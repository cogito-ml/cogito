/**
 * Speculative Decoding - Draft model prediction with target verification
 */

#ifndef CG_SPECULATIVE_H
#define CG_SPECULATIVE_H

#include "cg_tensor.h"
#include <stdbool.h>

/*============================================================================
 * TOKEN TREE (for parallel verification)
 *============================================================================*/

typedef struct {
    int* tokens;                /* Draft tokens [gamma] */
    float* draft_probs;         /* Draft model probs [gamma, vocab] */
    float* target_probs;        /* Target model probs [gamma, vocab] */
    bool* accepted;             /* Acceptance mask [gamma] */
    cg_tensor* hidden_states;   /* Cached hidden states */
    
    int depth;                  /* Tree depth (gamma) */
    int* branch_points;         /* Where tree branches */
    int num_branches;
} cg_token_tree;

/*============================================================================
 * SPECULATIVE DECODER
 *============================================================================*/

typedef struct {
    /* Models */
    void* draft_model;          /* Small fast model */
    void* target_model;         /* Large accurate model */
    
    /* Configuration */
    int gamma;                  /* Draft tokens per step (5-8 typical) */
    float temperature;          /* Sampling temperature */
    float top_p;                /* Nucleus sampling threshold */
    int top_k;                  /* Top-k sampling */
    
    /* Acceptance */
    float accept_threshold;     /* Min acceptance probability */
    bool use_typical_sampling;  /* Use typical decoding */
    
    /* Statistics */
    int total_draft_tokens;
    int accepted_draft_tokens;
    float avg_acceptance_rate;
    int num_steps;
    
    /* KV cache management */
    void* draft_kv_cache;
    void* target_kv_cache;
} cg_speculative_decoder;

/*============================================================================
 * DECODING API
 *============================================================================*/

/**
 * Create speculative decoder.
 */
cg_speculative_decoder* cg_speculative_decoder_new(
    void* draft_model, void* target_model, int gamma
);

/**
 * Generate tokens speculatively.
 */
cg_tensor* cg_speculative_generate(
    cg_speculative_decoder* decoder,
    cg_tensor* input_ids,
    int max_new_tokens,
    float temperature,
    float top_p
);

/**
 * Single speculative step.
 */
int cg_speculative_step(
    cg_speculative_decoder* decoder,
    cg_tensor* input_ids,
    cg_token_tree* tree
);

/**
 * Draft model generates gamma tokens.
 */
cg_token_tree* cg_draft_generate(
    cg_speculative_decoder* decoder,
    cg_tensor* input_ids
);

/**
 * Target model verifies draft tokens.
 */
int cg_target_verify(
    cg_speculative_decoder* decoder,
    cg_tensor* input_ids,
    cg_token_tree* tree
);

void cg_speculative_decoder_free(cg_speculative_decoder* decoder);

/*============================================================================
 * TOKEN TREE API
 *============================================================================*/

cg_token_tree* cg_token_tree_new(int depth);
void cg_token_tree_add_token(cg_token_tree* tree, int token, float* probs);
cg_tensor* cg_token_tree_to_tensor(cg_token_tree* tree);
void cg_token_tree_free(cg_token_tree* tree);

/*============================================================================
 * TREE ATTENTION (for parallel verification)
 *============================================================================*/

/**
 * Generate tree attention mask.
 */
cg_tensor* cg_tree_attention_mask(cg_token_tree* tree, int seqlen);

/**
 * Forward with tree attention (verifies all branches in parallel).
 */
cg_tensor* cg_forward_tree_attention(
    void* model, cg_tensor* input, cg_tensor* tree_mask
);

/*============================================================================
 * EAGLE-STYLE DRAFT MODEL
 *============================================================================*/

typedef struct {
    void* base_draft;           /* Standard draft model */
    void* feature_proj;         /* Feature projection layer */
    void* uncertainty_head;     /* Predicts uncertainty */
    float* feature_cache;       /* Cached features */
} cg_eagle_draft;

/**
 * Create EAGLE-style draft model.
 */
cg_eagle_draft* cg_eagle_draft_new(void* base_draft);

/**
 * Generate tree with dynamic branching.
 */
cg_token_tree* cg_eagle_generate_tree(
    cg_eagle_draft* eagle,
    cg_tensor* input,
    int max_depth
);

/**
 * Compute branching factor based on uncertainty.
 */
int* cg_eagle_compute_branching(cg_eagle_draft* eagle, cg_tensor* features);

void cg_eagle_draft_free(cg_eagle_draft* eagle);

/*============================================================================
 * SAMPLING UTILITIES
 *============================================================================*/

/**
 * Sample from logits with temperature.
 */
int cg_sample_temperature(float* logits, int vocab_size, float temperature);

/**
 * Top-p (nucleus) sampling.
 */
int cg_sample_top_p(float* logits, int vocab_size, float top_p);

/**
 * Top-k sampling.
 */
int cg_sample_top_k(float* logits, int vocab_size, int top_k);

/**
 * Typical sampling (entropy-based).
 */
int cg_sample_typical(float* logits, int vocab_size, float typical_p);

/**
 * Compute acceptance probability for speculative sampling.
 */
float cg_acceptance_prob(float draft_prob, float target_prob);

/**
 * Sample corrected token when draft rejected.
 */
int cg_sample_corrected(float* draft_probs, float* target_probs, int vocab_size);

#endif /* CG_SPECULATIVE_H */
