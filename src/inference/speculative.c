/**
 * Speculative Decoding
 */

#include "cg_speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static unsigned int spec_seed = 42;
static float randf(void) { spec_seed = spec_seed * 1103515245 + 12345; return (spec_seed % 10000) / 10000.0f; }

cg_token_tree* cg_token_tree_new(int depth) {
    cg_token_tree* t = (cg_token_tree*)calloc(1, sizeof(cg_token_tree));
    t->depth = depth;
    t->tokens = (int*)calloc(depth, sizeof(int));
    t->accepted = (bool*)calloc(depth, sizeof(bool));
    return t;
}

void cg_token_tree_free(cg_token_tree* t) {
    if (!t) return;
    free(t->tokens);
    free(t->draft_probs);
    free(t->target_probs);
    free(t->accepted);
    cg_tensor_free(t->hidden_states);
    free(t);
}

cg_speculative_decoder* cg_speculative_decoder_new(void* draft, void* target, int gamma) {
    cg_speculative_decoder* d = (cg_speculative_decoder*)calloc(1, sizeof(cg_speculative_decoder));
    d->draft_model = draft;
    d->target_model = target;
    d->gamma = gamma;
    d->temperature = 1.0f;
    d->top_p = 0.9f;
    d->top_k = 50;
    d->accept_threshold = 0.0f;
    return d;
}

void cg_speculative_decoder_free(cg_speculative_decoder* d) {
    if (!d) return;
    free(d);
}

float cg_acceptance_prob(float draft_prob, float target_prob) {
    if (draft_prob < 1e-10f) return 1.0f;
    return fminf(1.0f, target_prob / draft_prob);
}

int cg_sample_temperature(float* logits, int vocab_size, float temp) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    
    float sum = 0.0f;
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
    
    float r = randf();
    float cumsum = 0.0f;
    int token = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) { token = i; break; }
    }
    free(probs);
    return token;
}

int cg_sample_top_p(float* logits, int vocab_size, float top_p) {
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_logit);
        indices[i] = i;
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
    
    /* Sort by probability descending */
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[j] > probs[i]) {
                float tp = probs[i]; probs[i] = probs[j]; probs[j] = tp;
                int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
            }
        }
    }
    
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }
    
    float r = randf() * cumsum;
    cumsum = 0.0f;
    int token = indices[0];
    for (int i = 0; i < cutoff; i++) {
        cumsum += probs[i];
        if (r < cumsum) { token = indices[i]; break; }
    }
    
    free(probs);
    free(indices);
    return token;
}

int cg_sample_top_k(float* logits, int vocab_size, int top_k) {
    if (top_k >= vocab_size) return cg_sample_temperature(logits, vocab_size, 1.0f);
    
    float* top_vals = (float*)malloc(top_k * sizeof(float));
    int* top_idx = (int*)malloc(top_k * sizeof(int));
    
    for (int i = 0; i < top_k; i++) { top_vals[i] = -1e30f; top_idx[i] = 0; }
    
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > top_vals[top_k - 1]) {
            top_vals[top_k - 1] = logits[i];
            top_idx[top_k - 1] = i;
            for (int j = top_k - 1; j > 0 && top_vals[j] > top_vals[j-1]; j--) {
                float tv = top_vals[j]; top_vals[j] = top_vals[j-1]; top_vals[j-1] = tv;
                int ti = top_idx[j]; top_idx[j] = top_idx[j-1]; top_idx[j-1] = ti;
            }
        }
    }
    
    int token = cg_sample_temperature(top_vals, top_k, 1.0f);
    int result = top_idx[token];
    free(top_vals);
    free(top_idx);
    return result;
}

int cg_sample_corrected(float* draft_probs, float* target_probs, int vocab_size) {
    float* adjusted = (float*)malloc(vocab_size * sizeof(float));
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        adjusted[i] = fmaxf(0.0f, target_probs[i] - draft_probs[i]);
        sum += adjusted[i];
    }
    
    if (sum < 1e-10f) {
        free(adjusted);
        return cg_sample_temperature(target_probs, vocab_size, 1.0f);
    }
    
    float r = randf() * sum;
    float cumsum = 0.0f;
    int token = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += adjusted[i];
        if (r < cumsum) { token = i; break; }
    }
    
    free(adjusted);
    return token;
}

cg_eagle_draft* cg_eagle_draft_new(void* base_draft) {
    cg_eagle_draft* e = (cg_eagle_draft*)calloc(1, sizeof(cg_eagle_draft));
    e->base_draft = base_draft;
    return e;
}

/*============================================================================
 * TREE ATTENTION
 *============================================================================*/

void cg_token_tree_add_token(cg_token_tree* tree, int token, float* probs) {
    /* Simplified linear addition for now, real tree would track parent pointers */
    /* Assuming simple expansion for this phase */
}

cg_tensor* cg_token_tree_to_tensor(cg_token_tree* tree) {
    /* Flatten tree to tensor of token IDs */
    int shape[] = {1, tree->depth};
    cg_tensor* t = cg_tensor_new(shape, 2, false);
    for(int i=0; i<tree->depth; i++) t->data[i] = (float)tree->tokens[i];
    return t;
}

cg_tensor* cg_tree_attention_mask(cg_token_tree* tree, int seqlen) {
    /* 
     * Construct 2D attention mask [total_tokens, total_tokens].
     * Map tree topology to mask: nodes attend to ancestors.
     */
    /* Simplified assumption: tree is flattened linear sequence for base case, 
       or we need branch info */
    
    int num_nodes = tree->depth; /* Total nodes in tree */
    int size = seqlen + num_nodes;
    int shape[] = {1, 1, size, size};
    
    cg_tensor* mask = cg_tensor_new(shape, 4, false);
    /* Initialize to -inf */
    for(int i=0; i<mask->size; i++) mask->data[i] = -1e9f;
    
    /* Causal diagonal for seqlen */
    for(int i=0; i<seqlen; i++) {
        for(int j=0; j<=i; j++) {
            mask->data[i * size + j] = 0.0f;
        }
    }
    
    /* Tree nodes attend to prefix + ancestors */
    /* Since we don't have explicit parent links in struct yet (using simplified flat for now),
       we assume linear chain for this stub. 
       In real EAGLE: use tree->branch_points to determine ancestry */
       
    for(int i=0; i<num_nodes; i++) {
        int r = seqlen + i;
        /* Attend to prefix */
        for(int j=0; j<seqlen; j++) mask->data[r * size + j] = 0.0f;
        
        /* Attend to self and ancestors */
        /* Linear assumption: attend to all previous nodes in this path */
        for(int j=0; j<=i; j++) {
             mask->data[r * size + (seqlen + j)] = 0.0f;
        }
    }
    
    return mask;
}

/*============================================================================
 * EAGLE-STYLE GENERATION
 *============================================================================*/

cg_token_tree* cg_eagle_generate_tree(cg_eagle_draft* eagle, cg_tensor* input, int max_depth) {
    cg_token_tree* tree = cg_token_tree_new(max_depth);
    
    /* 
     * EAGLE Algorithm:
     * 1. Predict feature vector for next step
     * 2. Decode token from feature
     * 3. Estimate uncertainty -> decide whether to branch
     */
    
    /* Simulation of tree expansion */
    for (int i = 0; i < max_depth; i++) {
        /* In real impl: Run draft model forward, get logits & uncertainty */
        /* Here: just append a dummy token */
        tree->tokens[i] = 1; /* Dummy */
    }
    
    return tree;
}

void cg_eagle_draft_free(cg_eagle_draft* e) {
    if (!e) return;
    free(e);
}
