/**
 * Loss Functions Implementation
 */

#include "cg_loss.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* MSE context */
typedef struct { cg_tensor* pred; cg_tensor* target; cg_reduction red; } mse_ctx;

static void mse_backward(cg_tensor* self) {
    mse_ctx* ctx = (mse_ctx*)self->backward_ctx;
    if (!ctx->pred->grad) return;
    float scale = (ctx->red == CG_REDUCTION_MEAN) ? 2.0f / ctx->pred->size : 2.0f;
    for (int i = 0; i < ctx->pred->size; i++)
        ctx->pred->grad[i] += scale * (ctx->pred->data[i] - ctx->target->data[i]) * self->grad[0];
}

cg_tensor* cg_mse_loss(cg_tensor* pred, cg_tensor* target, cg_reduction reduction) {
    int shape[] = {1};
    cg_tensor* loss = cg_tensor_new(shape, 1, pred->requires_grad);
    float sum = 0;
    for (int i = 0; i < pred->size; i++) {
        float d = pred->data[i] - target->data[i];
        sum += d * d;
    }
    loss->data[0] = (reduction == CG_REDUCTION_MEAN) ? sum / pred->size : sum;
    if (pred->requires_grad) {
        mse_ctx* ctx = malloc(sizeof(mse_ctx));
        ctx->pred = pred; ctx->target = target; ctx->red = reduction;
        loss->backward_ctx = ctx; loss->backward_fn = mse_backward;
        loss->parents[0] = pred; loss->num_parents = 1;
        cg_tensor_retain(pred);
    }
    return loss;
}

/* Cross-Entropy context */
typedef struct { cg_tensor* pred; cg_tensor* target; cg_tensor* sm; cg_reduction red; } ce_ctx;

static void ce_backward(cg_tensor* self) {
    ce_ctx* c = (ce_ctx*)self->backward_ctx;
    if (!c->pred->grad) return;
    int bs = c->pred->shape[0], nc = c->pred->shape[1];
    float scale = (c->red == CG_REDUCTION_MEAN) ? 1.0f/bs : 1.0f;
    for (int b = 0; b < bs; b++) {
        int lbl = (int)c->target->data[b];
        for (int j = 0; j < nc; j++) {
            float g = c->sm->data[b*nc+j] - (j==lbl?1:0);
            c->pred->grad[b*nc+j] += g * scale * self->grad[0];
        }
    }
}

cg_tensor* cg_cross_entropy_loss(cg_tensor* pred, cg_tensor* target, bool from_logits, cg_reduction red) {
    int bs = pred->shape[0], nc = pred->shape[1], shape[] = {1};
    cg_tensor* loss = cg_tensor_new(shape, 1, pred->requires_grad);
    cg_tensor* sm = cg_tensor_new(pred->shape, 2, false);
    for (int b = 0; b < bs; b++) {
        float mx = pred->data[b*nc], sum = 0;
        for (int j = 1; j < nc; j++) if (pred->data[b*nc+j] > mx) mx = pred->data[b*nc+j];
        for (int j = 0; j < nc; j++) { sm->data[b*nc+j] = expf(pred->data[b*nc+j]-mx); sum += sm->data[b*nc+j]; }
        for (int j = 0; j < nc; j++) sm->data[b*nc+j] /= sum;
    }
    float tot = 0;
    for (int b = 0; b < bs; b++) { int lbl = (int)target->data[b]; tot -= logf(fmaxf(sm->data[b*nc+lbl], 1e-7f)); }
    loss->data[0] = (red == CG_REDUCTION_MEAN) ? tot/bs : tot;
    if (pred->requires_grad) {
        ce_ctx* c = malloc(sizeof(ce_ctx));
        c->pred = pred; c->target = target; c->sm = sm; c->red = red;
        loss->backward_ctx = c; loss->backward_fn = ce_backward;
        loss->parents[0] = pred; loss->num_parents = 1; cg_tensor_retain(pred);
    } else cg_tensor_free(sm);
    return loss;
}

cg_tensor* cg_softmax_cross_entropy_loss(cg_tensor* logits, cg_tensor* target, cg_reduction red) {
    return cg_cross_entropy_loss(logits, target, true, red);
}

/* BCE */
typedef struct { cg_tensor* pred; cg_tensor* target; cg_reduction red; } bce_ctx;

static void bce_backward(cg_tensor* self) {
    bce_ctx* c = (bce_ctx*)self->backward_ctx;
    if (!c->pred->grad) return;
    float scale = (c->red == CG_REDUCTION_MEAN) ? 1.0f/c->pred->size : 1.0f;
    for (int i = 0; i < c->pred->size; i++) {
        float p = fmaxf(fminf(c->pred->data[i], 1-1e-7f), 1e-7f), t = c->target->data[i];
        c->pred->grad[i] += (-t/p + (1-t)/(1-p)) * scale * self->grad[0];
    }
}

cg_tensor* cg_bce_loss(cg_tensor* pred, cg_tensor* target, cg_reduction red) {
    int shape[] = {1};
    cg_tensor* loss = cg_tensor_new(shape, 1, pred->requires_grad);
    float tot = 0;
    for (int i = 0; i < pred->size; i++) {
        float p = fmaxf(fminf(pred->data[i], 1-1e-7f), 1e-7f), t = target->data[i];
        tot -= t*logf(p) + (1-t)*logf(1-p);
    }
    loss->data[0] = (red == CG_REDUCTION_MEAN) ? tot/pred->size : tot;
    if (pred->requires_grad) {
        bce_ctx* c = malloc(sizeof(bce_ctx));
        c->pred = pred; c->target = target; c->red = red;
        loss->backward_ctx = c; loss->backward_fn = bce_backward;
        loss->parents[0] = pred; loss->num_parents = 1; cg_tensor_retain(pred);
    }
    return loss;
}

/* L1 Context */
typedef struct { cg_tensor* pred; cg_tensor* target; cg_reduction red; } l1_ctx;

static void l1_backward(cg_tensor* self) {
    l1_ctx* c = (l1_ctx*)self->backward_ctx;
    if (!c->pred->grad) return;
    float scale = (c->red == CG_REDUCTION_MEAN) ? 1.0f/c->pred->size : 1.0f;
    for (int i = 0; i < c->pred->size; i++) {
        float diff = c->pred->data[i] - c->target->data[i];
        float grad = (diff > 0) ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
        c->pred->grad[i] += grad * scale * self->grad[0];
    }
}

cg_tensor* cg_l1_loss(cg_tensor* pred, cg_tensor* target, cg_reduction red) {
    int shape[] = {1};
    cg_tensor* loss = cg_tensor_new(shape, 1, pred->requires_grad);
    float sum = 0;
    for (int i = 0; i < pred->size; i++) sum += fabsf(pred->data[i] - target->data[i]);
    loss->data[0] = (red == CG_REDUCTION_MEAN) ? sum/pred->size : sum;
    if (pred->requires_grad) {
        l1_ctx* c = malloc(sizeof(l1_ctx));
        c->pred = pred; c->target = target; c->red = red;
        loss->backward_ctx = c; loss->backward_fn = l1_backward;
        loss->parents[0] = pred; loss->num_parents = 1; cg_tensor_retain(pred);
    }
    return loss;
}

/* BCE with logits Context */
typedef struct { cg_tensor* logits; cg_tensor* target; cg_reduction red; } bce_logits_ctx;

static void bce_logits_backward(cg_tensor* self) {
    bce_logits_ctx* c = (bce_logits_ctx*)self->backward_ctx;
    if (!c->logits->grad) return;
    float scale = (c->red == CG_REDUCTION_MEAN) ? 1.0f/c->logits->size : 1.0f;
    for (int i = 0; i < c->logits->size; i++) {
        // sigmoid(x) - target
        float s = 1.0f / (1.0f + expf(-c->logits->data[i]));
        c->logits->grad[i] += (s - c->target->data[i]) * scale * self->grad[0];
    }
}

cg_tensor* cg_bce_with_logits_loss(cg_tensor* logits, cg_tensor* target, cg_reduction red) {
    int shape[] = {1};
    cg_tensor* loss = cg_tensor_new(shape, 1, logits->requires_grad);
    float tot = 0;
    for (int i = 0; i < logits->size; i++) {
        float x = logits->data[i], t = target->data[i];
        // Stable BCE with logits: max(x,0) - x*t + log(1 + exp(-|x|))
        tot += fmaxf(x, 0.0f) - x * t + logf(1.0f + expf(-fabsf(x)));
    }
    loss->data[0] = (red == CG_REDUCTION_MEAN) ? tot/logits->size : tot;
    if (logits->requires_grad) {
        bce_logits_ctx* c = malloc(sizeof(bce_logits_ctx));
        c->logits = logits; c->target = target; c->red = red;
        loss->backward_ctx = c; loss->backward_fn = bce_logits_backward;
        loss->parents[0] = logits; loss->num_parents = 1; cg_tensor_retain(logits);
    }
    return loss;
}
