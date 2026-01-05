/**
 * Loss Functions for Cogito
 * 
 * All loss functions return a scalar tensor and support backward.
 */

#ifndef CG_LOSS_H
#define CG_LOSS_H

#include "cg_tensor.h"

/*============================================================================
 * REDUCTION MODES
 *============================================================================*/

typedef enum {
    CG_REDUCTION_NONE,             /* No reduction, return per-element loss */
    CG_REDUCTION_MEAN,             /* Return mean of all losses */
    CG_REDUCTION_SUM               /* Return sum of all losses */
} cg_reduction;

/*============================================================================
 * MEAN SQUARED ERROR LOSS
 *============================================================================*/

/**
 * Mean Squared Error loss.
 * 
 * L = (1/n) * sum((pred - target)^2)
 * 
 * @param pred Predictions [batch_size, ...]
 * @param target Targets [batch_size, ...]
 * @param reduction Reduction mode
 * @return Scalar loss tensor
 */
cg_tensor* cg_mse_loss(cg_tensor* pred, cg_tensor* target, cg_reduction reduction);

/*============================================================================
 * CROSS ENTROPY LOSS
 *============================================================================*/

/**
 * Cross Entropy loss for classification.
 * 
 * For hard labels (indices):
 *   L = -log(pred[target])
 * 
 * For soft labels (probabilities):
 *   L = -sum(target * log(pred))
 * 
 * @param pred Predictions (logits or probabilities) [batch_size, num_classes]
 * @param target Target class indices [batch_size] or probabilities [batch_size, num_classes]
 * @param from_logits If true, applies softmax to pred first
 * @param reduction Reduction mode
 * @return Scalar loss tensor
 */
cg_tensor* cg_cross_entropy_loss(cg_tensor* pred, cg_tensor* target,
                                  bool from_logits, cg_reduction reduction);

/**
 * Softmax Cross Entropy (combined for numerical stability).
 * 
 * More stable than separate softmax + cross_entropy.
 * Uses log-sum-exp trick.
 * 
 * @param logits Raw logits [batch_size, num_classes]
 * @param target Target class indices [batch_size]
 * @param reduction Reduction mode
 * @return Scalar loss tensor
 */
cg_tensor* cg_softmax_cross_entropy_loss(cg_tensor* logits, cg_tensor* target,
                                          cg_reduction reduction);

/*============================================================================
 * BINARY CROSS ENTROPY LOSS
 *============================================================================*/

/**
 * Binary Cross Entropy loss for binary classification.
 * 
 * L = -[target * log(pred) + (1 - target) * log(1 - pred)]
 * 
 * @param pred Predictions (probabilities in [0, 1]) [batch_size, ...]
 * @param target Binary targets [batch_size, ...]
 * @param reduction Reduction mode
 * @return Scalar loss tensor
 */
cg_tensor* cg_bce_loss(cg_tensor* pred, cg_tensor* target, cg_reduction reduction);

/**
 * Binary Cross Entropy with logits (numerically stable).
 * 
 * Applies sigmoid internally.
 */
cg_tensor* cg_bce_with_logits_loss(cg_tensor* logits, cg_tensor* target,
                                    cg_reduction reduction);

/*============================================================================
 * L1 LOSS (MAE)
 *============================================================================*/

/**
 * L1 / Mean Absolute Error loss.
 * 
 * L = (1/n) * sum(|pred - target|)
 */
cg_tensor* cg_l1_loss(cg_tensor* pred, cg_tensor* target, cg_reduction reduction);

/*============================================================================
 * SMOOTH L1 LOSS (HUBER)
 *============================================================================*/

/**
 * Smooth L1 / Huber loss.
 * 
 * L = 0.5 * x^2        if |x| < beta
 *   = |x| - 0.5*beta   otherwise
 */
cg_tensor* cg_smooth_l1_loss(cg_tensor* pred, cg_tensor* target,
                              float beta, cg_reduction reduction);

#endif /* CG_LOSS_H */
