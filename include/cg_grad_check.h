/**
 * Numerical Gradient Checker
 * 
 * Validates analytical gradients using finite difference method.
 */

#ifndef CG_GRAD_CHECK_H
#define CG_GRAD_CHECK_H

#include "cg_tensor.h"
#include <stdbool.h>

/**
 * Check gradients for a function y = f(x).
 * 
 * @param forward_fn Function that computes output from input
 * @param input Input tensor to perturb
 * @param grad_output Gradient w.r.t output (usually 1.0)
 * @param epsilon Perturbation size (e.g. 1e-4)
 * @param tolerance Max allowed relative error (e.g. 1e-3)
 * @return true if check passes
 */
bool cg_grad_check(void (*forward_fn)(cg_tensor* in, cg_tensor* out),
                   cg_tensor* input,
                   cg_tensor* grad_output,
                   float epsilon,
                   float tolerance);

#endif /* CG_GRAD_CHECK_H */
