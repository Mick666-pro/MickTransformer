# Task A — Debugging Report

## 1. Problem

When I was building my Transformer, I had an indentation problem in my Python training loop. The `optimizer.step()` line was not correctly placed inside the `if step < 200:` block.

## 2. Investigation

I checked my training code with ChatGPT. We looked at the indentation around `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.

I found that the indentation was inconsistent. This could cause the optimizer to update parameters at the wrong time.

## 3. Solution

I corrected the indentation so all three optimizer operations were inside the same `if` block.

```python
if step < 200:
    optimizer_transformer.zero_grad()
    train_loss.backward()
    optimizer_transformer.step()
```

## 4. Verification

After correcting the code, I ran my Transformer again.

My results were:

- Starting Loss: 2.3910
- Final Loss: 0.0039
- Generated text: `the cat sleeps.`
- Inference speedup: approximately 1.13×
- Predictions match: True

The program ran successfully, and the training Loss decreased.

## 5. What I Learned

I learned that indentation is important in Python because it controls which instructions belong inside a block.

I also learned that `zero_grad()` clears previous gradients, `backward()` calculates new gradients, and `step()` updates the model parameters.

When debugging, I should check the code structure and verify my changes by running the program again.