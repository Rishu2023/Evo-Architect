---
name: "🎯 Steer Evolution"
about: "Guide the CDLE evolution by suggesting architecture changes, objectives, or constraints"
title: "[STEER] "
labels: ["evolution-steering"]
assignees: []
---

## 🎯 Steering Directive

**What aspect of the CDLE would you like to steer?** (check one)

- [ ] Architecture change (e.g., adjust d_model, n_layers, d_state)
- [ ] New objective (e.g., prioritise memory efficiency, sparsity)
- [ ] Training change (e.g., curriculum, learning rate schedule)
- [ ] New benchmark task (e.g., add a specific evaluation dataset)
- [ ] Constraint modification (e.g., change max parameter count)
- [ ] Other (describe below)

## 📝 Description

<!-- Describe the change you'd like Agent 1 to consider in its next architecture proposal. -->

## 🎛️ Suggested Hyperparameters (optional)

<!-- If you have specific values in mind, list them here. -->

```yaml
model:
  d_model: 
  n_layers: 
  d_state: 
  fractal_levels: 
  ff_variant: 
  complexity_gate_threshold: 
training:
  learning_rate: 
  max_steps: 
  batch_size: 
```

## 🎯 Priority

- [ ] High — apply in the next generation
- [ ] Medium — consider over the next few generations
- [ ] Low — long-term exploration direction

## 📊 Expected Impact

<!-- What do you expect this change to improve? (loss, speed, generalization, etc.) -->

## 📎 References (optional)

<!-- Link any relevant papers, discussions, or benchmark results. -->
