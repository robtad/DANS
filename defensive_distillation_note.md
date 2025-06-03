**Distillation & Defensive Distillation Summary**

**Distillation** is a technique where a smaller (or equally large) student model learns to mimic a larger (or pretrained) teacher model by training on both:
- Defensive distillation (first introduced by Papernot et al. 2016) is designed to improve a model’s robustness against adversarial attacks by training a student model on the softened outputs (logits) of a teacher model rather than just the one-hot labels.



- **Hard labels** (e.g. class labels like cat/dog)
- **Soft targets** (teacher’s probability outputs, e.g. \[cat: 0.7, dog: 0.2, fox: 0.1]).

**Why?** The soft targets (teacher logits) contain richer information about class relationships than hard labels, improving generalization.

**Defensive distillation** is a variant of distillation originally proposed to make models more robust against adversarial attacks by smoothing the student’s decision boundaries using softened teacher outputs (logits).

**Implementation:**

- Teacher logits are softened using a temperature parameter, e.g.

  teacher_soft = softmax(teacher_logits / T)
  student_soft = log_softmax(student_logits / T)

- The student is trained to minimize:

  - KL divergence between its outputs and the teacher’s (soft loss)
  - plus standard cross-entropy with hard labels.

**Use cases:**

- Model compression (smaller, faster models)
- Defensive training (robustness)
- Transfer learning (leveraging a stronger teacher to improve a weaker student).

**note**
- accurary of the distilled(student) model is expected to be 
  slightly lower than the accuracy of the teacher model.(within 1-3%) 

**How does distillation help?**

- Softened logits (using temperature) from the teacher capture more information about class relationships (like “this is 70% ‘3’ but also 20% ‘5’”).

- The student trained on these logits learns a smoother decision boundary, making it harder for small perturbations to flip the decision.

Example scenario:
    - Teacher model (standard) is easily fooled by a small noise perturbation because its decision boundary is sharp.

    - Student model (distilled) has a smoother boundary — adversarial examples crafted for the teacher might not transfer easily to fool the student, and crafting them directly against the student requires larger (more obvious) perturbations.